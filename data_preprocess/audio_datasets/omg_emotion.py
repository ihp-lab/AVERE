from data_constants import EMOTION_INSTRUCTIONS, REPLACE_LABEL_STRING
from .raw_dataset import RawAudioDataset
from utils import convert_to_wav_parallel

import logging, os, glob, json
from tqdm import tqdm
import pandas as pd


class OMGEmotion(RawAudioDataset):
    def __init__(self, dataset_name, dataset_path):
        self.task_type = "emotion"
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        if dataset_path.endswith("16khz"): ## checks if the dataset is already processed to 16khz
            self.dataset_16khz_path = self.dataset_path
        else:
            self.dataset_16khz_path = self.convert_to_16khz(self.dataset_path)
        
        self.wav_depth = 3

        self.classes = [0, 1, 2, 3, 4, 5, 6]
        self.classes_map = {3:"happiness", 0:"anger", 5:"sadness", 4:"neutral", 1: "disgust", 2:"fear", 6:"surprise"}

        self.label_dict = {}
        metadata_files = ["omg_TrainVideos.csv", "omg_ValidationVideos.csv", "omg_TestVideos_WithLabels.csv"]
        metadata_files = [os.path.join(self.dataset_path, mdf) for mdf in metadata_files]
        for mdf in metadata_files:
            cur_md = pd.read_csv(mdf)
            for _, row in cur_md.iterrows():
                vid = row["video"]
                utt = row["utterance"][:-4]+".wav"
                emotion = self.classes_map[int(row["EmotionMaxVote"])]
                self.label_dict[f"{vid}/{utt}"] = emotion


        self.gemini_prompt = (f"The given audio is labelled as having {REPLACE_LABEL_STRING} emotion. " 
                              f"Describe the properties of the audio which suggest that the given audio is having {REPLACE_LABEL_STRING} emotion. DO NOT base your answer only on the transcript of the audio. "
                              "Your response should be in json format as follows - {'reason':'reason for the label'}. The provided reason should not mention that you were provided with the ground truth label. You should first mention that the 'predicted emotion in the given audio is ...' not in the same words.")
    
    def convert_to_16khz(self, dataset_path):

        dataset_16khz_path = dataset_path+"_16khz"
        if os.path.exists(dataset_16khz_path):
            logging.warning(f"Dataset conversion path to 16khz exists at {dataset_16khz_path}. Already converted files will be skipped.")
            # return dataset_16khz_path
        logging.info(f"Reading all the wav files inside the dataset directory - {dataset_path}")

        ## we have to reorganize before converting to 16khz

        all_relevant_audios = glob.glob(f"{dataset_path}/data/*/*.mp4")
        all_files = []
        for audio in all_relevant_audios:
            if audio[0]==".":
                continue
            all_files.append((os.path.dirname(audio), os.path.basename(audio)))
        
        # converting the files to 16khz
        logging.info(f"Converting all the wav files inside the dataset directory - {dataset_path}")
        convert_to_wav_parallel(all_files, dataset_path, dataset_16khz_path)
        logging.info(f"Conversion complete. Saving files inside the directory - {dataset_16khz_path}")
        return dataset_16khz_path


    def get_label_from_fname(self, fpath):
        rel_fpath = "/".join(fpath.split("/")[-2:])
        return self.label_dict[rel_fpath]

    def get_gemini_instruction_format_data(self, gemini_predictions):

        gemini_predictions = {os.path.join(self.dataset_16khz_path, "/".join(k.split("/")[-self.wav_depth:])):v for k, v in gemini_predictions.items()}

        all_files = []
        logging.info("Creating gemini instruction data for Emov-DB...")
        for r, ds, fs in os.walk(self.dataset_16khz_path):
            for f in fs:
                all_files.append((r, f))
        
        instr_options = len(EMOTION_INSTRUCTIONS)
        instr_data = []
        for idx, (r, f) in tqdm(enumerate(all_files), total = len(all_files)):
            try:
                cur_response = gemini_predictions[os.path.join(r, f)]
            except:
                continue
            cur_instr = EMOTION_INSTRUCTIONS[idx%instr_options]
            instr_data.append({
                "audio_path":os.path.join(r, f),
                "instruction":cur_instr,
                "response":cur_response
            })
        return instr_data




    
    


