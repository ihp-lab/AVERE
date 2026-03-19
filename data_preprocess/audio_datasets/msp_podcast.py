from data_constants import EMOTION_INSTRUCTIONS, EMOTION_PERCEPTION_INSTRUCTIONS, REPLACE_LABEL_STRING
from .raw_dataset import RawAudioDataset
from utils import convert_to_wav_parallel

import logging, os, glob, json
from tqdm import tqdm
import pandas as pd


class MSPPodcast(RawAudioDataset):
    def __init__(self, dataset_name, dataset_path):
        self.task_type = "emotion"
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        if dataset_path.endswith("16khz"): ## checks if the dataset is already processed to 16khz
            self.dataset_16khz_path = self.dataset_path
        else:
            self.dataset_16khz_path = self.convert_to_16khz(self.dataset_path)
        
        self.wav_depth = 2

        self.classes = ["N", "H", "S", "A", "D", "F", "U"]
        self.classes_map = {"H":"happiness", "A":"anger", "S":"sadness", "N":"neutral", "D":"disgust", "F":"fear", "U":"surprise"}

        ## read label metadata for all sessions
        label_data = pd.read_csv(os.path.join(self.dataset_path, "labels_consensus.csv"))
        self.label_dict = {}
        for _, row in label_data.iterrows():
            if row["EmoClass"] in ["X", "O", "C"]: # if emotion is contempt, other or there is no majority agreement, then skip the file 
                continue
            fname = row["FileName"]
            emotion = self.classes_map[row["EmoClass"]]
            self.label_dict[fname] = emotion

        self.gemini_prompt = (f"The given audio is labelled as having {REPLACE_LABEL_STRING} emotion. " 
                              f"Describe the properties of the audio which suggest that the given audio is having {REPLACE_LABEL_STRING} emotion. DO NOT base your answer only on the transcript of the audio. "
                              "Your response should be in json format as follows - {'reason':'reason for the label'}. The provided reason should not mention that you were provided with the ground truth label. You should first mention that the 'predicted emotion in the given audio is ...' not in the same words.")


    def get_label_from_fname(self, fpath):
        rel_fpath = fpath.split("/")[-1]
        return self.label_dict[rel_fpath]

    def get_gemini_instruction_format_data(self, gemini_predictions):

        gemini_predictions = {os.path.join(self.dataset_16khz_path, "/".join(k.split("/")[-self.wav_depth:])):v for k, v in gemini_predictions.items()}

        all_files = []
        logging.info(f"Creating gemini instruction data for {self.dataset_name}...")
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




    
    


