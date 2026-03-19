from data_constants import EMOTION_INSTRUCTIONS, REPLACE_LABEL_STRING
from .raw_dataset import RawAudioDataset

import logging, os
from tqdm import tqdm


class TESS(RawAudioDataset):
    def __init__(self, dataset_name, dataset_path):
        self.task_type = "emotion"
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        if dataset_path.endswith("16khz"): ## checks if the dataset is already processed to 16khz
            self.dataset_16khz_path = self.dataset_path
        else:
            self.dataset_16khz_path = self.convert_to_16khz(self.dataset_path)
        
        self.wav_depth = 4

        self.classes = ["angry", "disgust", "fear", "happy", "neutral", "ps", "sad"]
        self.classes_map = {"happy":"happiness", "angry":"anger", "disgust":"disgust", "fear":"fear", "neutral":"neutral", "ps":"surprise", "sad":"sadness"}
        self.gemini_prompt = (f"The given audio is labelled as having {REPLACE_LABEL_STRING} emotion. " 
                              f"Describe the properties of the audio which suggest that the given audio is having {REPLACE_LABEL_STRING} emotion. DO NOT base your answer only on the transcript of the audio. "
                              "Your response should be in json format as follows - {'reason':'reason for the label'}. The provided reason should not mention that you were provided with the ground truth label. You should first mention that the 'predicted emotion in the given audio is ...' not in the same words.")
    
    def get_label_from_fname(self, fpath):
        fname = fpath.split("/")[-1].split(".")[0]
        return self.classes_map[fname.split("_")[-1].lower()]

    def get_gemini_instruction_format_data(self, gemini_predictions):

        gemini_predictions = {os.path.join(self.dataset_16khz_path, "/".join(k.split("/")[-self.wav_depth:])):v for k, v in gemini_predictions.items()}

        all_files = []
        logging.info("Creating gemini instruction data for TESS...")
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




    
    


