from data_constants import VIDEO_EMOTION_INSTRUCTIONS, REPLACE_LABEL_STRING, VIDEO_EXTENSIONS
from .raw_video_dataset import RawVideoDataset

import logging, os
from tqdm import tqdm
import pandas as pd


class CremaDVideo(RawVideoDataset):
    def __init__(self, dataset_name, dataset_path):
        self.task_type = "emotion"
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.has_audio = True
        if dataset_path.endswith("24fps"): ## checks if the dataset is already processed to 16khz
            self.dataset_24fps_path = self.dataset_path
        else:
            self.dataset_24fps_path = self.convert_to_24fps(self.dataset_path)

        if self.has_audio:
            self.dataset_16khz_path = self.convert_to_16khz(self.dataset_path)
        
        self.video_depth = 1

        self.classes_map = {"HAP":"happiness", "SAD":"sadness", "NEU": "neutral", "ANG": "anger", "SUR":"surprise", "DIS": "disgust", "FEA":"fear"}

        self.gemini_prompt = (f"The given video is labelled as having {REPLACE_LABEL_STRING} emotion. " 
                              f"Describe the properties of the audio visual aspects of the video which suggest that it is having {REPLACE_LABEL_STRING} emotion. Also, describe the facial expression in the video and their variation to support the emotion." 
                              "Detail the audio and/or verbal aspects which lead to the given emotion. DO NOT base your answer only on the transcript of the audio (if any speech is present). Include as much detail in your response as possible."
                              "Your response should be in json format as follows - {'reason':'reason for the label'}. The provided reason should not mention that you were provided with the ground truth label. You should first mention that the 'predicted emotion in the given video is ...' not in the same words.")
    
    def get_label_from_fname(self, fpath):
        video_id = fpath.split("/")[-1].split(".")[0]
        emotion_id = video_id.split("_")[-2]
        return self.classes_map[emotion_id]

    def get_gemini_instruction_format_data(self, gemini_predictions):

        gemini_predictions = {os.path.join(self.dataset_24fps_path, "/".join(k.split("/")[-self.video_depth:])):v for k, v in gemini_predictions.items()}

        all_files = []
        logging.info("Creating gemini instruction data for CREMAD-Video...")
        for r, ds, fs in os.walk(self.dataset_24fps_path):
            for f in fs:
                all_files.append((r, f))
        
        instr_options = len(VIDEO_EMOTION_INSTRUCTIONS)
        instr_data = []
        for idx, (r, f) in tqdm(enumerate(all_files), total=len(all_files)):
            try:
                cur_response = gemini_predictions[os.path.join(r, f)]
            except:
                continue
            cur_instr = VIDEO_EMOTION_INSTRUCTIONS[idx % instr_options]
            video_path = os.path.join(r, f)
            # Construct corresponding audio path
            rel_path = os.path.relpath(video_path, self.dataset_24fps_path)
            audio_path = os.path.join(self.dataset_16khz_path, rel_path.replace(".mp4", ".wav"))
            if not (os.path.exists(video_path) and os.path.exists(audio_path)):
                continue
            instr_data.append({
                "video_path": video_path,
                "audio_path": audio_path,
                "instruction": cur_instr,
                "response": cur_response
            })
        logging.info(f"Total number of instruction data: {len(instr_data)}")
        return instr_data




    
    


