from data_constants import VIDEO_EMOTION_INSTRUCTIONS, REPLACE_LABEL_STRING, VIDEO_EXTENSIONS, VIDEO_EMOTION_INSTRUCTIONS
from .raw_video_dataset import RawVideoDataset

import logging, os
from tqdm import tqdm
import pandas as pd


class MER2025Track3Desc(RawVideoDataset):
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

        mer_cap_plus = pd.read_csv(f"{self.dataset_path}/../track3_train_mercaptionplus.csv")
        mer_ovmerd = pd.read_csv(f"{self.dataset_path}/../track3_train_ovmerd.csv")

        self.label_dict = {}
        for idx, row in mer_cap_plus.iterrows():
            video_id = str(row["name"])
            cur_desc = row["reason"]
            self.label_dict[video_id] = cur_desc
        for idx, row in mer_ovmerd.iterrows():
            video_id = str(row["name"])
            cur_desc = row["reason"]
            self.label_dict[video_id] = cur_desc


        self.gemini_prompt = (f"The given video is labelled as having {REPLACE_LABEL_STRING} emotion. " 
                              f"Describe the properties of the audio visual aspects of the video which suggest that it is having {REPLACE_LABEL_STRING} emotion. Also, describe the facial expression in the video and their variation to support the emotion." 
                              "Detail the audio and/or verbal aspects which lead to the given emotion. DO NOT base your answer only on the transcript of the audio (if any speech is present). Include as much detail in your response as possible."
                              "Your response should be in json format as follows - {'reason':'reason for the label'}. The provided reason should not mention that you were provided with the ground truth label. You should first mention that the 'predicted emotion in the given video is ...' not in the same words.")
    
    def get_label_from_fname(self, fpath):
        video_id = fpath.split("/")[-1].split(".")[0]
        if video_id not in self.label_dict:
            raise Exception(f"Label not found for {video_id}")
        return self.label_dict[video_id]

    def get_naive_instruction_format_data(self):
        
        instructions = VIDEO_EMOTION_INSTRUCTIONS

        all_files = []
        logging.info(f"Creating naive instruction format data for {self.dataset_name}...")
        for r, ds, fs in os.walk(self.dataset_24fps_path):
            for f in fs:
                all_files.append((r, f))
        
        instr_options = len(instructions)
        instr_data = []
        num_missing_labels = 0
        for idx, (r, f) in tqdm(enumerate(all_files), total = len(all_files)):
            try:
                cur_desc = self.get_label_from_fname(os.path.join(r,f))
            except:
                num_missing_labels+=1
                continue
            cur_instr = instructions[idx%instr_options]
            instance = {
                "video_path": os.path.join(r, f),
                "instruction": cur_instr,
                "response": cur_desc
            }
            if getattr(self, "has_audio", False):
                # Construct audio path based on dataset_16khz_path and relative path from dataset_24fps_path
                rel_path = os.path.relpath(os.path.join(r, f), self.dataset_24fps_path)
                audio_path = os.path.join(self.dataset_16khz_path, rel_path.replace(".mp4", ".wav"))
                instance["audio_path"] = audio_path
                if not os.path.exists(audio_path):
                    num_missing_labels+=1
                    continue
            if not os.path.exists(instance["video_path"]):
                num_missing_labels+=1
                continue
            instr_data.append(instance)
        logging.info(f"Created naive instruction data. {num_missing_labels}/{len(all_files)} files had missing labels and/or missing audio/video.")
        return instr_data

    def get_gemini_instruction_format_data(self, gemini_predictions):

        gemini_predictions = {os.path.join(self.dataset_24fps_path, "/".join(k.split("/")[-self.video_depth:])):v for k, v in gemini_predictions.items()}

        all_files = []
        logging.info("Creating gemini instruction data for MER 2025 Track 1 - Single Label...")
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
        return instr_data




    
    


