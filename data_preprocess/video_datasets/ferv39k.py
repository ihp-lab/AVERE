from data_constants import VIDEO_EMOTION_INSTRUCTIONS, REPLACE_LABEL_STRING, VIDEO_EXTENSIONS
from prompts_eqa import overall_captioning_prompt_video, qa_identification_prompt_video, \
    qa_visual_reasoning_prompt_video, \
    qa_temporal_variation_prompt_video, \
    qa_implicit_cause_reasoning_prompt_video
from .raw_video_dataset import RawVideoDataset

import logging, os
from tqdm import tqdm
import pandas as pd
import random


class FERV39k(RawVideoDataset):
    def __init__(self, dataset_name, dataset_path):
        self.task_type = "emotion"
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.has_audio = False
        if dataset_path.endswith("24fps"): ## checks if the dataset is already processed to 16khz
            self.dataset_24fps_path = self.dataset_path
        else:
            self.dataset_24fps_path = self.convert_to_24fps(self.dataset_path)
        
        self.video_depth = 3

        self.classes_map = {"Happy":"happiness", "Sad":"sadness", "Neutral": "neutral", "Angry": "anger", "Surprise":"surprise", "Disgust": "disgust", "Fear":"fear"}

        self.gemini_prompt = (f"The given video is labelled as having {REPLACE_LABEL_STRING} emotion. " 
                              f"Describe the properties of the visual aspects of the video which suggest that it is having {REPLACE_LABEL_STRING} emotion. Also, describe the facial expression in the video and their variation to support the emotion." 
                              "Include as much detail in your response as possible."
                              "Your response should be in json format as follows - {'reason':'reason for the label'}. The provided reason should not mention that you were provided with the ground truth label. You should first mention that the 'predicted emotion in the given video is ...' not in the same words.")
        self.gemini_qa_prompts = [qa_identification_prompt_video, qa_visual_reasoning_prompt_video,
                                  qa_temporal_variation_prompt_video,
                                  qa_implicit_cause_reasoning_prompt_video]
        self.gemini_overall_captioning_prompt = overall_captioning_prompt_video

        self.valid_qa_categories = ["primary", "open_vocabulary", "valence_arousal", "intensity",  ##identification
                                   "facial_expression_reasoning", "body_language_reasoning", "visual_context_reasoning", "implicit_cause_reasoning", ## visual reasoning
                                   "temporal_variation_identification", "temporal_variation_reasoning", "transient_sustained_emotion_identification", ## temporal variation
                                   "modality_agreement_identification", "modality_agreement_reasoning", "modality_saliency_identification", "modality_saliency_reasoning"] ## modality agreement/saliency
    
    def get_label_from_fname(self, fpath):
        emotion_raw = fpath.split("/")[-2]
        return self.classes_map[emotion_raw]

    def get_gemini_instruction_format_data(self, gemini_predictions, gemini_mode = "description"):

        gemini_predictions = {os.path.join(self.dataset_24fps_path, "/".join(k.split("/")[-self.video_depth:])):v for k, v in gemini_predictions.items()}

        all_files = []
        logging.info(f"Creating gemini instruction data for {self.dataset_name}...")
        for r, ds, fs in os.walk(self.dataset_24fps_path):
            for f in fs:
                all_files.append((r, f))
        
        
        instr_data = []
        missing_files = 0
        if gemini_mode == "description":
            instr_options = len(VIDEO_EMOTION_INSTRUCTIONS)
            for idx, (r, f) in tqdm(enumerate(all_files), total=len(all_files)):
                try:
                    cur_response = gemini_predictions[os.path.join(r, f)]
                except:
                    missing_files += 1
                    continue
                cur_instr = VIDEO_EMOTION_INSTRUCTIONS[idx % instr_options]
                video_path = os.path.join(r, f)
                # Construct corresponding audio path
                rel_path = os.path.relpath(video_path, self.dataset_24fps_path)
                if not os.path.exists(video_path):
                    continue
                instr_data.append({
                    "video_path": video_path,
                    "instruction": cur_instr,
                    "response": cur_response
                })
        elif gemini_mode == "qa":
            missing_cat = 0
            for idx, (r, f) in tqdm(enumerate(all_files), total=len(all_files)):
                try:
                    cur_response = gemini_predictions[os.path.join(r, f)]
                except:
                    missing_files += 1
                    continue
                for question in cur_response:
                    cur_instr = question["question"]
                    cur_instr_mcq = question["question"] + " " + " ".join(question["choices"])
                    if "category" not in question:
                        logging.warning(f"Category not found in question for {os.path.join(r, f)}")
                        missing_cat += 1
                        continue
                    category = question["category"]
                    if category not in self.valid_qa_categories:
                        continue
                    video_path = os.path.join(r, f)
                    rel_path = os.path.relpath(video_path, self.dataset_24fps_path)
                    if not os.path.exists(video_path):
                        continue
                    if random.random() > 0.5:
                        instr_data.append({
                            "video_path": video_path,
                            "instruction": "<video>\n" + cur_instr,
                            "response": question["answer"]["text"],
                            "category": category
                        })
                    else:
                        instr_data.append({
                            "video_path": video_path,
                            "instruction": "<video>\n" + cur_instr_mcq,
                            "response": f"({question['answer']['choice']}). {question['answer']['text']}",
                            "category": category
                        })
            logging.info(f"Missing categories in the questions: {missing_cat}")
        logging.info(f"Missing files in the dataset: {missing_files}/{len(all_files)}")
        return instr_data




    
    


