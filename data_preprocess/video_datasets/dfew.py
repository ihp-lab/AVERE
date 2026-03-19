from data_constants import VIDEO_EMOTION_INSTRUCTIONS, REPLACE_LABEL_STRING, VIDEO_EXTENSIONS
from prompts_eqa import overall_captioning_prompt_video_audio, qa_identification_prompt_video_audio, \
    qa_visual_reasoning_prompt_video_audio, qa_audio_reasoning_prompt_video_audio, \
    qa_temporal_variation_prompt_video_audio, qa_modality_agreement_prompt_video_audio, \
    qa_implicit_cause_reasoning_prompt_video_audio
from utils import get_res_from_jsonl_gpt
from .raw_video_dataset import RawVideoDataset

import logging, os
from tqdm import tqdm
import pandas as pd
import json


class DFEW(RawVideoDataset):
    def __init__(self, dataset_name, dataset_path):
        # super().__init__(dataset_name, dataset_path)
        self.task_type = "emotion"
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.has_audio = True
        if dataset_path.endswith("24fps"): ## checks if the dataset is already processed to 16khz
            self.dataset_24fps_path = self.dataset_path
        else:
            self.dataset_24fps_path = self.convert_to_24fps(self.dataset_path, with_audio=True)
            self.dataset_24fps_path_no_audio = self.convert_to_24fps(self.dataset_path, with_audio=False)
            self.dataset_8_frame_path = self.convert_to_8_frames(self.dataset_path)

        if self.has_audio:
            self.dataset_16khz_path = self.convert_to_16khz(self.dataset_path)
        
        self.video_depth = 2

        dfew_df = pd.read_excel(f"{self.dataset_path}/../dfew_annotation/annotation.xlsx", sheet_name="Sheet1")
        self.classes_map = {1:"happiness", 2:"sadness", 3: "neutral", 4: "anger", 5:"surprise", 6: "disgust", 7:"fear"}

        self.label_dict = {}
        for idx, row in dfew_df.iterrows():
            video_id = str(row["order"])
            cur_label = row["label"]
            if cur_label == 0:
                continue
            self.label_dict[video_id] = self.classes_map[cur_label]

        self.gemini_prompt = (f"The given video is labelled as having {REPLACE_LABEL_STRING} emotion. " 
                              f"Describe the properties of the audio visual aspects of the video which suggest that it is having {REPLACE_LABEL_STRING} emotion. Also, describe the facial expression in the video and their variation to support the emotion." 
                              "Detail the audio and/or verbal aspects which lead to the given emotion. DO NOT base your answer only on the transcript of the audio (if any speech is present). Include as much detail in your response as possible."
                              "Your response should be in json format as follows - {'reason':'reason for the label'}. The provided reason should not mention that you were provided with the ground truth label. You should first mention that the 'predicted emotion in the given video is ...' not in the same words.")

        self.gemini_qa_prompts = [qa_identification_prompt_video_audio, qa_visual_reasoning_prompt_video_audio,
                                  qa_audio_reasoning_prompt_video_audio, qa_temporal_variation_prompt_video_audio,
                                  qa_modality_agreement_prompt_video_audio, qa_implicit_cause_reasoning_prompt_video_audio]
        self.gemini_overall_captioning_prompt = overall_captioning_prompt_video_audio

        self.valid_qa_categories = ["primary", "open_vocabulary", "valence_arousal", "intensity",  ##identification
                                   "facial_expression_reasoning", "body_language_reasoning", "visual_context_reasoning", "implicit_cause_reasoning", ## visual reasoning
                                   "semantic_speech_reasoning", "paralinguistic_speech_reasoning", "audio_context_reasoning", ## audio reasoning
                                   "temporal_variation_identification", "temporal_variation_reasoning", "transient_sustained_emotion_identification", ## temporal variation
                                #    "modality_agreement_identification", "modality_agreement_reasoning", "modality_saliency_identification", "modality_saliency_reasoning", ## modality agreement/saliency
                                   "modality_agreement", "modality_saliency", ## modality agreement/saliency
                                   "vision_induced_hallucination", "audio_induced_hallucination"] ## hallucination

        hallucination_categories = ["audio_driven_visual_hallucination_emotion_relevant", "audio_driven_visual_hallucination_video_relevant",
                            "video_driven_visual_hallucination_video_relevant", "video_driven_visual_no_hallucination", "video_driven_visual_hallucination_emotion_relevant",
                            "video_driven_audio_hallucination_emotion_relevant", "video_driven_audio_hallucination_video_relevant",
                            "audio_driven_audio_hallucination_audio_relevant", "audio_driven_audio_no_hallucination", "audio_driven_audio_hallucination_emotion_relevant"]
        self.valid_qa_categories.extend(hallucination_categories)

    def get_label_from_fname(self, fpath):
        video_id = fpath.split("/")[-1].split(".")[0]
        if video_id not in self.label_dict:
            raise Exception(f"Label not found for {video_id}")
        return self.label_dict[video_id]

    def save_modality_emotion_prediction_results(self, pred_dict, save_path):
        # pred_dict_video = pred_dict[0]
        gpt_video_caption_prediction_path = "/wekafs/ict/achaubey/emotion_reasoning/audio_exp/data/gpt_annotations/jsonl_files/dfew_modality_separate_caption_predictions.jsonl"
        if not os.path.exists(gpt_video_caption_prediction_path):
            raise FileNotFoundError(f"{gpt_video_caption_prediction_path} not found")
        pred_dict_video = get_res_from_jsonl_gpt(gpt_video_caption_prediction_path, annotation_mode = "predict_emotion_from_video_captions")
        print("Pred dict video keys ==>", list(pred_dict_video.keys())[:10])
        pred_dict_audio = pred_dict[1]
        correct_labels_video = []
        correct_labels_audio = []
        corr_neutral_video, corr_neutral_audio, corr_neutral_both = 0, 0, 0
        for video_id, pred in pred_dict_video.items():
            if self.get_label_from_fname(video_id) == pred:
                if self.get_label_from_fname(video_id) == "neutral":
                    corr_neutral_video += 1
                correct_labels_video.append(video_id)
        for video_id, pred in pred_dict_audio.items():
            if self.get_label_from_fname(video_id) == pred:
                if self.get_label_from_fname(video_id) == "neutral":
                    corr_neutral_audio += 1
                correct_labels_audio.append(video_id)
        logging.info(f"Correct labels (video): {len(correct_labels_video)}/ {len(pred_dict_video)}")
        logging.info(f"Correct labels (audio): {len(correct_labels_audio)}/ {len(pred_dict_audio)}")
        logging.info(f"Neutral videos out of video predictions: {corr_neutral_video}/{len(correct_labels_video)}| audio predictions: {corr_neutral_audio}/{len(correct_labels_audio)}")
        logging.info(f"Saving video predictions to {save_path}/{self.dataset_name}_video.json")
        with open(f"{save_path}/{self.dataset_name}_video.json", "w") as f:
            json.dump(correct_labels_video, f, indent=4)
        logging.info(f"Saving audio predictions to {save_path}/{self.dataset_name}_audio.json")
        with open(f"{save_path}/{self.dataset_name}_audio.json", "w") as f:
            json.dump(correct_labels_audio, f, indent=4)
        ## save the video ids where both the audio and video predictions are correct
        both_correct = set(correct_labels_video) & set(correct_labels_audio)
        both_correct = list(both_correct)
        logging.info(f"Correct predictions in both audio and video: {len(both_correct)}/{len(pred_dict_video)}")
        logging.info(f"Saving both correct predictions to {save_path}/{self.dataset_name}_audio_video.json")
        with open(f"{save_path}/{self.dataset_name}_audio_video.json", "w") as f:
            json.dump(both_correct, f, indent=4)

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
                audio_path = os.path.join(self.dataset_16khz_path, rel_path.replace(".mp4", ".wav"))
                if not (os.path.exists(video_path) and os.path.exists(audio_path)):
                    continue
                instr_data.append({
                    "video_path": video_path,
                    "audio_path": audio_path,
                    "instruction": cur_instr,
                    "response": cur_response
                })
        elif gemini_mode in ["qa", "modality_qa"]:
            missing_cat = 0
            for idx, (r, f) in tqdm(enumerate(all_files), total=len(all_files)):
                try:
                    cur_response = gemini_predictions[os.path.join(r, f)]
                except:
                    missing_files += 1
                    continue
                for question in cur_response:
                    if "question" not in question or "answer" not in question:
                        logging.warning(f"Question or answer not found in response for {os.path.join(r, f)}")
                        missing_cat += 1
                        continue
                    cur_instr = question["question"]
                    if "choices" not in question:
                        logging.warning(f"Choices not found in question for {os.path.join(r, f)}")
                        missing_cat += 1
                        continue
                    cur_instr_mcq = question["question"] + " " + " ".join(question["choices"])
                    if "category" not in question:
                        logging.warning(f"Category not found in question for {os.path.join(r, f)}")
                        missing_cat += 1
                        continue
                    category = question["category"]
                    if category not in self.valid_qa_categories:
                        continue
                    video_path = os.path.join(r, f)
                    # Construct corresponding audio path
                    rel_path = os.path.relpath(video_path, self.dataset_24fps_path)
                    audio_path = os.path.join(self.dataset_16khz_path, rel_path.replace(".mp4", ".wav"))
                    if not (os.path.exists(video_path) and os.path.exists(audio_path)):
                        continue
                    instr_data.append({
                        "video_path": video_path,
                        "audio_path": audio_path,
                        "instruction": "<video>\n<audio>\n" + cur_instr,
                        "response": question["answer"]["text"],
                        "category": category
                    })
                    instr_data.append({
                        "video_path": video_path,
                        "audio_path": audio_path,
                        "instruction": "<video>\n<audio>\n" + cur_instr_mcq,
                        "response": question["answer"]["choice"],
                        "category": category
                    })
            logging.info(f"Missing categories in the questions: {missing_cat}")
        logging.info(f"Missing files in the dataset: {missing_files}/{len(all_files)}")
        return instr_data




    
    


