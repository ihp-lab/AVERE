from data_constants import (VIDEO_EXTENSIONS, BUCKET_NAME, REPLACE_LABEL_STRING, BUCKET_AUDIO_FOLDER, REPLACE_CAPTION_STRING, REPLACE_VIDEO_ID_STRING,
                            GEMINI_PRICE, EMOTION_PERCEPTION_INSTRUCTIONS, VIDEO_EMOTION_PERCEPTION_INSTRUCTIONS, AGE_PERCEPTION_INSTRUCTIONS,
                            GENDER_PERCEPTION_INSTRUCTIONS, REPLACE_AUDIO_CAPTION_STRING, REPLACE_VIDEO_CAPTION_STRING)
from utils import (extra_path, google_cloud_upload_directory, google_cloud_directory_exists, get_res_from_jsonl, extract_frames_parallel,
                    convert_to_24fps_parallel, total_wav_duration, total_video_duration, convert_to_wav_parallel, convert_to_wav_if_audio_stream, 
                    get_res_from_jsonl_gpt)

from prompts_eqa import audio_video_modality_agreement_prompt, audio_video_modality_hallucination_prompt, audio_video_modality_identification_prompt, \
    audio_video_modality_visual_reasoning_prompt, audio_video_modality_audio_reasoning_prompt, audio_video_modality_implicit_cause_reasoning_prompt, \
        audio_modality_emotion_prediction_prompt, video_modality_emotion_prediction_prompt, audio_emotion_driven_visual_hallucination_emotion_relevant_prompt, \
            audio_emotion_driven_visual_hallucination_video_relevant_prompt, video_emotion_driven_visual_no_hallucination_prompt, \
                video_emotion_driven_visual_hallucination_video_relevant_prompt, video_emotion_driven_audio_hallucination_emotion_relevant_prompt, \
                    video_emotion_driven_audio_hallucination_audio_relevant_prompt, audio_emotion_driven_audio_hallucination_audio_relevant_prompt, \
                        audio_emotion_driven_audio_no_hallucination_prompt, caption_rewrite_prompt, video_emotion_driven_visual_hallucination_emotion_relevant_prompt, \
                            audio_emotion_driven_audio_hallucination_emotion_relevant_prompt

from abc import ABC, abstractmethod
import logging, os, json
from tqdm import tqdm
import subprocess
import multiprocessing
import base64
import requests

from prompts_eqa import overall_captioning_prompt_video_audio, video_modality_caption_prompt, audio_modality_caption_prompt

class RawVideoDataset(ABC):

    required = ("dataset_path", "dataset_name", "task_type", "dataset_24fps_path", "has_audio")

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        orig_init = cls.__init__

        def wrapped_init(self, *a, **kw):
            orig_init(self, *a, **kw)                      # run child's init
            missing = [attr for attr in cls.required if not hasattr(self, attr)]
            if missing:
                raise AttributeError(
                    f"{cls.__name__}.__init__ didn't set attrs: {', '.join(missing)}"
                )
        
        cls.__init__ = wrapped_init                       # monkey-patch once
    
    def convert_to_24fps(self, dataset_path, with_audio=True):
        if with_audio:
            dataset_24fps_path = dataset_path + "_24fps"
        else:
            dataset_24fps_path = dataset_path + "_24fps_no_audio"
        # Check if the dataset_24fps_path already exists
        # If it exists, we will skip the conversion and return the path
        if os.path.exists(dataset_24fps_path):
            logging.warning(f"Dataset conversion path to 24fps exists at {dataset_24fps_path}. Already converted files will be skipped.")
        else:
            os.makedirs(dataset_24fps_path, exist_ok=True)

        logging.info(f"Converting all the video files inside the dataset directory - {dataset_path} to 24fps. Conversion will happen with audio?: {with_audio}")
        convert_to_24fps_parallel(dataset_path, dataset_24fps_path, with_audio=with_audio)
        logging.info(f"Conversion complete. Saving files inside the directory - {dataset_24fps_path}")
        return dataset_24fps_path

    def convert_to_8_frames(self, dataset_path):
        dataset_8_frame_path = dataset_path+"_8_frames"
        if os.path.exists(dataset_8_frame_path):
            logging.warning(f"Dataset conversion path to 8 frames exists at {dataset_8_frame_path}. Already converted files will be skipped.")
            # return dataset_8_frame_path   
        else:
            os.makedirs(dataset_8_frame_path, exist_ok=True)
        logging.info(f"Reading all the video files inside the dataset directory - {dataset_path} and extracting 8 frames per video.")
        extract_frames_parallel(dataset_path, dataset_8_frame_path, n_frames=8)
        logging.info(f"Conversion complete. Saving files inside the directory - {dataset_8_frame_path}")
        return dataset_8_frame_path
    
    def convert_to_16khz(self, dataset_path):
        dataset_16khz_path = dataset_path+"_16khz"
        if os.path.exists(dataset_16khz_path):
            logging.warning(f"Dataset conversion path to 16khz exists at {dataset_16khz_path}. Already converted files will be skipped.")
            # return dataset_16khz_path
        logging.info(f"Reading all the wav files inside the dataset directory - {dataset_path}")

        # read all the files first so that we can use a progress bar
        all_files = []
        for r, ds, fs in os.walk(dataset_path):
            for f in fs:
                if (f.split(".")[-1] not in VIDEO_EXTENSIONS) or f[0] == ".":
                    continue
                all_files.append((r, f))

        # converting the files to 16khz
        logging.info(f"Converting all the video files inside the dataset directory to audio files - {dataset_path}")
        convert_to_wav_if_audio_stream(all_files, dataset_path, dataset_16khz_path)
        logging.info(f"Conversion complete. Saving files inside the directory - {dataset_16khz_path}")
        return dataset_16khz_path

    def upload_data_to_gcloud(self, audio_video_separate = False):
        if not audio_video_separate:
            already_uploaded = google_cloud_directory_exists(BUCKET_NAME, f"{BUCKET_AUDIO_FOLDER}/{self.dataset_24fps_path.split('/')[-1]}")
            if not already_uploaded:
                logging.info(f"Starting upload of dataset - {self.dataset_24fps_path} to google cloud...")
                google_cloud_upload_directory(self.dataset_24fps_path, BUCKET_NAME, f"{BUCKET_AUDIO_FOLDER}/{self.dataset_24fps_path.split('/')[-1]}")
                logging.info(f"Upload of dataset - {self.dataset_24fps_path} to google cloud complete...")
            else:
                logging.info(f"Dataset already exists - {self.dataset_24fps_path} in google cloud...")
        else:
            audio_already_uploaded = google_cloud_directory_exists(BUCKET_NAME, f"{BUCKET_AUDIO_FOLDER}/{self.dataset_16khz_path.split('/')[-1]}")
            video_already_uploaded = google_cloud_directory_exists(BUCKET_NAME, f"{BUCKET_AUDIO_FOLDER}/{self.dataset_24fps_path_no_audio.split('/')[-1]}")
            if not audio_already_uploaded:
                logging.info(f"Starting upload of audio files - {self.dataset_16khz_path} to google cloud...")
                google_cloud_upload_directory(self.dataset_16khz_path, BUCKET_NAME, f"{BUCKET_AUDIO_FOLDER}/{self.dataset_16khz_path.split('/')[-1]}")
                logging.info(f"Upload of audio files - {self.dataset_16khz_path} to google cloud complete...")
            else:
                logging.info(f"Audio files already exist - {self.dataset_16khz_path} in google cloud...")
            if not video_already_uploaded:
                logging.info(f"Starting upload of video files - {self.dataset_24fps_path_no_audio} to google cloud...")
                google_cloud_upload_directory(self.dataset_24fps_path_no_audio, BUCKET_NAME, f"{BUCKET_AUDIO_FOLDER}/{self.dataset_24fps_path_no_audio.split('/')[-1]}")
                logging.info(f"Upload of video files - {self.dataset_24fps_path_no_audio} to google cloud complete...")
            else:
                logging.info(f"Video files already exist - {self.dataset_24fps_path_no_audio} in google cloud...")

    def upload_frames_to_gcloud(self):
        already_uploaded = google_cloud_directory_exists(BUCKET_NAME, f"{BUCKET_AUDIO_FOLDER}/{self.dataset_8_frame_path.split('/')[-1]}")
        if not already_uploaded:
            logging.info(f"Starting upload of dataset - {self.dataset_8_frame_path} to google cloud...")
            google_cloud_upload_directory(self.dataset_8_frame_path, BUCKET_NAME, f"{BUCKET_AUDIO_FOLDER}/{self.dataset_8_frame_path.split('/')[-1]}")
            logging.info(f"Upload of dataset - {self.dataset_8_frame_path} to google cloud complete...")
        else:
            logging.info(f"Dataset already exists - {self.dataset_8_frame_path} in google cloud...")

        
    def get_naive_instruction_format_data(self):
        
        if self.task_type == "emotion":
            perception_instructions = VIDEO_EMOTION_PERCEPTION_INSTRUCTIONS
        else:
            raise NotImplementedError

        all_files = []
        logging.info(f"Creating naive instruction format data for {self.dataset_name}...")
        for r, ds, fs in os.walk(self.dataset_24fps_path):
            for f in fs:
                all_files.append((r, f))
        
        instr_options = len(perception_instructions)
        instr_data = []
        num_missing_labels = 0
        for idx, (r, f) in tqdm(enumerate(all_files), total = len(all_files)):
            try:
                cur_class = self.get_label_from_fname(os.path.join(r,f))
            except:
                num_missing_labels+=1
                continue
            cur_instr = perception_instructions[idx%instr_options]
            instance = {
                "video_path": os.path.join(r, f),
                "instruction": cur_instr,
                "response": cur_class
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

    def create_jsonl_for_batch_inference(self, jsonl_par, jsonl_path = None, annotation_mode = "description", api_mode = "gemini"):
        os.makedirs(jsonl_par, exist_ok=True)
        if api_mode not in ["gemini", "gpt"]:
            raise ValueError(f"Unknown api_mode - {api_mode}. Supported modes are 'gemini' and 'gpt'.")
        logging.info(f"Creating JSONL file for annotation through {api_mode}.")
        if jsonl_path is None:
            if annotation_mode == "description":
                jsonl_path = os.path.join(jsonl_par, f"{self.dataset_name}.jsonl")
            elif annotation_mode == "caption":
                jsonl_path = os.path.join(jsonl_par, f"{self.dataset_name}_caption.jsonl")
            elif annotation_mode == "qa":
                jsonl_path = os.path.join(jsonl_par, f"{self.dataset_name}_qa.jsonl")
            elif annotation_mode == "modality_separate_caption":
                jsonl_path = os.path.join(jsonl_par, f"{self.dataset_name}_modality_separate_caption.jsonl")
            elif annotation_mode == "modality_qa":
                jsonl_path = os.path.join(jsonl_par, f"{self.dataset_name}_modality_qa.jsonl")
            elif annotation_mode == "modality_qa_all":
                jsonl_path = os.path.join(jsonl_par, f"{self.dataset_name}_modality_qa_all.jsonl")
            elif annotation_mode == "predict_emotion_from_modality_captions":
                jsonl_path = os.path.join(jsonl_par, f"{self.dataset_name}_predict_emotion_from_modality_captions.jsonl")
            elif annotation_mode == "hallucination_qa":
                jsonl_path = os.path.join(jsonl_par, f"{self.dataset_name}_hallucination_qa.jsonl")
            elif annotation_mode == "hallucination_qa_extras":
                jsonl_path = os.path.join(jsonl_par, f"{self.dataset_name}_hallucination_qa_extras.jsonl")
            elif annotation_mode == "av_long_caption_rewrite":
                jsonl_path = os.path.join(jsonl_par, f"{self.dataset_name}_av_long_caption_rewrite.jsonl")
            else:
                raise NotImplementedError(f"Unknown annotation mode - {annotation_mode}")
        if os.path.exists(jsonl_path):
            logging.warning(f"JSONL file already exists at {jsonl_path} for {self.dataset_name}")
            return jsonl_path

        if not self.has_audio and annotation_mode in ["modality_separate_caption", "modality_qa", "modality_qa_all", "predict_emotion_from_modality_captions", "hallucination_qa", "av_long_caption_rewrite", "hallucination_qa_extras"]:
            raise ValueError(f"Annotation mode {annotation_mode} requires audio files, but has_audio is set to False for {self.dataset_name}. Please set has_audio to True or choose a different annotation mode.")

        
        all_files = []
        for r, ds, fs in os.walk(self.dataset_24fps_path):
            for f in fs:
                if (f.split(".")[-1] not in VIDEO_EXTENSIONS) or f[0] == ".":
                    continue
                all_files.append((r, f))
        if annotation_mode == "qa":
            logging.info("Since annotation_mode is qa, we need to find captions for all the videos available and read the captions")
            caption_prediction_jsonl_path = os.path.join(jsonl_par, f"{self.dataset_name}_caption_predictions.jsonl")
            if not os.path.exists(caption_prediction_jsonl_path):
                raise FileNotFoundError(f"Caption prediction jsonl file not found at {caption_prediction_jsonl_path}. Please run the caption prediction script first, or move the file to the correct location.")
            all_captions = get_res_from_jsonl(caption_prediction_jsonl_path, gemini_mode="caption")
            print(list(all_captions.keys())[:5])  # Print first 5 captions for debugging
        elif annotation_mode in ["modality_qa", "modality_qa_all", "predict_emotion_from_modality_captions", "hallucination_qa", "av_long_caption_rewrite", "hallucination_qa_extras"]:
            logging.info(f"Since annotation_mode is {annotation_mode}, we need to find captions for all the videos available and read the captions")
            audio_video_modality_caption_prediction_jsonl_path = os.path.join(jsonl_par.replace("gpt_annotations/jsonl_files", "gemini_annotations"), f"{self.dataset_name}_modality_separate_caption_predictions.jsonl")
            if not os.path.exists(audio_video_modality_caption_prediction_jsonl_path):
                raise FileNotFoundError(f"Audio-Video modality caption prediction jsonl file not found at {audio_video_modality_caption_prediction_jsonl_path}. Please run the caption prediction script first, or move the file to the correct location.")
            _, caption_audio = get_res_from_jsonl(audio_video_modality_caption_prediction_jsonl_path, gemini_mode="modality_separate_caption")

            ## read video captions from GPT
            audio_video_modality_caption_prediction_jsonl_path = os.path.join(jsonl_par.replace("gemini_annotations", "gpt_annotations/jsonl_files"), f"{self.dataset_name}_modality_separate_caption_predictions.jsonl")
            if not os.path.exists(audio_video_modality_caption_prediction_jsonl_path):
                raise FileNotFoundError(f"Audio-Video modality caption prediction jsonl file not found at {audio_video_modality_caption_prediction_jsonl_path}. Please run the caption prediction script first, or move the file to the correct location.")
            caption_video = get_res_from_jsonl_gpt(audio_video_modality_caption_prediction_jsonl_path, annotation_mode="modality_separate_caption")

            print("Video captions example --", list(caption_video.keys())[:5])  # Print first 5 video captions
            print("Audio captions example --", list(caption_audio.keys())[:5])  # Print first 5 audio captions
            
            
        if annotation_mode in ["hallucination_qa", "hallucination_qa_extras"]:
            ## read the audio video correct incorrect samples for the current dataset
            correct_sample_par = jsonl_par.replace("gpt_annotations/jsonl_files", "emotion_prediction_results").replace("gemini_annotations", "emotion_prediction_results")
            audio_correct_samples_path = os.path.join(correct_sample_par, f"{self.dataset_name}_audio.json")
            with open(audio_correct_samples_path, "r") as f:
                audio_correct_samples = json.load(f)
            video_correct_samples_path = os.path.join(correct_sample_par, f"{self.dataset_name}_video.json")
            with open(video_correct_samples_path, "r") as f:
                video_correct_samples = json.load(f)

        dataset_gcloud_dir = self.dataset_24fps_path.split('/')[-1]
        len_local_data_dir_subpath = len(self.dataset_24fps_path)

        all_requests = []
        num_missing_labels, num_missing_captions = 0, 0
        num_error_captions = 0
        num_video, num_audio, num_video_only, num_audio_only = 0, 0, 0, 0
        for idx, (r, f) in tqdm(enumerate(all_files), total = len(all_files)):
            # if idx>=20:
            #     break
            cur_video_subpath = os.path.join(r,f)[len_local_data_dir_subpath+1:]
            try:
                cur_label = self.get_label_from_fname(f"{dataset_gcloud_dir}/{cur_video_subpath}")
            except:
                num_missing_labels+=1
                continue
            if api_mode == "gemini" and annotation_mode in ["description", "caption"]:
                if annotation_mode == "description":
                    cur_prompt = self.gemini_prompt.replace(REPLACE_LABEL_STRING, cur_label)
                elif annotation_mode == "caption":
                    cur_prompt = self.gemini_overall_captioning_prompt.replace(REPLACE_LABEL_STRING, cur_label)
                #### add request to all_requests
                cur_parts = [{"text":cur_prompt}]

                # Construct the Google Cloud URI for the video file
                cur_gcloud_uri = f"gs://{BUCKET_NAME}/{BUCKET_AUDIO_FOLDER}/{dataset_gcloud_dir}/{cur_video_subpath}"
                cur_parts.append({"fileData": {"fileUri": cur_gcloud_uri, "mimeType": "video/mp4"}})
                
                cur_request = {"request":{"contents": [{"role": "user", "parts": cur_parts}]}}
                all_requests.append(cur_request)
            elif api_mode == "gemini" and annotation_mode == "modality_separate_caption":
                ## one request for video modality caption
                cur_video_caption_prompt = video_modality_caption_prompt
                cur_video_parts = [{"text": cur_video_caption_prompt}]
                cur_video_gcloud_uri = f"gs://{BUCKET_NAME}/{BUCKET_AUDIO_FOLDER}/{dataset_gcloud_dir.replace('_24fps', '_24fps_no_audio')}/{cur_video_subpath}"
                cur_video_parts.append({"fileData": {"fileUri": cur_video_gcloud_uri, "mimeType": "video/mp4"}})
                cur_video_request = {"request":{"contents": [{"role": "user", "parts": cur_video_parts}]}}
                all_requests.append(cur_video_request)

                ## one request for audio modality caption
                cur_audio_caption_prompt = audio_modality_caption_prompt
                cur_audio_parts = [{"text": cur_audio_caption_prompt}]
                cur_audio_gcloud_uri = f"gs://{BUCKET_NAME}/{BUCKET_AUDIO_FOLDER}/{dataset_gcloud_dir.replace('_24fps', '_16khz')}/{cur_video_subpath.replace('.mp4', '.wav')}"
                cur_audio_parts.append({"fileData": {"fileUri": cur_audio_gcloud_uri, "mimeType": "audio/wav"}})
                cur_audio_request = {"request":{"contents": [{"role": "user", "parts": cur_audio_parts}]}}
                all_requests.append(cur_audio_request)

            elif api_mode == "gpt" and annotation_mode == "modality_separate_caption":
                ## one request for video modality caption
                cur_video_caption_prompt = video_modality_caption_prompt
                cur_message_contents = [{
                    "type": "text",
                    "text": cur_video_caption_prompt
                }]
                for frame_idx in range(8):
                    cur_message_contents.append({
                        "type": "image_url",
                        "image_url":{
                            "url": f"https://storage.googleapis.com/{BUCKET_NAME}/{BUCKET_AUDIO_FOLDER}/{dataset_gcloud_dir.replace('_24fps', '_8_frames')}/{cur_video_subpath.replace('.mp4', '')}--{frame_idx}.png", 
                            "detail":"low"
                        }
                    })
                cur_parts = {"role": "user", "content": cur_message_contents}
                cur_request = {"custom_id": cur_video_subpath+"|video", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o", "messages": [cur_parts],"max_completion_tokens": 400}}
                all_requests.append(cur_request)

                ## audio not supported for batch inference GPT

            elif api_mode == "gemini" and annotation_mode == "qa":

                cur_gcloud_uri = f"gs://{BUCKET_NAME}/{BUCKET_AUDIO_FOLDER}/{dataset_gcloud_dir}/{cur_video_subpath}"
                if cur_gcloud_uri not in all_captions:
                    num_missing_captions += 1
                    if num_missing_captions <10:
                        logging.warning(f"Caption not found for {cur_gcloud_uri}. Skipping this video.")
                    continue
                video_caption = all_captions[cur_gcloud_uri]
                if "ERROR" in video_caption:
                    num_error_captions += 1
                    continue  # Skip if the caption is an error message
                cur_video_id = cur_video_subpath
                all_prompts = self.gemini_qa_prompts
                for prompt in all_prompts:
                    cur_prompt = prompt.replace(REPLACE_LABEL_STRING, cur_label).replace(REPLACE_CAPTION_STRING, video_caption).replace(REPLACE_VIDEO_ID_STRING, cur_video_id)
                    # Construct the Google Cloud URI for the video file
                    cur_parts = [{"text":cur_prompt}]
                    
                    cur_request = {"request":{"contents": [{"role": "user", "parts": cur_parts}]}}
                    all_requests.append(cur_request)

            elif api_mode == "gemini" and annotation_mode in ["modality_qa", "predict_emotion_from_modality_captions", "modality_qa_all"]:
                cur_gcloud_uri = f"gs://{BUCKET_NAME}/{BUCKET_AUDIO_FOLDER}/{dataset_gcloud_dir}/{cur_video_subpath}"
                if ((cur_gcloud_uri not in caption_video) and (cur_video_subpath not in caption_video)) or cur_gcloud_uri not in caption_audio:
                    num_missing_captions += 1
                    if num_missing_captions <10:
                        logging.warning(f"Caption not found for {cur_gcloud_uri}. Skipping this video.")
                    continue
                video_caption = caption_video[cur_gcloud_uri] if cur_gcloud_uri in caption_video else caption_video[cur_video_subpath]
                audio_caption = caption_audio[cur_gcloud_uri]
                if "ERROR" in video_caption or "ERROR" in audio_caption:
                    num_error_captions += 1
                    continue
                cur_video_id = cur_video_subpath
                if annotation_mode == "modality_qa":
                    all_prompts = [audio_video_modality_agreement_prompt, audio_video_modality_hallucination_prompt]
                elif annotation_mode == "predict_emotion_from_modality_captions":
                    all_prompts = [audio_modality_emotion_prediction_prompt, video_modality_emotion_prediction_prompt]
                elif annotation_mode == "modality_qa_all":
                    # all_prompts = [audio_video_modality_agreement_prompt, audio_video_modality_hallucination_prompt, audio_video_modality_implicit_cause_reasoning_prompt, 
                    #     audio_video_modality_identification_prompt, audio_video_modality_visual_reasoning_prompt, audio_video_modality_audio_reasoning_prompt]
                    all_prompts = [audio_video_modality_agreement_prompt, audio_video_modality_visual_reasoning_prompt, audio_video_modality_audio_reasoning_prompt]
                
                
                for prompt in all_prompts:
                    cur_prompt = prompt.replace(REPLACE_LABEL_STRING, cur_label).replace(REPLACE_VIDEO_CAPTION_STRING, str(video_caption)).replace(REPLACE_AUDIO_CAPTION_STRING, str(audio_caption)).replace(REPLACE_VIDEO_ID_STRING, cur_video_id)
                    # Construct the Google Cloud URI for the video file
                    cur_parts = [{"text":cur_prompt}]
                    
                    cur_request = {"request":{"contents": [{"role": "user", "parts": cur_parts}]}}
                    all_requests.append(cur_request)
            elif api_mode == "gpt" and annotation_mode in ["modality_qa_all", "hallucination_qa", "av_long_caption_rewrite", "hallucination_qa_extras"]:
                cur_gcloud_uri = f"gs://{BUCKET_NAME}/{BUCKET_AUDIO_FOLDER}/{dataset_gcloud_dir}/{cur_video_subpath}"
                if ((cur_gcloud_uri not in caption_video) and (cur_video_subpath not in caption_video)) or cur_gcloud_uri not in caption_audio:
                    num_missing_captions += 1
                    if num_missing_captions <10:
                        logging.warning(f"Caption not found for {cur_gcloud_uri}. Skipping this video.")
                    continue
                video_caption = caption_video[cur_gcloud_uri] if cur_gcloud_uri in caption_video else caption_video[cur_video_subpath]
                audio_caption = caption_audio[cur_gcloud_uri]
                if "ERROR" in video_caption or "ERROR" in audio_caption:
                    num_error_captions += 1
                    continue
                cur_video_id = cur_video_subpath
                # all_prompts = [audio_video_modality_agreement_prompt, audio_video_modality_hallucination_prompt, audio_video_modality_implicit_cause_reasoning_prompt, 
                #     audio_video_modality_identification_prompt, audio_video_modality_visual_reasoning_prompt, audio_video_modality_audio_reasoning_prompt]
                max_completion_tokens = 200
                model = "gpt-4o-mini"
                if annotation_mode == "modality_qa_all":
                    max_completion_tokens = 600
                    all_prompts = [audio_video_modality_agreement_prompt, audio_video_modality_visual_reasoning_prompt, audio_video_modality_audio_reasoning_prompt]
                elif annotation_mode == "av_long_caption_rewrite":
                    max_completion_tokens = 600
                    all_prompts = [caption_rewrite_prompt]
                elif annotation_mode == "hallucination_qa":
                    model = "gpt-4o"
                    all_prompts = []
                    video_correct = cur_video_subpath in video_correct_samples
                    audio_correct = cur_video_subpath in audio_correct_samples
                    
                    if video_correct:
                        all_prompts.append(video_emotion_driven_visual_hallucination_video_relevant_prompt)
                        all_prompts.append(video_emotion_driven_visual_no_hallucination_prompt)
                        num_video += 1
                        if not audio_correct:
                            all_prompts.append(video_emotion_driven_audio_hallucination_emotion_relevant_prompt)
                            all_prompts.append(video_emotion_driven_audio_hallucination_audio_relevant_prompt)
                            num_video_only += 1
                    if audio_correct:
                        all_prompts.append(audio_emotion_driven_audio_hallucination_audio_relevant_prompt)
                        all_prompts.append(audio_emotion_driven_audio_no_hallucination_prompt)
                        num_audio += 1
                        if not video_correct:
                            all_prompts.append(audio_emotion_driven_visual_hallucination_emotion_relevant_prompt)
                            all_prompts.append(audio_emotion_driven_visual_hallucination_video_relevant_prompt)
                            num_audio_only += 1
                    if len(all_prompts) == 0:
                        continue
                elif annotation_mode == "hallucination_qa_extras":
                    model = "gpt-4o"
                    all_prompts = []
                    video_correct = cur_video_subpath in video_correct_samples
                    audio_correct = cur_video_subpath in audio_correct_samples
                    
                    if video_correct:
                        all_prompts.append(video_emotion_driven_visual_hallucination_emotion_relevant_prompt)
                        num_video += 1
                    if audio_correct:
                        all_prompts.append(audio_emotion_driven_audio_hallucination_emotion_relevant_prompt)
                        num_audio += 1
                    if len(all_prompts) == 0:
                        continue

                for prompt_idx, prompt in enumerate(all_prompts):
                    cur_prompt = prompt.replace(REPLACE_LABEL_STRING, cur_label).replace(REPLACE_VIDEO_CAPTION_STRING, str(video_caption)).replace(REPLACE_AUDIO_CAPTION_STRING, str(audio_caption)).replace(REPLACE_VIDEO_ID_STRING, cur_video_id)
                    # Construct the Google Cloud URI for the video file
                    cur_parts = {"role": "user", "content": cur_prompt}

                    cur_request = {"custom_id": cur_video_id+"|"+str(prompt_idx), "method": "POST", "url": "/v1/chat/completions", "body": {"model": model, "messages": [cur_parts],"max_completion_tokens": max_completion_tokens}}
                    all_requests.append(cur_request)

            else:
                raise NotImplementedError(f"Unknown annotation mode - {annotation_mode} and api_mode - {api_mode} combination. Please check. Supported modes are 'description', 'caption', 'qa', 'modality_separate_caption', 'modality_qa', and 'modality_qa_all' for api_mode 'gemini'.")

        logging.info(f"{num_missing_labels}/{len(all_files)} labels were missing while creating the jsonl files for the dataset...")
        logging.info(f"Video support emotion: {num_video}/{len(all_files)}, Only video supports emotion: {num_video_only}/{len(all_files)}")
        logging.info(f"Audio support emotion: {num_audio}/{len(all_files)}, Only audio supports emotion: {num_audio_only}/{len(all_files)}")
        if annotation_mode in ["qa", "modality_qa", "modality_qa_all", "hallucination_qa", "av_long_caption_rewrite", "hallucination_qa_extras"]:
            logging.info(f"{num_missing_captions}/{len(all_files)} captions were missing while creating the jsonl files for the dataset...")
            logging.info(f"{num_error_captions}/{len(all_files)} captions had errors in captions while creating the jsonl files for the dataset...")
        logging.info(f"Total requests created: {len(all_requests)} for {self.dataset_name} dataset in {annotation_mode} mode.")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for req in all_requests:
                # Convert the dictionary to a JSON string and write it as a single line
                json_line = json.dumps(req)
                f.write(json_line + "\n")
        logging.info(f"Wrote JSONL file to {jsonl_path}")
        return jsonl_path

    def get_dataset_gemini_annotation_budget(self, gemini_mode = "description", gemini_version = "gemini-2.0-flash-001"):
        if gemini_mode == "description":
            input_chars_per_request = len(self.gemini_prompt)
            output_chars_per_response = 300
        elif gemini_mode == "caption":
            input_chars_per_request = len(self.gemini_overall_captioning_prompt)
            output_chars_per_response = 2500
        else:
            input_chars_per_request = sum(len(prompt) for prompt in self.gemini_qa_prompts)
            output_chars_per_response = 2500 * len(self.gemini_qa_prompts)  # Assuming each prompt has a similar response length
        logging.info(f"Calculating budget for {self.dataset_name} dataset in {gemini_mode} mode...")
        if gemini_mode in ["description", "caption"]:
            logging.info(f"Checking total duration of all video files in the dataset...")
            total_files, total_video_secs = total_video_duration(self.dataset_24fps_path)
        else:
            video_exts = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.mpeg', '.mpg')
            video_files = [
                os.path.join(root, file)
                for root, _, files in os.walk(self.dataset_24fps_path)
                for file in files if file.lower().endswith(video_exts)
            ]
            total_files = len(video_files)
        total_cost = 0
        if gemini_mode in ["description", "caption"]:
            video_secs_cost = total_video_secs * GEMINI_PRICE[gemini_version]["input_video_per_sec"]
            print(f"Total video duration in seconds: {total_video_secs} x ${GEMINI_PRICE[gemini_version]['input_video_per_sec']}, cost: ${video_secs_cost:.2f}")
            total_cost += video_secs_cost
            if self.has_audio:
                audio_secs_cost = total_video_secs * GEMINI_PRICE[gemini_version]["input_audio_per_sec"]
                print(f"Total audio duration in seconds: {total_video_secs} x ${GEMINI_PRICE[gemini_version]['input_audio_per_sec']}, cost: ${audio_secs_cost:.2f}")
                total_cost += audio_secs_cost
        total_input_prompt_cost = total_files * input_chars_per_request * GEMINI_PRICE[gemini_version]["input_text_per_million_char"] / 1.e6
        print(f"Total input prompt characters: {total_files * input_chars_per_request} x ${GEMINI_PRICE[gemini_version]['input_text_per_million_char']}/1e-6, cost: ${total_input_prompt_cost:.2f}")
        total_cost += total_input_prompt_cost
        if gemini_mode == "qa":
            total_input_caption_cost = total_files * 2500 * GEMINI_PRICE[gemini_version]["input_text_per_million_char"] / 1.e6
            print(f"Total input caption characters: {total_files * 2500} x ${GEMINI_PRICE[gemini_version]['input_text_per_million_char']}/1e-6, cost: ${total_input_caption_cost:.2f}")
            total_cost += total_input_caption_cost
        total_output_cost = total_files * output_chars_per_response * GEMINI_PRICE[gemini_version]["output_text_per_million_char"] / 1.e6
        print(f"Total output response characters: {total_files * output_chars_per_response} x ${GEMINI_PRICE[gemini_version]['output_text_per_million_char']}/1e-6, cost: ${total_output_cost:.2f}")
        total_cost += total_output_cost
        print(f"Total cost for running gemini inference is estimated as ${total_cost:.2f}")
        return total_cost

        

