from data_constants import VIDEO_EMOTION_INSTRUCTIONS, REPLACE_LABEL_STRING, VIDEO_EXTENSIONS, BUCKET_NAME, REPLACE_LABEL_STRING, BUCKET_AUDIO_FOLDER
from prompts_eqa import overall_captioning_prompt_video_audio, qa_identification_prompt_video_audio, \
    qa_visual_reasoning_prompt_video_audio, qa_audio_reasoning_prompt_video_audio, \
    qa_temporal_variation_prompt_video_audio, qa_modality_agreement_prompt_video_audio, \
    qa_implicit_cause_reasoning_prompt_video_audio
from .raw_video_dataset import RawVideoDataset
from utils import google_cloud_directory_exists, get_res_from_jsonl_gpt, shuffle_choices
import random


import logging, os, json
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

import logging
from pathlib import Path
from tqdm import tqdm
from google.cloud import storage
import multiprocessing as mp

from pathlib import Path
try:
    from google.cloud import storage        # ⬅ official GCS client
    import vertexai
    from vertexai.batch_prediction import BatchPredictionJob
except ImportError:
    print("Google Cloud libraries not found...")
    print("You must switch environment to the correct one before using google cloud libraries.")
    
import fsspec

### GCLOUD ESSENTIAALS
# --- at module top (not inside a class) ---
from pathlib import Path
from google.cloud import storage

# Globals set per process via the initializer
_GCS_BUCKET = None
_LOCAL_DIR = None
_GCS_PREFIX = None

def _init_gcs(bucket_name: str, local_dir_str: str, gcs_prefix: str):
    """Per-process initializer: create a client/bucket once per process."""
    global _GCS_BUCKET, _LOCAL_DIR, _GCS_PREFIX
    _GCS_BUCKET = storage.Client().bucket(bucket_name)
    _LOCAL_DIR = Path(local_dir_str)
    _GCS_PREFIX = gcs_prefix

def _gcs_upload_worker(path_str: str) -> str:
    """Module-level worker (picklable)."""
    p = Path(path_str)
    rel_path = p.relative_to(_LOCAL_DIR).as_posix()
    blob_name = f"{_GCS_PREFIX}/{rel_path}".lstrip("/")
    blob = _GCS_BUCKET.blob(blob_name)
    blob.upload_from_filename(p)
    return blob_name


class MER2023Test1(RawVideoDataset):
    def __init__(self, dataset_name, dataset_path):
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
        
        self.video_depth = 1

        mer_single_label_df = pd.read_csv(f"{self.dataset_path}/../test_labels/test1-label.csv")
        self.classes_map = {"happy":"happiness", "sad":"sadness", "neutral": "neutral", "angry": "anger", "surprise":"surprise"}

        self.label_dict = {}
        labels_skipped = 0
        for idx, row in mer_single_label_df.iterrows():
            video_id = str(row["name"])
            cur_label = row["discrete"]
            try:
                self.label_dict[video_id] = self.classes_map[cur_label]
            except KeyError:
                labels_skipped += 1
                continue
        logging.info(f"Skipped {labels_skipped}/{len(mer_single_label_df.index)} labels in MER2023 Test 1 dataset due to missing or incorrect labels.")


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
                            "video_driven_visual_hallucination_video_relevant", "video_driven_visual_no_hallucination", 
                            "video_driven_audio_hallucination_emotion_relevant", "video_driven_audio_hallucination_audio_relevant",
                            "audio_driven_audio_hallucination_audio_relevant", "audio_driven_audio_no_hallucination"]
        self.valid_qa_categories.extend(hallucination_categories)
    
    def get_label_from_fname(self, fpath):
        video_id = fpath.split("/")[-1].split(".")[0]
        if video_id not in self.label_dict:
            raise Exception(f"Label not found for {video_id}")
        return self.label_dict[video_id]

    def upload_data_to_gcloud(self, audio_video_separate=False):
        if audio_video_separate == False:
            ## overriding the upload function becuase we only want to upload the files which have single labels.
            already_uploaded = google_cloud_directory_exists(BUCKET_NAME, f"{BUCKET_AUDIO_FOLDER}/{self.dataset_24fps_path.split('/')[-1]}")
            if not already_uploaded:
                logging.info(f"Starting upload of dataset - {self.dataset_24fps_path} to google cloud...")
                # google_cloud_upload_directory(self.dataset_24fps_path, BUCKET_NAME, f"{BUCKET_AUDIO_FOLDER}/{self.dataset_24fps_path.split('/')[-1]}")
                local_dir = Path(self.dataset_24fps_path).resolve()
                bucket_name = BUCKET_NAME
                gcs_prefix = f"{BUCKET_AUDIO_FOLDER}/{self.dataset_24fps_path.split('/')[-1]}"
                threads = 8  # Number of parallel uploads
                if not local_dir.is_dir():
                    raise ValueError(f"{local_dir} is not a directory")

                client = storage.Client()            # uses ADC or service-account JSON
                bucket = client.bucket(bucket_name)

                # Collect every *file* under the directory
                files = [p for p in local_dir.rglob("*") 
                        if p.is_file() and p.stem in self.label_dict]

                def _upload(path: Path):
                    rel_path = path.relative_to(local_dir).as_posix()   # keep slashes
                    blob_name = f"{gcs_prefix}/{rel_path}".lstrip("/")  # final GCS key
                    blob = bucket.blob(blob_name)
                    blob.upload_from_filename(path)
                    return blob_name

                # Fan-out parallel uploads
                with ThreadPoolExecutor(max_workers=threads) as pool:
                    futures = [pool.submit(_upload, f) for f in files]
                    for _ in tqdm(as_completed(futures), total=len(futures), desc="Uploading"):
                        pass                                             # raises on failure

                print(f"✅ Uploaded {len(files)} objects to gs://{bucket_name}/{gcs_prefix}")
                logging.info(f"Upload of dataset - {self.dataset_24fps_path} to google cloud complete...")
            else:
                logging.info(f"Dataset already exists - {self.dataset_24fps_path} in google cloud...")
        else:
            audio_already_uploaded = google_cloud_directory_exists(BUCKET_NAME, f"{BUCKET_AUDIO_FOLDER}/{self.dataset_16khz_path.split('/')[-1]}")
            video_already_uploaded = google_cloud_directory_exists(BUCKET_NAME, f"{BUCKET_AUDIO_FOLDER}/{self.dataset_24fps_path_no_audio.split('/')[-1]}")
            if not audio_already_uploaded:
                logging.info(f"Starting upload of audio in the dataset - {self.dataset_16khz_path} to google cloud...")
                local_dir = Path(self.dataset_16khz_path).resolve()
                bucket_name = BUCKET_NAME
                gcs_prefix = f"{BUCKET_AUDIO_FOLDER}/{self.dataset_16khz_path.split('/')[-1]}"
                threads = 8  # Number of parallel uploads
                if not local_dir.is_dir():
                    raise ValueError(f"{local_dir} is not a directory")

                client = storage.Client()            # uses ADC or service-account JSON
                bucket = client.bucket(bucket_name)

                # Collect every *file* under the directory
                files = [p for p in local_dir.rglob("*") 
                        if p.is_file() and p.stem in self.label_dict]

                def _upload(path: Path):
                    rel_path = path.relative_to(local_dir).as_posix()   # keep slashes
                    blob_name = f"{gcs_prefix}/{rel_path}".lstrip("/")  # final GCS key
                    blob = bucket.blob(blob_name)
                    blob.upload_from_filename(path)
                    return blob_name

                # Fan-out parallel uploads
                with ThreadPoolExecutor(max_workers=threads) as pool:
                    futures = [pool.submit(_upload, f) for f in files]
                    for _ in tqdm(as_completed(futures), total=len(futures), desc="Uploading"):
                        pass                                             # raises on failure

                print(f"✅ Uploaded {len(files)} objects to gs://{bucket_name}/{gcs_prefix}")
                logging.info(f"Upload of dataset - {self.dataset_16khz_path} to google cloud complete...")
            else:
                logging.info(f"Audio dataset already exists - {self.dataset_16khz_path} in google cloud...")
            if not video_already_uploaded:
                logging.info(f"Starting upload of dataset - {self.dataset_24fps_path_no_audio} to google cloud...")
                local_dir = Path(self.dataset_24fps_path_no_audio).resolve()
                bucket_name = BUCKET_NAME
                gcs_prefix = f"{BUCKET_AUDIO_FOLDER}/{self.dataset_24fps_path_no_audio.split('/')[-1]}"
                threads = 8  # Number of parallel uploads
                if not local_dir.is_dir():
                    raise ValueError(f"{local_dir} is not a directory")

                client = storage.Client()            # uses ADC or service-account JSON
                bucket = client.bucket(bucket_name)

                # Collect every *file* under the directory
                files = [p for p in local_dir.rglob("*") 
                        if p.is_file() and p.stem in self.label_dict]

                def _upload(path: Path):
                    rel_path = path.relative_to(local_dir).as_posix()   # keep slashes
                    blob_name = f"{gcs_prefix}/{rel_path}".lstrip("/")  # final GCS key
                    blob = bucket.blob(blob_name)
                    blob.upload_from_filename(path)
                    return blob_name

                # Fan-out parallel uploads
                with ThreadPoolExecutor(max_workers=threads) as pool:
                    futures = [pool.submit(_upload, f) for f in files]
                    for _ in tqdm(as_completed(futures), total=len(futures), desc="Uploading"):
                        pass                                             # raises on failure

                print(f"✅ Uploaded {len(files)} objects to gs://{bucket_name}/{gcs_prefix}")
                logging.info(f"Upload of dataset - {self.dataset_24fps_path_no_audio} to google cloud complete...")
            else:
                logging.info(f"Video dataset already exists - {self.dataset_24fps_path_no_audio} in google cloud...")

    # def upload_frames_to_gcloud(self):
    #     ## overriding the upload function becuase we only want to upload the files which have single labels.
        # already_uploaded = google_cloud_directory_exists(BUCKET_NAME, f"{BUCKET_AUDIO_FOLDER}/{self.dataset_8_frame_path.split('/')[-1]}")
        # if not already_uploaded:
    #     logging.info(f"Starting upload of dataset - {self.dataset_8_frame_path} to google cloud...")
    #     local_dir = Path(self.dataset_8_frame_path).resolve()
    #     bucket_name = BUCKET_NAME
    #     gcs_prefix = f"{BUCKET_AUDIO_FOLDER}/{self.dataset_8_frame_path.split('/')[-1]}"
    #     threads = 8  # Number of parallel uploads
    #     if not local_dir.is_dir():
    #         raise ValueError(f"{local_dir} is not a directory")

    #     client = storage.Client()            # uses ADC or service-account JSON
    #     bucket = client.bucket(bucket_name)

    #     # Collect every *file* under the directory
    #     files = [p for p in local_dir.rglob("*") 
    #             if p.is_file() and str(p.stem).split("--")[0] in self.label_dict]

    #     def _upload(path: Path):
    #         rel_path = path.relative_to(local_dir).as_posix()   # keep slashes
    #         blob_name = f"{gcs_prefix}/{rel_path}".lstrip("/")  # final GCS key
    #         blob = bucket.blob(blob_name)
    #         blob.upload_from_filename(path)
    #         return blob_name

    #     # Fan-out parallel uploads
    #     with ThreadPoolExecutor(max_workers=threads) as pool:
    #         futures = [pool.submit(_upload, f) for f in files]
    #         for _ in tqdm(as_completed(futures), total=len(futures), desc="Uploading"):
    #             pass                                             # raises on failure

    #     print(f"✅ Uploaded {len(files)} objects to gs://{bucket_name}/{gcs_prefix}")
    #     logging.info(f"Upload of dataset - {self.dataset_8_frame_path} to google cloud complete...")
        # else:
        #     logging.info(f"Dataset already exists - {self.dataset_8_frame_path} in google cloud...")

    def upload_frames_to_gcloud(self):
        already_uploaded = google_cloud_directory_exists(BUCKET_NAME, f"{BUCKET_AUDIO_FOLDER}/{self.dataset_8_frame_path.split('/')[-1]}")
        if not already_uploaded:
            logging.info(f"Starting upload of dataset - {self.dataset_8_frame_path} to google cloud...")
            local_dir = Path(self.dataset_8_frame_path).resolve()
            bucket_name = BUCKET_NAME
            gcs_prefix = f"{BUCKET_AUDIO_FOLDER}/{self.dataset_8_frame_path.split('/')[-1]}"
            processes = 8  # parallelism

            if not local_dir.is_dir():
                raise ValueError(f"{local_dir} is not a directory")

            # Pre-filter files in the main process
            files = [
                p for p in local_dir.rglob("*")
                if p.is_file() and str(p.stem).split("--")[0] in self.label_dict
            ]
            file_strs = [str(p) for p in files]  # pass strings to workers (safer to pickle)

            ctx = mp.get_context("spawn")  # safer across platforms and with cloud SDKs
            with ctx.Pool(
                processes=processes,
                initializer=_init_gcs,
                initargs=(bucket_name, str(local_dir), gcs_prefix),
                maxtasksperchild=200,  # optional: avoid long-lived workers
            ) as pool:
                # Consume results to surface exceptions; show progress
                for _ in tqdm(
                    pool.imap_unordered(_gcs_upload_worker, file_strs, chunksize=32),
                    total=len(file_strs),
                    desc="Uploading",
                ):
                    pass

            print(f"✅ Uploaded {len(files)} objects to gs://{bucket_name}/{gcs_prefix}")
            logging.info(f"Upload of dataset - {self.dataset_8_frame_path} to google cloud complete...")
        else:
            logging.info(f"Dataset already exists - {self.dataset_8_frame_path} in google cloud...")

    def get_gemini_instruction_format_data(self, gemini_predictions, gemini_mode = "description"):

        gemini_predictions = {os.path.join(self.dataset_24fps_path, "/".join(k.split("/")[-self.video_depth:])):v for k, v in gemini_predictions.items()}

        all_files = []
        logging.info("Creating gemini instruction data for MER 2025 Track 1 - Single Label...")
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
        elif gemini_mode in ["qa", "modality_qa", "modality_qa_all"]:
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
                    try:
                        new_choices, new_correct_choice = shuffle_choices(question["choices"], question["answer"]["choice"])
                        cur_instr_mcq = question["question"] + " " + " ".join(new_choices)
                    except:
                        logging.warning(f"Something went wrong while constructing the mcq instruction for {os.path.join(r, f)}")
                        missing_cat += 1
                        continue
                    if "category" not in question:
                        # logging.warning(f"Category not found in question for {os.path.join(r, f)}")
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
                    if "hallucination" in category:
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
                            "response": f"({new_correct_choice}). {question['answer']['text']}",
                            "category": category
                        })
                    else:
                        # randomly pick the mcq instruction or the open ended instruction
                        if random.random() <0.5:
                            instr_data.append({
                                "video_path": video_path,
                                "audio_path": audio_path,
                                "instruction": "<video>\n<audio>\n" + cur_instr,
                                "response": question["answer"]["text"],
                                "category": category
                            })
                        else:
                            instr_data.append({
                                "video_path": video_path,
                                "audio_path": audio_path,
                                "instruction": "<video>\n<audio>\n" + cur_instr_mcq,
                                "response": f"({new_correct_choice}). {question['answer']['text']}",
                                "category": category
                            })
            logging.info(f"Missing categories in the questions: {missing_cat}")
        logging.info(f"Missing files in the dataset: {missing_files}/{len(all_files)}")
        return instr_data





    
    


