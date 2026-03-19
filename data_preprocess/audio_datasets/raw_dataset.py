from data_constants import (AUDIO_EXTENSIONS, BUCKET_NAME, BUCKET_AUDIO_FOLDER, REPLACE_LABEL_STRING, 
                            GEMINI_PRICE, EMOTION_PERCEPTION_INSTRUCTIONS, AGE_PERCEPTION_INSTRUCTIONS,
                            GENDER_PERCEPTION_INSTRUCTIONS)
from utils import (extra_path, google_cloud_upload_directory, google_cloud_directory_exists,
                    convert_to_wav_parallel, total_wav_duration)

from abc import ABC, abstractmethod
import logging, os, json
from tqdm import tqdm

class RawAudioDataset(ABC):

    required = ("dataset_path", "dataset_16khz_path", "dataset_name", "task_type")

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

    def __init__(self, *args, **kwargs):
        self.has_audio = True

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
                if (f.split(".")[-1] not in AUDIO_EXTENSIONS) or f[0] == ".":
                    continue
                all_files.append((r, f))

        # converting the files to 16khz
        logging.info(f"Converting all the wav files inside the dataset directory - {dataset_path}")
        convert_to_wav_parallel(all_files, dataset_path, dataset_16khz_path)
        logging.info(f"Conversion complete. Saving files inside the directory - {dataset_16khz_path}")
        return dataset_16khz_path

    def upload_data_to_gcloud(self):
        already_uploaded = google_cloud_directory_exists(BUCKET_NAME, f"{BUCKET_AUDIO_FOLDER}/{self.dataset_16khz_path.split('/')[-1]}")
        if not already_uploaded:
            logging.info(f"Starting upload of dataset - {self.dataset_16khz_path} to google cloud...")
            google_cloud_upload_directory(self.dataset_16khz_path, BUCKET_NAME, f"{BUCKET_AUDIO_FOLDER}/{self.dataset_16khz_path.split('/')[-1]}")
            logging.info(f"Upload of dataset - {self.dataset_16khz_path} to google cloud complete...")
        else:
            logging.info(f"Dataset already exists - {self.dataset_16khz_path} in google cloud...")
        
    def get_naive_instruction_format_data(self):
        
        if self.task_type == "age":
            perception_instructions = AGE_PERCEPTION_INSTRUCTIONS
        elif self.task_type == "emotion":
            perception_instructions = EMOTION_PERCEPTION_INSTRUCTIONS
        elif self.task_type == "gender":
            perception_instructions = GENDER_PERCEPTION_INSTRUCTIONS
        else:
            raise NotImplementedError

        all_files = []
        logging.info(f"Creating naive instruction format data for {self.dataset_name}...")
        for r, ds, fs in os.walk(self.dataset_16khz_path):
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
            instr_data.append({
                "audio_path":os.path.join(r, f),
                "instruction":cur_instr,
                "response":cur_class
            })
        logging.info(f"Created naive instruction data. {num_missing_labels}/{len(all_files)} files had missing labels.")
        return instr_data

    def create_jsonl_for_batch_inference(self, jsonl_par, jsonl_path = None):
        logging.info("Creating JSONL file for annotation through Gemini.")
        if jsonl_path is None:
            jsonl_path = os.path.join(jsonl_par, f"{self.dataset_name}.jsonl")
        if os.path.exists(jsonl_path):
            logging.warning(f"JSONL file already exists at {jsonl_path} for {self.dataset_name}")
            return jsonl_path
        all_files = []
        for r, ds, fs in os.walk(self.dataset_16khz_path):
            for f in fs:
                if (f.split(".")[-1] not in AUDIO_EXTENSIONS) or f[0] == ".":
                    continue
                all_files.append((r, f))

        dataset_gcloud_dir = self.dataset_16khz_path.split('/')[-1]
        len_local_data_dir_subpath = len(self.dataset_16khz_path)

        all_requests = []
        num_missing_labels = 0
        for idx, (r, f) in tqdm(enumerate(all_files), total = len(all_files)):
            cur_audio_subpath = os.path.join(r,f)[len_local_data_dir_subpath+1:]
            try:
                cur_label = self.get_label_from_fname(f"{dataset_gcloud_dir}/{cur_audio_subpath}")
            except:
                num_missing_labels+=1
                continue
            cur_prompt = self.gemini_prompt.replace(REPLACE_LABEL_STRING, cur_label)
            cur_gcloud_uri = f"gs://{BUCKET_NAME}/{BUCKET_AUDIO_FOLDER}/{dataset_gcloud_dir}/{cur_audio_subpath}"

            #### add request to all_requests
            cur_parts = [{"text":cur_prompt}]
            cur_parts.append({"fileData": {"fileUri": cur_gcloud_uri, "mimeType": "audio/wav"}})

            cur_request = {"request":{"contents": [{"role": "user", "parts": cur_parts}]}}
            all_requests.append(cur_request)
        logging.info(f"{num_missing_labels}/{len(all_files)} labels were missing while creating the jsonl files for the dataset...")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for req in all_requests:
                # Convert the dictionary to a JSON string and write it as a single line
                json_line = json.dumps(req)
                f.write(json_line + "\n")
        logging.info(f"Wrote JSONL file to {jsonl_path}")
        return jsonl_path

    def get_dataset_gemini_annotation_budget(self):
        logging.info(f"Computing total budget for gemini-2.0 flash for {self.dataset_name}. Estimated total output characters per response = 300. Change this number if you want to in the code.")
        logging.info(f"Checking total duration of all wav files in the dataset...")
        total_files, total_audio_secs = total_wav_duration(self.dataset_16khz_path)
        total_cost = total_audio_secs * GEMINI_PRICE["input_audio_per_sec"]
        total_cost += total_files * len(self.gemini_prompt) * GEMINI_PRICE["input_text_per_million_char"] / 1.e6
        total_cost += total_files * 300 * GEMINI_PRICE["output_text_per_million_char"] / 1.e6
        return total_cost

        

