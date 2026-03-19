## read dataset - done
## convert audio file to 16khz - done
## upload the audio files to google cloud for inference - done
## create jsonl for gemini to process - done
## run batch inference on created jsonl - done
## retrieve results from the inference - done
## postprocess the generated file - done

import os
import logging, json, time
from datetime import datetime
from openai import OpenAI
import argparse

from utils import (format_instruction_data, google_cloud_upload_file, google_cloud_gemini_batch_inference, google_cloud_directory_exists, format_dpo_data,
                    google_cloud_check_batch_prediction_result, google_cloud_download_file, get_res_from_jsonl, get_res_from_jsonl_gpt, split_gpt_requests)
from data_constants import BUCKET_NAME, BUCKET_AUDIO_FOLDER
from audio_datasets import EmovDB, IEMOCAP, TESS, OMGEmotion, MSPPodcast, AgeVoxCeleb, RAVDESS
from video_datasets import DFEW, MAFW, FERV39k, MER2025Track1Single, MER2025Track3Desc, CremaDVideo, RavdessVideo, MER2023Test1, MELD


MASTER_DATA_DIR = "/wekafs/ict/achaubey/emotion_reasoning/audio_exp/data"
MASTER_VIDEO_DATA_DIR = "/wekafs/ict/achaubey/emotion_reasoning/audio_exp/data_video"

logging.basicConfig(
    level=logging.INFO,                      # default threshold (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class DataPreprocess:
    def __init__(self, dataset_name, data_parent = None, dataset_path = None, use_gemini_annotation = True):
        self.dataset_name = dataset_name
        if data_parent is None and dataset_path is None:
            raise NotImplementedError
        if dataset_name == "emov_db":
            if dataset_path is None:
                dataset_path = os.path.join(data_parent, "emotion_datasets/emov-db")
                logging.info(f"You did not pass the dataset_path, so reading the dataset from possible directory - {dataset_path}")
            if not os.path.exists(dataset_path):
                raise Exception(f"We did not find the dataset at {dataset_path}. Please provide the dataset_path to DataPreprocess initialization")
            self.dataset = EmovDB(dataset_name, dataset_path)
        elif dataset_name == "iemocap":
            if dataset_path is None:
                dataset_path = os.path.join(data_parent, "emotion_datasets/IEMOCAP_full_release")
                logging.info(f"You did not pass the dataset_path, so reading the dataset from possible directory - {dataset_path}")
            if not os.path.exists(dataset_path):
                raise Exception(f"We did not find the dataset at {dataset_path}. Please provide the dataset_path to DataPreprocess initialization")
            self.dataset = IEMOCAP(dataset_name, dataset_path)
        elif dataset_name == "tess":
            if dataset_path is None:
                dataset_path = os.path.join(data_parent, "emotion_datasets/tess")
                logging.info(f"You did not pass the dataset_path, so reading the dataset from possible directory - {dataset_path}")
            if not os.path.exists(dataset_path):
                raise Exception(f"We did not find the dataset at {dataset_path}. Please provide the dataset_path to DataPreprocess initialization")
            self.dataset = TESS(dataset_name, dataset_path)
        elif dataset_name == "omg_emotion":
            if dataset_path is None:
                dataset_path = os.path.join(data_parent, "emotion_datasets/OMGEmotionChallenge")
                logging.info(f"You did not pass the dataset_path, so reading the dataset from possible directory - {dataset_path}")
            if not os.path.exists(dataset_path):
                raise Exception(f"We did not find the dataset at {dataset_path}. Please provide the dataset_path to DataPreprocess initialization")
            self.dataset = OMGEmotion(dataset_name, dataset_path)
        elif dataset_name == "msp_podcast":
            if dataset_path is None:
                dataset_path = os.path.join(data_parent, "emotion_datasets/MSP-Podcast")
                logging.info(f"You did not pass the dataset_path, so reading the dataset from possible directory - {dataset_path}")
            if not os.path.exists(dataset_path):
                raise Exception(f"We did not find the dataset at {dataset_path}. Please provide the dataset_path to DataPreprocess initialization")
            self.dataset = MSPPodcast(dataset_name, dataset_path)
        elif dataset_name == "ravdess":
            if dataset_path is None:
                dataset_path = os.path.join(data_parent, "emotion_datasets/ravdess")
                logging.info(f"You did not pass the dataset_path, so reading the dataset from possible directory - {dataset_path}")
            if not os.path.exists(dataset_path):
                raise Exception(f"We did not find the dataset at {dataset_path}. Please provide the dataset_path to DataPreprocess initialization")
            self.dataset = RAVDESS(dataset_name, dataset_path)
        elif dataset_name.startswith("age_vox"):
            if dataset_path is None:
                dataset_path = os.path.join(data_parent, "emotion_datasets/VoxCeleb2Audios")
                logging.info(f"You did not pass the dataset_path, so reading the dataset from possible directory - {dataset_path}")
            if not os.path.exists(dataset_path):
                raise Exception(f"We did not find the dataset at {dataset_path}. Please provide the dataset_path to DataPreprocess initialization")
            train = True if dataset_name.endswith("train") else False 
            self.dataset = AgeVoxCeleb(dataset_name, dataset_path, train=train)
        elif dataset_name == "dfew":
            if dataset_path is None:
                dataset_path = os.path.join(data_parent, "dfew/dfew_original_clips")
                logging.info(f"You did not pass the dataset_path, so reading the dataset from possible directory - {dataset_path}")
            if not os.path.exists(dataset_path):
                raise Exception(f"We did not find the dataset at {dataset_path}. Please provide the dataset_path to DataPreprocess initialization") 
            self.dataset = DFEW(dataset_name, dataset_path)
        elif dataset_name == "meld":
            if dataset_path is None:
                dataset_path = os.path.join(data_parent, "meld/output_repeated_splits_test")
                logging.info(f"You did not pass the dataset_path, so reading the dataset from possible directory - {dataset_path}")
            if not os.path.exists(dataset_path):
                raise Exception(f"We did not find the dataset at {dataset_path}. Please provide the dataset_path to DataPreprocess initialization") 
            self.dataset = MELD(dataset_name, dataset_path)
        elif dataset_name == "mer2025_single":
            if dataset_path is None:
                dataset_path = os.path.join(data_parent, "mer2025/video")
                logging.info(f"You did not pass the dataset_path, so reading the dataset from possible directory - {dataset_path}")
            if not os.path.exists(dataset_path):
                raise Exception(f"We did not find the dataset at {dataset_path}. Please provide the dataset_path to DataPreprocess initialization") 
            self.dataset = MER2025Track1Single(dataset_name, dataset_path)
        elif dataset_name == "mer2023_test1":
            if dataset_path is None:
                dataset_path = os.path.join(data_parent, "mer2023/MER2023/test1")
                logging.info(f"You did not pass the dataset_path, so reading the dataset from possible directory - {dataset_path}")
            if not os.path.exists(dataset_path):
                raise Exception(f"We did not find the dataset at {dataset_path}. Please provide the dataset_path to DataPreprocess initialization") 
            self.dataset = MER2023Test1(dataset_name, dataset_path)
        elif dataset_name == "mer2025_desc":
            if dataset_path is None:
                dataset_path = os.path.join(data_parent, "mer2025/video")
                logging.info(f"You did not pass the dataset_path, so reading the dataset from possible directory - {dataset_path}")
            if not os.path.exists(dataset_path):
                raise Exception(f"We did not find the dataset at {dataset_path}. Please provide the dataset_path to DataPreprocess initialization") 
            self.dataset = MER2025Track3Desc(dataset_name, dataset_path)
        elif dataset_name == "mafw":
            if dataset_path is None:
                dataset_path = os.path.join(data_parent, "mafw/mafw_clips")
                logging.info(f"You did not pass the dataset_path, so reading the dataset from possible directory - {dataset_path}")
            if not os.path.exists(dataset_path):
                raise Exception(f"We did not find the dataset at {dataset_path}. Please provide the dataset_path to DataPreprocess initialization") 
            self.dataset = MAFW(dataset_name, dataset_path)
        elif dataset_name == "ferv39k":
            if dataset_path is None:
                dataset_path = os.path.join(data_parent, "ferv39k/0_7_LabelClips")
                logging.info(f"You did not pass the dataset_path, so reading the dataset from possible directory - {dataset_path}")
            if not os.path.exists(dataset_path):
                raise Exception(f"We did not find the dataset at {dataset_path}. Please provide the dataset_path to DataPreprocess initialization") 
            self.dataset = FERV39k(dataset_name, dataset_path)
        elif dataset_name == "cremad_video":
            if dataset_path is None:
                dataset_path = os.path.join(data_parent, "cremad/cremad_video")
                logging.info(f"You did not pass the dataset_path, so reading the dataset from possible directory - {dataset_path}")
            if not os.path.exists(dataset_path):
                raise Exception(f"We did not find the dataset at {dataset_path}. Please provide the dataset_path to DataPreprocess initialization") 
            self.dataset = CremaDVideo(dataset_name, dataset_path)
        elif dataset_name == "ravdess_video":
            if dataset_path is None:
                dataset_path = os.path.join(data_parent, "RAVDESS/ravdess_videos")
                logging.info(f"You did not pass the dataset_path, so reading the dataset from possible directory - {dataset_path}")
            if not os.path.exists(dataset_path):
                raise Exception(f"We did not find the dataset at {dataset_path}. Please provide the dataset_path to DataPreprocess initialization") 
            self.dataset = RavdessVideo(dataset_name, dataset_path)
        else:
            raise NotImplementedError
    
    def upload_dataset_to_gcloud(self):
        self.dataset.upload_data_to_gcloud()
    
    def get_dataset_gemini_annotation_budget(self, gemini_mode = "description", gemini_version = "gemini-2.0-flash-001"):
        if gemini_mode in ["description", "caption"]:
            budget = self.dataset.get_dataset_gemini_annotation_budget(gemini_mode=gemini_mode, gemini_version="gemini-2.0-flash-001")
        elif gemini_mode == "qa":
            budget = self.dataset.get_dataset_gemini_annotation_budget(gemini_mode=gemini_mode, gemini_version="gemini-2.0-flash-lite")
        else:
            raise NotImplementedError(f"Unknown gemini mode - {gemini_mode}")
        # print(f"Total cost for running gemini inference is estimated as ${budget:.2f}")

    
    def get_instruction_data_from_gemini(self, gemini_mode = "description"):

        if self.dataset_name in ["mer2025_desc"]:
            raise NotImplementedError("Getting instruction data is not implemented for MER 2025 Track 3 Description Dataset.")

        ## upload the audio files to gemini if they are not already uploaded 
        if gemini_mode in ["description", "caption"]:
            self.dataset.upload_data_to_gcloud(audio_video_separate=False)
        elif gemini_mode == "modality_separate_caption":
            self.dataset.upload_data_to_gcloud(audio_video_separate=True)
        
        ## create jsonl for the current dataset
        jsonl_par_path = os.path.join(MASTER_DATA_DIR, "gemini_annotations")
        jsonl_path = self.dataset.create_jsonl_for_batch_inference(jsonl_par = jsonl_par_path, annotation_mode=gemini_mode)
        predictions_jsonl_path = ".".join(jsonl_path.split(".")[:-1])+"_predictions.jsonl"

        if not os.path.exists(predictions_jsonl_path):
            ## upload jsonl file to google cloud
            google_cloud_upload_file(jsonl_path, BUCKET_NAME, f"{BUCKET_AUDIO_FOLDER}/jsonl_files")
            gcloud_input_uri = f"gs://{BUCKET_NAME}/{BUCKET_AUDIO_FOLDER}/jsonl_files/{jsonl_path.split('/')[-1]}"
            gcloud_output_uri = ".".join(gcloud_input_uri.split(".")[:-1])+"_predictions"

            ## check if the predictions already exist in google cloud to avoid running multiple times
            output_pred_uri = google_cloud_check_batch_prediction_result(gcloud_output_uri)
            if not output_pred_uri:
                # check if the output_uri exists. If this exists and the output_pred_uri does not exist, then the batch prediction is under progress. Wait for it to complete.
                # TODO add a way to see the running batch processes.
                output_uri_directory_exists = google_cloud_directory_exists(BUCKET_NAME, f"{BUCKET_AUDIO_FOLDER}/jsonl_files/{jsonl_path.split('/')[-1].split('.')[0]}_predictions")
                if output_uri_directory_exists:
                    raise NotImplementedError("The predictions for batch inference do not exist, but batch inference has started. Wait for some time for the batch inference to complete.")
                else:
                    logging.info(f"Running gemini batch inference...")
                    if gemini_mode in ["description", "caption", "modality_separate_caption", "modality_qa", "predict_emotion_from_modality_captions", "modality_qa_all"]:
                        gemini_version = "gemini-2.0-flash-001"
                    else:
                        gemini_version = "gemini-2.0-flash-lite-001"
                    logging.info(f"Using gemini version - {gemini_version} for batch inference.")
                    ## run the batch inference on the jsonl file
                    output_pred_uri = google_cloud_gemini_batch_inference(gcloud_input_uri, gcloud_output_uri, gemini_version=gemini_version)
            else:
                output_pred_uri = "gs://"+output_pred_uri
                logging.warning(f"The predictions.jsonl file already exists at {output_pred_uri}. Not computing again...")
            
            ## download the predictions jsonl file to local
            google_cloud_download_file(output_pred_uri, predictions_jsonl_path)
        else:
            logging.warning(f"Gemini predictions are already saved at {predictions_jsonl_path}. Not rerunning gemini inference.")

        if gemini_mode in ["caption", "modality_separate_caption"]:
            logging.info(f"Since gemini_mode is {gemini_mode}, we only got the captions from the input video/audio files. Run the script again with different gemini_mode ('qa' or 'modality_qa') to obtain the instruction data in qa format.")
            return
        

        gemini_prediction_dict = get_res_from_jsonl(predictions_jsonl_path, gemini_mode=gemini_mode)
        if gemini_mode == "predict_emotion_from_modality_captions":
            save_path = os.path.join(MASTER_DATA_DIR, "emotion_prediction_results")
            os.makedirs(save_path, exist_ok=True)
            self.dataset.save_modality_emotion_prediction_results(gemini_prediction_dict, save_path=save_path)
        # print(gemini_prediction_dict)
        return self.dataset.get_gemini_instruction_format_data(gemini_prediction_dict, gemini_mode=gemini_mode)

    def get_instruction_data_from_gpt(self, gpt_mode = "modality_qa_all"):

        if self.dataset_name in ["mer2025_desc"]:
            raise NotImplementedError("Getting instruction data is not implemented for MER 2025 Track 3 Description Dataset.")

        if gpt_mode in ["modality_separate_caption"]:
            self.dataset.upload_frames_to_gcloud()

        ## create jsonl for the current dataset
        jsonl_par_path = os.path.join(MASTER_DATA_DIR, "gpt_annotations", "jsonl_files")
        jsonl_path = self.dataset.create_jsonl_for_batch_inference(jsonl_par = jsonl_par_path, annotation_mode=gpt_mode, api_mode="gpt")
        predictions_jsonl_path = ".".join(jsonl_path.split(".")[:-1])+"_predictions.jsonl"
        cur_gpt_log_file = os.path.join(MASTER_DATA_DIR, "gpt_annotations", f"log-{self.dataset_name}-{gpt_mode}.txt")

        if not os.path.exists(predictions_jsonl_path):
            
            client = OpenAI()

            ## create smaller batches of jsonl files to avoid hitting API limits
            split_jsonl_files = split_gpt_requests(jsonl_path)
            all_prediction_paths = []

            for split_jsonl_path in split_jsonl_files:
                cur_split_predictions_jsonl_path = ".".join(split_jsonl_path.split(".")[:-1])+"_predictions.jsonl"
                if os.path.exists(cur_split_predictions_jsonl_path):
                    all_prediction_paths.append(cur_split_predictions_jsonl_path)
                    continue

                batch_input_file = client.files.create(
                    file=open(split_jsonl_path, "rb"),
                    purpose="batch"
                )

                batch_input_file_id = batch_input_file.id
                with open(cur_gpt_log_file, "a") as f:
                    f.write(f"{datetime.now()} -- Running batch inference for {self.dataset_name} in {gpt_mode}--  Batch input file ID: {batch_input_file_id}\n")
                batch_inf = client.batches.create(
                    input_file_id=batch_input_file_id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={
                        "description": split_jsonl_path
                    }
                )
                batch_inf_id = batch_inf.id
                with open(cur_gpt_log_file, "a") as f:
                    f.write(f"{datetime.now()} -- Running batch inference for {self.dataset_name} in {gpt_mode} -- Batch inference ID: {batch_inf_id}\n")
                
                while batch_inf.status not in ["completed", "failed", "expired", "cancelled"]:
                    batch_inf = client.batches.retrieve(batch_inf.id)
                    print(f"[{split_jsonl_path.split('/')[-1]}] [{batch_inf.id}] Batch status: {batch_inf.status}")
                    time.sleep(5)

                if batch_inf.status == "completed":
                    print("Batch completed successfully!")
                    # Retrieve the results
                    batch_inf = client.batches.retrieve(batch_inf.id)
                    file_response = client.files.content(batch_inf.output_file_id)
                    
                    with open(cur_split_predictions_jsonl_path, "w") as f:
                        f.write(file_response.text)
                    all_prediction_paths.append(cur_split_predictions_jsonl_path)
                    logging.info(f"Output file saved as {cur_split_predictions_jsonl_path}")
                else:
                    batch_inf = client.batches.retrieve(batch_inf.id)
                    error_response = client.files.content(batch_inf.error_file_id)
                    raise Exception(f"Batch inference failed with error: {error_response.text}")
            
            # combine all the split jsonl files into one
            with open(predictions_jsonl_path, "w") as outfile:
                for prediction_path in all_prediction_paths:
                    with open(prediction_path, "r") as infile:
                        outfile.write(infile.read())
                    # os.remove(prediction_path)  # remove the split file after combining
        else:
            logging.warning(f"GPT predictions are already saved at {predictions_jsonl_path}. Not rerunning GPT inference.")

        if gpt_mode == "modality_separate_caption":
            return

        gpt_prediction_dict = get_res_from_jsonl_gpt(predictions_jsonl_path, annotation_mode=gpt_mode)

        if gpt_mode in ["modality_qa_all", "hallucination_qa", "hallucination_qa_extras"]:
            gemini_mode = "modality_qa"
        elif gpt_mode in ["av_long_caption_rewrite"]:
            gemini_mode = "description"
        return self.dataset.get_gemini_instruction_format_data(gpt_prediction_dict, gemini_mode=gemini_mode)


    def get_dpo_data(self, interest_category = "identification"):

        par_dir = os.path.join(MASTER_DATA_DIR, "dpo_data")
        os.makedirs(par_dir, exist_ok=True)
        save_path = os.path.join(MASTER_DATA_DIR, "dpo_data", f"{self.dataset_name}-{interest_category}.json")
        
        if os.path.exists(save_path):
            logging.warning(f"DPO dataset already exists for {self.dataset_name} at {save_path}.")
            
        
        jsonl_par_path = os.path.join(MASTER_DATA_DIR, "gemini_annotations")
        dataset_modality_qa_all_jsonl_path = os.path.join(jsonl_par_path, f"{self.dataset_name}_modality_qa_all_predictions.jsonl")
        if not os.path.exists(dataset_modality_qa_all_jsonl_path):
            raise Exception(f"Modality QA All jsonl file does not exist at {dataset_modality_qa_all_jsonl_path}. Please run the script to create the instruction data first.")

        raw_dpo_dataset = self.dataset.get_dpo_format_data(dataset_modality_qa_all_jsonl_path, interest_category=interest_category)

        formatted_dpo_dataset = format_dpo_data(raw_dpo_dataset, MASTER_VIDEO_DATA_DIR)

        with open(save_path, "w", encoding='utf-8') as f:
            json.dump(formatted_dpo_dataset, f, ensure_ascii=False, indent=4)
    

    def get_instruction_data(self, use_gemini = False, save_path = None, gemini_mode = "description", use_gpt = False, gpt_mode = None):
        if save_path is None:
            if not use_gemini and not use_gpt:
                instruction_data_json_path = os.path.join(MASTER_DATA_DIR, "instruction_data", f"{self.dataset_name}.json")
            else:
                if gemini_mode == "description":
                    instruction_data_json_path = os.path.join(MASTER_DATA_DIR, "instruction_data", f"{self.dataset_name}_gemini.json")
                elif gemini_mode == "caption":
                    instruction_data_json_path = os.path.join(MASTER_DATA_DIR, "instruction_data", f"{self.dataset_name}_gemini_caption.json")
                elif gemini_mode == "qa":
                    instruction_data_json_path = os.path.join(MASTER_DATA_DIR, "instruction_data", f"{self.dataset_name}_gemini_qa.json")
                elif gemini_mode == "modality_separate_caption":
                    instruction_data_json_path = os.path.join(MASTER_DATA_DIR, "instruction_data", f"{self.dataset_name}_gemini_modality_separate_caption.json")
                elif gemini_mode == "modality_qa":
                    instruction_data_json_path = os.path.join(MASTER_DATA_DIR, "instruction_data", f"{self.dataset_name}_gemini_modality_qa.json")
                elif gemini_mode == "modality_qa_all":
                    instruction_data_json_path = os.path.join(MASTER_DATA_DIR, "instruction_data", f"{self.dataset_name}_gemini_modality_qa_all.json")
                elif gemini_mode == "predict_emotion_from_modality_captions":
                    instruction_data_json_path = os.path.join(MASTER_DATA_DIR, "instruction_data", f"{self.dataset_name}_gemini_predict_emotion_from_modality_captions.json")
                else:
                    if use_gpt and gpt_mode == "modality_qa_all":
                        instruction_data_json_path = os.path.join(MASTER_DATA_DIR, "instruction_data", f"{self.dataset_name}_gpt_modality_qa_all.json")
                    elif use_gpt and gpt_mode == "modality_separate_caption":
                        instruction_data_json_path = os.path.join(MASTER_DATA_DIR, "instruction_data", f"{self.dataset_name}_gpt_modality_separate_caption.json")
                    elif use_gpt and gpt_mode == "hallucination_qa":
                        instruction_data_json_path = os.path.join(MASTER_DATA_DIR, "instruction_data", f"{self.dataset_name}_gpt_hallucination_qa.json")
                    elif use_gpt and gpt_mode == "hallucination_qa_extras":
                        instruction_data_json_path = os.path.join(MASTER_DATA_DIR, "instruction_data", f"{self.dataset_name}_gpt_hallucination_qa_extras.json")
                    elif use_gpt and gpt_mode == "av_long_caption_rewrite":
                        instruction_data_json_path = os.path.join(MASTER_DATA_DIR, "instruction_data", f"{self.dataset_name}_gpt_av_long_caption_rewrite.json")
                    else:
                        raise NotImplementedError(f"Unknown gemini mode - {gemini_mode}, or GPT mode - {gpt_mode} for instruction data creation.")
        else:
            instruction_data_json_path = save_path
        
        if os.path.exists(instruction_data_json_path):
            logging.warning(f"Instruction dataset already exists for {self.dataset_name} at {instruction_data_json_path}. If you want to create a new instruction data, pass a new save_path. ")
            return instruction_data_json_path

        dataset_parent_names = [base.__name__ for base in self.dataset.__class__.__bases__]
        # print(dataset_parent_names)
        if (not use_gemini) and (not use_gpt):
            if gemini_mode != "description":
                info.warning(f"You have provided some gemini_mode = {gemini_mode} but you are not using gemini for creating instruction data. Use --use_gemini to use gemini for creating instruction data.")
                raise NotImplementedError(f"Unknown gemini mode - {gemini_mode} for non-gemini instruction data creation.")
            unformatted_instr_data = self.dataset.get_naive_instruction_format_data()
        else:
            if use_gemini:
                unformatted_instr_data = self.get_instruction_data_from_gemini(gemini_mode=gemini_mode)
            elif use_gpt:
                if gpt_mode is None:
                    raise NotImplementedError("You have set use_gpt to True, but you have not provided gpt_mode. Please provide gpt_mode.")
                if gpt_mode in ["modality_qa_all", "modality_separate_caption", "hallucination_qa", "av_long_caption_rewrite", "hallucination_qa_extras"]:
                    unformatted_instr_data = self.get_instruction_data_from_gpt(gpt_mode=gpt_mode)
                else:
                    raise NotImplementedError(f"Unknown GPT mode - {gpt_mode} for instruction data creation.")
        if (gemini_mode in ["caption", "modality_separate_caption"]) or (gpt_mode in ["modality_separate_caption"]):
            return
        logging.info("Formatting the instruction data...")
        if "RawVideoDataset" in dataset_parent_names:
            formatted_instr_data = format_instruction_data(unformatted_instr_data, MASTER_VIDEO_DATA_DIR)
        else:
            formatted_instr_data = format_instruction_data(unformatted_instr_data, MASTER_DATA_DIR)
        
        with open(instruction_data_json_path, "w", encoding='utf-8') as f:
            json.dump(formatted_instr_data, f, ensure_ascii=False, indent=4)
        
        return instruction_data_json_path
        

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for running a benchmark/eval script.
    """
    parser = argparse.ArgumentParser(
        description="Convert emotion datasets to instruction data for tuning."
    )

    # --- Required / core paths -------------------------------------------------
    parser.add_argument(
        "--mode",
        type=str,
        default="gemini_budget",
        choices=["gemini_budget", "generate_instruction_data", "generate_dpo_data"],
        help="Name of the dataset to process.",
    )

    # --- Required / core paths -------------------------------------------------
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["emov_db", "iemocap", "tess", "omg_emotion", "msp_podcast", "age_vox-train", "age_vox-test", "ravdess", "dfew", "meld", "mafw", "ferv39k", "mer2025_single", "mer2025_desc", "cremad_video", "ravdess_video", "mer2023_test1"],
        help="Name of the dataset to process.",
    )

    # --- Optional modifiers ----------------------------------------------------
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=False,
        help="Identifier of the *base* model in case of only projector or lora"
             "Leave empty if model_path already contains all weights.",
    )

    # --- Benchmark selection ---------------------------------------------------
    parser.add_argument(
        "--use_gemini",
        action="store_true",
        help="Whether to use gemini for creating instruction data or not",
    )

    parser.add_argument(
        "--gemini_mode",
        type=str,
        default=None,
        choices=[None, "description", "caption", "qa", "modality_separate_caption", "modality_qa", "predict_emotion_from_modality_captions", "modality_qa_all"],
    )

    parser.add_argument(
        "--use_gpt",
        action="store_true",
        help="Whether to use GPT for creating instruction data or not",
    )

    parser.add_argument(
        "--gpt_mode",
        type=str,
        default=None,
        choices=[None, "modality_qa_all", "modality_separate_caption", "hallucination_qa", "av_long_caption_rewrite", "hallucination_qa_extras"],
    )

    return parser.parse_args()

def main():

    args = parse_args()

    dp_obj = DataPreprocess(args.dataset_name, dataset_path = args.dataset_path)
    # dp_obj.upload_dataset_to_gcloud()
    if args.mode == "gemini_budget":
        dp_obj.get_dataset_gemini_annotation_budget(gemini_mode=args.gemini_mode)
    elif args.mode == "generate_instruction_data":
        dp_obj.get_instruction_data(use_gemini=args.use_gemini, gemini_mode=args.gemini_mode, use_gpt=args.use_gpt, gpt_mode=args.gpt_mode)
    elif args.mode == "generate_dpo_data":
        dp_obj.get_dpo_data()
    else:
        raise NotImplementedError(f"Unknown mode for data preprocessing ---- {args.mode}")

if __name__=="__main__":
    main()