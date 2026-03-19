from avere.model.multimodal_encoder.languagebind.video.processing_video import LanguageBindVideoProcessor
import json
import os
import soundfile as sf
import numpy as np
import torch
from tqdm import tqdm
from multiprocessing import Pool

class VisionConfig:
    def __init__(self):
        self.video_decode_backend = 'decord'
        self.num_frames = 8  # Example value, adjust as needed

class dummyConfig:
    def __init__(self):
        self.vision_config = VisionConfig()

def get_video_processor():
    # Initialize the video processor with the dummy config
    config = dummyConfig()
    video_processor = LanguageBindVideoProcessor(config=config)
    return video_processor


if __name__ == "__main__":
    # initialize the video processor
    video_processor = get_video_processor()

    ## load the json file for processing
    # data_json_path = "/wekafs/ict/achaubey/emotion_reasoning/audio_exp/data_preprocess/instruct_files/ferv39k_mafw_mer2025_single_descraw_cremad_audio_datasets-naive_and_gemini-mira_movies_qa.json"
    data_json_path = "/wekafs/ict/achaubey/emotion_reasoning/audio_exp/data_preprocess/instruct_files/ferv39k_mafw_mer2025_single_descraw_cremad_datasets-naive_gemini-ferv39k_mafw_mer2025_single-gemini_qa.json"
    with open(data_json_path, "r", encoding='utf-8') as f:
        data = json.load(f)
    print(f"Total samples in the data: {len(data)}")

    new_data = []

    def process_sample(sample):
        if "video" in sample and "processed_video" not in sample:
            try:
                video_path = os.path.join("/wekafs/ict/achaubey/emotion_reasoning/audio_exp/data_video", sample["video"])
                assert os.path.exists(video_path), f"Video file does not exist: {video_path}"
                par_video_dir_name = sample["video"].split("/")[1]
                assert video_path.endswith(".mp4"), f"Video file is not in mp4 format: {video_path}"
                processed_video_save_path = video_path.replace(par_video_dir_name, f"{par_video_dir_name}_processed").replace(".mp4", ".npy")
                if not os.path.exists(processed_video_save_path):
                    processed_video = video_processor(video_path, return_tensors="pt")["pixel_values"][0].numpy()
                    os.makedirs(os.path.dirname(processed_video_save_path), exist_ok=True)
                    np.save(processed_video_save_path, processed_video)
                sample["processed_video"] = sample["video"].replace(par_video_dir_name, f"{par_video_dir_name}_processed").replace(".mp4", ".npy")
            except:
                return sample
        return sample

    # with Pool(num_processes, initializer=init_worker) as pool:
    #     new_data = list(tqdm(pool.imap(process_sample, data), total=len(data)))
    for sample in tqdm(data):
        processed_sample = process_sample(sample)
        if processed_sample:
            new_data.append(processed_sample)
    
    ## save the new json data file
    new_data_json_path = data_json_path.replace(".json", "_video_processed.json")
    with open(new_data_json_path, "w", encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)
    print(f"Processed data saved to: {new_data_json_path}")
