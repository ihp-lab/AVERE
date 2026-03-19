from videollava.model.multimodal_encoder.whisper import WhisperSpeechTower
import json
import os
import soundfile as sf
import numpy as np
import torch
from tqdm import tqdm
from multiprocessing import Pool
import random

## initialize whisper speech tower
def get_speech_processor():

    speech_tower = WhisperSpeechTower(speech_tower = "./backbones/whisper-large-v3", args="", cache_dir='./cache_dir')

    speech_processor = speech_tower.speech_processor

    return speech_processor

if __name__ == "__main__":
    # initialize the speech processor
    speech_processor = get_speech_processor()

    ## load the json file for processing
    data_json_path = "/wekafs/ict/achaubey/emotion_reasoning/audio_exp/data_preprocess/dpo_files/mafw_mer2025single_classification.json"
    with open(data_json_path, "r", encoding='utf-8') as f:
        data = json.load(f)
    print(f"Total samples in the data: {len(data)}")

    new_data = []

    num_processes = 4  # specify the number of parallel processes

    def init_worker():
        global speech_processor
        speech_processor = get_speech_processor()

    def process_sample(sample):
        if "audio" in sample and "processed_audio" not in sample:
            try:
                if "video" in sample:
                    audio_path = os.path.join("/wekafs/ict/achaubey/emotion_reasoning/audio_exp/data_video", sample["audio"])
                    audio_path_2 = os.path.join("/wekafs/ict/achaubey/emotion_reasoning/audio_exp/data_video", sample["audio_l"])
                else:
                    audio_path = os.path.join("/wekafs/ict/achaubey/emotion_reasoning/audio_exp/data", sample["audio"])
                assert os.path.exists(audio_path), f"Audio file does not exist: {audio_path}"
                # par_audio_dir_name = sample["audio"].split("/")[1]
                # processed_audio_save_path = audio_path.replace(par_audio_dir_name, f"{par_audio_dir_name}_processed").replace(".wav", ".npy")
                # if not os.path.exists(processed_audio_save_path):
                processed_audio = speech_processor(audio_path, return_tensors="pt")["spectrogram"][0].numpy()
                processed_audio_2 = speech_processor(audio_path_2, return_tensors="pt")["spectrogram"][0].numpy()
                #     os.makedirs(os.path.dirname(processed_audio_save_path), exist_ok=True)
                #     np.save(processed_audio_save_path, processed_audio)
                # sample["processed_audio"] = sample["audio"].replace(par_audio_dir_name, f"{par_audio_dir_name}_processed").replace(".wav", ".npy")
            except Exception as e:
                print(f"Error processing sample {sample['audio']}: {e}")
                with open("errored_read_audios_dpo.txt", "a") as f:
                    f.write(f"{sample['audio']}: {e}\n")
                return None
        return sample

    # with Pool(num_processes, initializer=init_worker) as pool:
    #     new_data = list(tqdm(pool.imap(process_sample, data), total=len(data)))
    for sample in tqdm(data):
        processed_sample = process_sample(sample)
        if processed_sample:
            new_data.append(processed_sample)
    # random.shuffle(new_data)
    
    # ## save the new json data file
    new_data_json_path = data_json_path.replace(".json", "-audio_error_filtered.json")
    with open(new_data_json_path, "w", encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)
    # print(f"Processed data saved to: {new_data_json_path}")
