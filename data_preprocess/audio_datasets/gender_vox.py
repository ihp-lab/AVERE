from data_constants import EMOTION_INSTRUCTIONS, REPLACE_LABEL_STRING
from .raw_dataset import RawAudioDataset
from utils import convert_to_wav_parallel

import logging, os, glob
from tqdm import tqdm


class GenderVoxCeleb(RawAudioDataset):
    def __init__(self, dataset_name, dataset_path, train=True):
        self.task_type = "gender"
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        if dataset_path.endswith("16khz"): ## checks if the dataset is already processed to 16khz
            self.dataset_16khz_path = self.dataset_path
        else:
            self.dataset_16khz_path = self.convert_to_16khz(self.dataset_path)
        
        self.wav_depth = 4

        metadata_path = os.path.join(self.dataset_path, "vox2_meta.csv")
        metadata = pd.read_csv(metadata_path)
        self.label_dict = {}
        gender_map = {"f":"female", "m":"male"}
        for idx, row in metadata.iterrows():
            speaker_id = row["VoxCeleb2 ID"]
            gender = row["Gender"]
            gender = gender_map[gender]
            self.label_dict[speaker_id] = gender
        

    def get_label_from_fname(self, fpath):
        speaker_id = fpath.split("/")[-3]
        return self.label_dict[speaker_id]

    def convert_to_16khz(self, dataset_path):
        dataset_16khz_path = dataset_path+"_16khz"
        if os.path.exists(dataset_16khz_path):
            logging.warning(f"Dataset conversion path to 16khz exists at {dataset_16khz_path}. Already converted files will be skipped.")
            # return dataset_16khz_path
        logging.info(f"Reading all the mp4 files inside the dataset directory - {dataset_path}")

        # read all the files first so that we can use a progress bar
        mp4_files = glob.glob(f"{self.dataset_path}/videos/*/*/*.mp4")
        all_files = []
        for video in mp4_files:
            if video[0]==".":
                continue
            all_files.append((os.path.dirname(video), os.path.basename(video)))

        # converting the files to 16khz
        logging.info(f"Converting all the mp4 files inside the dataset directory - {dataset_path}")
        convert_to_wav_parallel(all_files, dataset_path, dataset_16khz_path)
        logging.info(f"Conversion complete. Saving files inside the directory - {dataset_16khz_path}")
        return dataset_16khz_path

    def get_gemini_instruction_format_data(self, gemini_predictions):

        raise NotImplementedError
