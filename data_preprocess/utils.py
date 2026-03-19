from pathlib import Path
from typing import Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_constants import VIDEO_EXTENSIONS

import sys
import shutil
import subprocess
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from collections import defaultdict
import random


from tqdm import tqdm                    # ⬅ nice progress bar (optional)

try:
    from google.cloud import storage        # ⬅ official GCS client
    import vertexai
    from vertexai.batch_prediction import BatchPredictionJob
except ImportError:
    print("Google Cloud libraries not found...")
    print("You must switch environment to the correct one before using google cloud libraries.")
    
import fsspec


import os
import shlex
import subprocess
import time
import json
import wave

def google_cloud_directory_exists(bucket_name: str, prefix: str) -> bool:
    """
    Return True if *prefix* (e.g. "data/2025/") exists in the bucket.

    A single lookup is enough: we ask for 1 object whose name starts
    with the prefix; if we get one, the folder is present.
    """
    client = storage.Client()                  # uses ADC / service acct
    bucket = client.bucket(bucket_name)

    # `max_results=1` keeps it cheap.
    iterator = bucket.list_blobs(prefix=prefix, max_results=1)
    return any(iterator)

def google_cloud_upload_directory(
    local_dir: Union[str, Path],
    bucket_name: str,
    gcs_prefix: str,                # e.g. "backup/2025-04-29"
    threads: int = 8                     # parallel uploads
) -> None:
    local_dir = Path(local_dir).resolve()
    if not local_dir.is_dir():
        raise ValueError(f"{local_dir} is not a directory")

    client = storage.Client()            # uses ADC or service-account JSON
    bucket = client.bucket(bucket_name)

    # Collect every *file* under the directory
    files = [p for p in local_dir.rglob("*") if p.is_file()]

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

def google_cloud_upload_file(local_file: str, bucket_name: str, gcs_path_prefix: str):
    """
    Upload a file to a Google Cloud Storage bucket under a given prefix.

    Parameters:
        local_file (str): Path to the local file to upload.
        bucket_name (str): Target GCS bucket.
        gcs_path_prefix (str): "Folder" path inside the bucket (e.g., 'my_folder/subdir')
    """
    client = storage.Client()                       # uses ADC or env var
    bucket = client.bucket(bucket_name)
    
    local_file_path = Path(local_file).resolve()
    blob_name = f"{gcs_path_prefix.strip('/')}/{local_file_path.name}"
    
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_file_path))

    print(f"✅ Uploaded {local_file_path.name} to gs://{bucket_name}/{blob_name}")

def google_cloud_gemini_batch_inference(input_uri, output_uri, gemini_version = "gemini-2.0-flash-001"):
    PROJECT_ID = "usc-ict-gcp-bgp"

    # ValueError: Unsupported region for Vertex AI, select from frozenset({'me-central1', 'us-west3', 'asia-south1', 'europe-west2', 'europe-west3', 
    # 'europe-southwest1', 'europe-west4', 'us-west4', 'southamerica-west1', 'southamerica-east1', 'asia-east1', 'me-central2', 'asia-northeast1', 
    # 'us-west2', 'africa-south1', 'northamerica-northeast1', 'us-east4', 'northamerica-northeast2', 'europe-central2', 'us-west1', 'europe-west1', 
    # 'us-south1', 'australia-southeast1', 'europe-west12', 'us-east1', 'asia-northeast2', 'asia-east2', 'asia-southeast1', 'me-west1', 'asia-southeast2', 
    # 'europe-west8', 'europe-west6', 'us-central1', 'europe-north1', 'us-east5', 'asia-northeast3', 'australia-southeast2', 'europe-west9'})

    # Initialize vertexai
    vertexai.init(project=PROJECT_ID, location="us-central1")

    # Submit a batch prediction job with Gemini model
    batch_prediction_job = BatchPredictionJob.submit(
        source_model=gemini_version,
        input_dataset=input_uri,
        output_uri_prefix=output_uri,
    )

    # Check job status
    print(f"Job resource name: {batch_prediction_job.resource_name}")
    print(f"Model resource name with the job: {batch_prediction_job.model_name}")
    print(f"Job state: {batch_prediction_job.state.name}")

    # Refresh the job until complete
    while not batch_prediction_job.has_ended:
        time.sleep(5)
        print("Ended?", batch_prediction_job.has_ended)
        batch_prediction_job.refresh()

    # Check if the job succeeds
    if batch_prediction_job.has_succeeded:
        print("Job succeeded!")
    else:
        print(f"Job failed: {batch_prediction_job.error}")
        return None

    # Check the location of the output
    output_location = f"{batch_prediction_job.output_location}/predictions.jsonl"
    print(f"Job output location: {output_location}")

    return output_location

def google_cloud_check_batch_prediction_result(output_uri):
    fs = fsspec.filesystem("gcs")

    file_paths = fs.glob(f"{output_uri}/*/predictions.jsonl")
    return file_paths[0] if len(file_paths)>0 else None


def google_cloud_download_file(gcs_uri: str, local_path: str):
    """
    Downloads a GCS file given its URI (gs://bucket/path/to/file) to a local path.
    
    Args:
        gcs_uri (str): Full GCS URI like 'gs://my-bucket/folder/file.txt'
        local_path (str): Desired local path to save the file (e.g., './downloads/file.txt')
    """
    assert gcs_uri.startswith("gs://"), "Invalid GCS URI"
    print("gcs_uri", gcs_uri)
    # Parse GCS URI
    _, _, bucket_name, *object_parts = gcs_uri.split("/")
    object_path = "/".join(object_parts)
    print("BUCKET_NAME ==>", bucket_name)

    # Set up GCS client and download
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_path)

    # Ensure local path's parent directories exist
    local_path = Path(local_path).resolve()
    local_path.parent.mkdir(parents=True, exist_ok=True)

    blob.download_to_filename(str(local_path))
    print(f"✅ Downloaded {gcs_uri} → {local_path}")

def extra_path(parent: Union[str, Path], child: Union[str, Path]) -> str:
    """
    Return the part of *child* that is below *parent*.
    Raises ValueError if *child* is not inside *parent*.
    Always prefixes the result with the platform’s path separator.
    """
    parent = Path(parent).resolve()
    child  = Path(child).resolve()

    try:
        relative = child.relative_to(parent)          # pathlib ≥ 3.9
    except ValueError:
        raise ValueError(f"{child} is not inside {parent}")

    return f"{Path().anchor}{relative}" if parent.anchor else f"{Path().root}{relative}"
    # or simply:
    # return parent.joinpath(relative).as_posix().removeprefix(parent.as_posix())


def convert_to_wav_parallel(
    all_files: list[tuple[str, str]],
    dataset_path: Union[str, os.PathLike],
    dataset_16khz_path: Union[str, os.PathLike],
    *,
    threads: int = 8,
) -> None:
    """
    Convert every audio file in *all_files* to 16-kHz mono WAV in parallel.

    Parameters
    ----------
    all_files            : list of (root_dir, filename) tuples (e.g. from os.walk)
    dataset_path         : top-level source directory (needed by extra_path)
    dataset_16khz_path   : top-level destination directory
    threads              : number of worker threads (CPU-bound? choose 1; I/O bound: 8-64)
    """
    src_root   = Path(dataset_path).resolve()
    dst_root   = Path(dataset_16khz_path).resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    def _one_job(root_dir: str, fname: str) -> None:
        """Convert a single file if the target WAV does not already exist."""
        cur_fpath = Path(root_dir) / fname
        subdir_extra = extra_path(src_root, root_dir)      # keeps relative layout
        convert_dir = dst_root / subdir_extra.lstrip(os.sep)
        convert_dir.mkdir(parents=True, exist_ok=True)

        wav_name = f"{cur_fpath.stem}.wav"
        convert_fpath = convert_dir / wav_name
        if convert_fpath.exists():
            return                                           # already converted

        # Build a safe ffmpeg command (shell=False avoids injection issues)
        cmd = [
            "ffmpeg",
            "-i", str(cur_fpath),
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            str(convert_fpath),
            "-loglevel", "warning",
        ]
        subprocess.run(cmd, check=True)

    # --- parallel fan-out with a progress bar --------------------------------
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [pool.submit(_one_job, r, f) for r, f in all_files]
        for _ in tqdm(as_completed(futures), total=len(futures),
                      desc="Converting files to wav", unit="file"):
            pass   # any exception propagates here and stops everything

def convert_to_wav_if_audio_stream(
    all_files: list[tuple[str, str]],
    dataset_path: Union[str, os.PathLike],
    dataset_16khz_path: Union[str, os.PathLike],
    *,
    threads: int = 8,
) -> None:
    """
    Convert every audio file in *all_files* to 16-kHz mono WAV in parallel,
    but only if the input file has an audio stream.

    Parameters
    ----------
    all_files            : list of (root_dir, filename) tuples (e.g. from os.walk)
    dataset_path         : top-level source directory (needed by extra_path)
    dataset_16khz_path   : top-level destination directory
    threads              : number of worker threads (CPU-bound? choose 1; I/O bound: 8-64)
    """
    src_root   = Path(dataset_path).resolve()
    dst_root   = Path(dataset_16khz_path).resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    def _has_audio_stream(file_path: Path) -> bool:
        """Check if the file has an audio stream using ffprobe."""
        cmd = [
            "ffprobe",
            "-i", str(file_path),
            "-show_streams",
            "-select_streams", "a",
            "-loglevel", "error",
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return bool(result.stdout.strip())  # If output exists, audio stream is present

    def _one_job(root_dir: str, fname: str) -> None:
        """Convert a single file if it has an audio stream and the target WAV does not already exist."""
        cur_fpath = Path(root_dir) / fname
        subdir_extra = extra_path(src_root, root_dir)      # keeps relative layout
        convert_dir = dst_root / subdir_extra.lstrip(os.sep)
        convert_dir.mkdir(parents=True, exist_ok=True)

        wav_name = f"{cur_fpath.stem}.wav"
        convert_fpath = convert_dir / wav_name
        if convert_fpath.exists():
            return                                           # already converted

        if not _has_audio_stream(cur_fpath):
            print(f"⚠️ Skipping {cur_fpath}: no audio stream found.")
            return

        # Build a safe ffmpeg command (shell=False avoids injection issues)
        cmd = [
            "ffmpeg",
            "-i", str(cur_fpath),
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            str(convert_fpath),
            "-loglevel", "warning",
        ]
        subprocess.run(cmd, check=True)

    # --- parallel fan-out with a progress bar --------------------------------
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [pool.submit(_one_job, r, f) for r, f in all_files]
        for _ in tqdm(as_completed(futures), total=len(futures),
                      desc="Converting files to wav", unit="file"):
            pass   # any exception propagates here and stops everything

def find_video_files_to_convert(src_dir, dst_dir):
    to_convert = []

    for root, _, files in os.walk(src_dir):
        for file in files:
            if Path(file).suffix.lower() in [f".{vext}" for vext in VIDEO_EXTENSIONS]:
                src_path = Path(root) / file
                rel_path = src_path.relative_to(src_dir)
                out_path = dst_dir / rel_path.with_suffix(".mp4")

                if not out_path.exists():
                    to_convert.append((src_path, src_dir, dst_dir))

    return to_convert

def convert_video(args):
    src_path, src_dir, dst_dir = args
    rel_path = src_path.relative_to(src_dir)
    out_path = dst_dir / rel_path.with_suffix(".mp4")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg", "-y",
        "-i", str(src_path),
        "-r", "24",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k", "-ar", "16000", "-ac", "1",
        str(out_path)
    ]
    
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

def convert_video_wo_audio(args):
    src_path, src_dir, dst_dir = args
    rel_path = src_path.relative_to(src_dir)
    out_path = dst_dir / rel_path.with_suffix(".mp4")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg", "-y",
        "-i", str(src_path),
        "-r", "24",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-an",  # No audio stream
        str(out_path)
    ]
    
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

def convert_to_24fps_parallel(src_dir, dst_dir, num_workers=None, with_audio=True):
    src_dir = Path(src_dir).resolve()
    dst_dir = Path(dst_dir).resolve()

    dst_dir.mkdir(parents=True, exist_ok=True)

    args = find_video_files_to_convert(src_dir, dst_dir)

    print(f"Found {len(args)} video files to convert.")
    print(f"Converting with audio?:  {with_audio}")

    if with_audio:
        with Pool(processes=num_workers or cpu_count()) as pool:
            results = list(tqdm(pool.imap_unordered(convert_video, args), total=len(args)))
    else:
        with Pool(processes=num_workers or cpu_count()) as pool:
            results = list(tqdm(pool.imap_unordered(convert_video_wo_audio, args), total=len(args)))

    print(f"✅ Conversion completed. {sum(results)}/{len(results)} succeeded.")


N_FRAMES = 8
_EPS = 1e-3  # to avoid seeking exactly at video end
EXTS = [f".{vext}".lower() for vext in VIDEO_EXTENSIONS]

# def _ffprobe_duration_seconds(src_path: Path):
#     """
#     Return video duration in seconds using ffprobe, or None if unavailable.
#     """
#     # Prefer container (format) duration; it's generally reliable and fast.
#     cmd = [
#         "ffprobe", "-v", "error",
#         "-show_entries", "format=duration",
#         "-of", "default=nokey=1:noprint_wrappers=1",
#         str(src_path)
#     ]
#     try:
#         out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf-8").strip()
#         if out and out.lower() != "n/a":
#             return float(out)
#     except Exception:
#         pass

#     # Fallback: try stream duration
#     cmd = [
#         "ffprobe", "-v", "error",
#         "-select_streams", "v:0",
#         "-show_entries", "stream=duration",
#         "-of", "default=nokey=1:noprint_wrappers=1",
#         str(src_path)
#     ]
#     try:
#         out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf-8").strip()
#         if out and out.lower() != "n/a":
#             return float(out)
#     except Exception:
#         pass

#     return None

def _frames_already_exist(dst_dir: Path, rel_path: Path, n_frames: int = N_FRAMES) -> bool:
    """
    Check whether all expected frame files already exist.
    """
    base_dir = dst_dir / rel_path.parent
    stem = rel_path.stem
    return all((base_dir / f"{stem}--{i}.png").exists() for i in range(n_frames))

def find_video_files_to_extract(src_dir, dst_dir, n_frames: int = N_FRAMES):
    """
    Walk src_dir and return a list of videos that still need frame extraction
    (i.e., fewer than n_frames outputs exist in the mirrored dst path).
    """
    src_dir = Path(src_dir).resolve()
    dst_dir = Path(dst_dir).resolve()

    to_process = []
    for root, _, files in os.walk(src_dir):
        for file in files:
            if Path(file).suffix.lower() in EXTS:
                src_path = Path(root) / file
                rel_path = src_path.relative_to(src_dir)
                if not _frames_already_exist(dst_dir, rel_path, n_frames=n_frames):
                    to_process.append((src_path, src_dir, dst_dir, n_frames))
    return to_process

def _ffprobe_duration_seconds(src_path: Path):
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nokey=1:noprint_wrappers=1",
        str(src_path)
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        return float(out) if out and out.lower() != "n/a" else None
    except Exception:
        return None

def _ffprobe_fps(src_path: Path):
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate",
        "-of", "default=nokey=1:noprint_wrappers=1",
        str(src_path)
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        if "/" in out:
            num, den = out.split("/")
            den = float(den) if float(den) != 0 else 1.0
            return float(num) / den
        return float(out)
    except Exception:
        return None

def _compute_midpoint_timestamps(duration: float, n: int, fps) -> list[float]:
    """
    Uniform bins with midpoint sampling to avoid t==0 and t==duration.
    Also clamp to (duration - 0.5 frame) if fps is known.
    """
    ts = [((i + 0.5) / n) * duration for i in range(n)]
    if fps and fps > 0:
        max_t = max(0.0, duration - (0.5 / fps))
        ts = [min(t, max_t) for t in ts]
    else:
        # conservative clamp if fps unknown
        max_t = max(0.0, duration - 0.05)
        ts = [min(t, max_t) for t in ts]
    return ts

def extract_8_uniform_frames_from_video(args) -> bool:
    """
    Robust 8-frame extractor (midpoint sampling + accurate seek).
    Saves: <dst_dir>/<subdirs>/<stem>--{0..7}.png
    """
    src_path, src_dir, dst_dir, n_frames = args
    rel_path = src_path.relative_to(src_dir)
    out_dir = dst_dir / rel_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = rel_path.stem

    duration = _ffprobe_duration_seconds(src_path)
    if not duration or duration <= 0:
        return False

    fps = _ffprobe_fps(src_path)
    ts_list = _compute_midpoint_timestamps(duration, n_frames, fps)

    # Accurate seek: place -ss AFTER -i (decodes up to t, slower but reliable)
    for i, t in enumerate(ts_list):
        out_path = out_dir / f"{stem}--{i}.png"
        # Try accurate seek; if it fails, retry slightly earlier
        tried = [t, max(0.0, t - (1.0 / (fps if fps and fps > 0 else 24.0)))]
        success = False
        for tt in tried:
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(src_path),
                "-ss", f"{tt:.3f}",
                "-frames:v", "1",
                str(out_path),
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                success = True
                break
            except subprocess.CalledProcessError:
                continue
        if not success:
            return False
    return True

def extract_frames_parallel(src_dir, dst_dir, num_workers=None, n_frames: int = N_FRAMES):
    """
    Orchestrates parallel extraction over all videos under src_dir.
    """
    src_dir = Path(src_dir).resolve()
    dst_dir = Path(dst_dir).resolve()
    dst_dir.mkdir(parents=True, exist_ok=True)

    args = find_video_files_to_extract(src_dir, dst_dir, n_frames=n_frames)
    print(f"Found {len(args)} video files to process.")
    if not args:
        print("Nothing to do.")
        return

    with Pool(processes=num_workers or cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(extract_8_uniform_frames_from_video, args), total=len(args)))

    print(f"✅ Frame extraction completed. {sum(results)}/{len(results)} succeeded.")


def format_instruction_data(instr_samples, parent_path):
    formatted_samples = []

    ## calculate length of common path
    len_parent_path = len(parent_path)

    idx = 0
    for inst in tqdm(instr_samples):
        cur_sample = {
            "id": idx,
            "conversations": [
                {
                "from": "human",
                "value": inst["instruction"]
                },
                {
                "from": "gpt",
                "value": inst["response"]
                }
            ]
        }
        # "audio": inst["audio_path"][len_parent_path+1:],
        if "audio_path" in inst:
            cur_sample["audio"] = inst["audio_path"][len_parent_path+1:]
        if "video_path" in inst:
            cur_sample["video"] = inst["video_path"][len_parent_path+1:]

        ## fix for the case when both video and audio are present in the video dataset
        if "video_path" in inst and "audio_path" in inst:
            if "<audio>" not in cur_sample["conversations"][0]["value"]:
                cur_sample["conversations"][0]["value"] = cur_sample["conversations"][0]["value"].replace("<video>","<video>\n<audio>")
        
        idx+=1
        formatted_samples.append(cur_sample)

    return formatted_samples

def format_dpo_data(dpo_samples, parent_path):
    formatted_samples = []
    random.shuffle(dpo_samples)  # Shuffle the samples for better training

    ## calculate length of common path
    len_parent_path = len(parent_path)
    missing_paths = 0
    for inst in tqdm(dpo_samples):
        new_inst = {
            "prompt": inst["prompt"],
            "chosen": inst["chosen"],
            "rejected": inst["rejected"]
        }
        ## check if all the paths exist
        for k, v in inst.items():
            if "path" in k:
                    if not os.path.exists(v):
                        # print(f"Path {v} does not exist for key {k} in DPO sample {inst}.")
                        missing_paths += 1
                        continue
        new_inst["video"] = inst["video_path"][len_parent_path+1:]
        new_inst["audio"] = inst["audio_path"][len_parent_path+1:]
        new_inst["video_l"] = inst["rejected_video_path"][len_parent_path+1:]
        new_inst["audio_l"] = inst["rejected_audio_path"][len_parent_path+1:]

        formatted_samples.append(new_inst)
    print(f"⚠️ {missing_paths}/{len(dpo_samples)} paths were missing in the DPO samples. Please check the dataset.")
    return formatted_samples

def read_jsonl(jsonl_path):
    data = []
    with open(jsonl_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def trim_junk(string):
    start_idx = string.find("{")
    end_idx = string.find("}")
    return string[start_idx:end_idx+1]

def get_res_from_jsonl(pred_jsonl, gemini_mode = "description"):
    if gemini_mode not in ["description", "caption", "qa", "modality_separate_caption", "modality_qa", "predict_emotion_from_modality_captions", "modality_qa_all"]:
        raise ValueError(f"Unknown gemini mode: {gemini_mode}. Supported modes are 'description', 'caption' and 'qa'.")
    response_json = read_jsonl(pred_jsonl)
    if gemini_mode in ["description", "caption"]:
        pred_dict = {}
    elif gemini_mode in ["qa", "modality_qa", "modality_qa_all"]:
        pred_dict = defaultdict(list)
    elif gemini_mode in ["modality_separate_caption", "predict_emotion_from_modality_captions"]:
        pred_dict_video = {}
        pred_dict_audio = {}
    incorr, tot = 0, 0
    for response in response_json:
        if gemini_mode in ["description", "caption", "modality_separate_caption"]:
            vname = response["request"]["contents"][0]["parts"][1]["fileData"]["fileUri"]
            # print(vname)
        tot+=1
        try:
            pred = response["response"]["candidates"][0]["content"]["parts"][0]["text"]
            if gemini_mode in ["description", "qa", "modality_separate_caption", "modality_qa", "predict_emotion_from_modality_captions", "modality_qa_all"]:
                pred = pred.replace("```json", "").replace("```", "").strip()
                if gemini_mode == "description":
                    pred = trim_junk(pred)
                pred = json.loads(pred)
            if gemini_mode == "description":
                pred_dict[vname] = pred["reason"]
            elif gemini_mode == "caption":
                pred_dict[vname] = pred
            elif gemini_mode in ["qa", "modality_qa", "modality_qa_all"]:
                # print(pred)
                vname = pred["video_id"]
                questions = pred["questions"]
                pred_dict[vname].extend(questions)
            elif gemini_mode == "modality_separate_caption":
                if vname.endswith(".mp4"):
                    pred_dict_video[vname.replace("_24fps_no_audio", "_24fps")] = pred
                elif vname.endswith(".wav"):
                    pred_dict_audio[vname.replace("_16khz", "_24fps").replace(".wav", ".mp4")] = pred
                else:
                    raise ValueError(f"Unknown file type for {vname}. Expected .mp4 or .wav.")
            elif gemini_mode == "predict_emotion_from_modality_captions":
                prompt = response["request"]["contents"][0]["parts"][0]["text"]
                vname = pred["video_id"]
                emo_pred = pred["emotion"]
                if "audio" in prompt[:50].lower():
                    pred_dict_audio[vname] = emo_pred
                elif "video" in prompt[:50].lower():
                    pred_dict_video[vname] = emo_pred
                else:
                    raise ValueError(f"Unknown modality for {vname}. Expected audio or video.")
        except:
            incorr+=1
            # print(response)
    print(f"Reading jsonl from {pred_jsonl} resulted in {incorr}/{tot} failures...")
    if gemini_mode in ["description", "caption", "qa", "modality_qa", "modality_qa_all"]:
        return pred_dict
    elif gemini_mode in ["modality_separate_caption", "predict_emotion_from_modality_captions"]:
        return pred_dict_video, pred_dict_audio

def get_res_from_jsonl_gpt(pred_jsonl, annotation_mode = "modality_qa_all"):
    if annotation_mode not in [ "modality_qa_all", "modality_separate_caption", "predict_emotion_from_video_captions", "hallucination_qa", "av_long_caption_rewrite", "hallucination_qa_extras"]:
        raise ValueError(f"Unknown annotation mode: {annotation_mode}.")
    response_json = read_jsonl(pred_jsonl)

    pred_dict = {}
    if annotation_mode in ["modality_qa_all", "hallucination_qa", "hallucination_qa_extras"]:
        pred_dict = defaultdict(list)
    incorr, tot = 0, 0
    for response in response_json:
            # print(vname)
        tot+=1
        try:
            pred = response["response"]["body"]["choices"][0]["message"]["content"]
            if annotation_mode in ["modality_qa_all", "modality_separate_caption", "predict_emotion_from_video_captions", "hallucination_qa", "hallucination_qa_extras"]:
                pred = pred.replace("```json", "").replace("```", "").strip()
                pred = json.loads(pred)
            if annotation_mode in ["modality_qa_all", "hallucination_qa", "hallucination_qa_extras"]:
                # print(pred)
                vname = pred["video_id"]
                questions = pred["questions"]
                pred_dict[vname].extend(questions)
            elif annotation_mode == "modality_separate_caption":
                vname = response["custom_id"].split("|")[0]
                pred_dict[vname] = pred
            elif annotation_mode == "predict_emotion_from_video_captions":
                vname = response["custom_id"].split("|")[0]
                pred_dict[vname] = pred["predicted_emotion"]
            elif annotation_mode == "av_long_caption_rewrite":
                vname = response["custom_id"].split("|")[0]
                if "ERROR" in pred.upper():
                    raise ValueError(f"Error in response for {vname}: {pred}")
                pred_dict[vname] = pred
        except:
            incorr+=1
            # print(response)
    print(f"Reading jsonl from {pred_jsonl} resulted in {incorr}/{tot} failures...")
    return pred_dict


def get_wav_duration(path: str) -> float:
    """Get duration of a .wav file in seconds."""
    try:
        with wave.open(path, 'rb') as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            return frames / float(rate)
    except Exception as e:
        print(f"⚠️ Failed to read {path}: {e}")
        return 0.0

def total_wav_duration(parent_path: str, threads: int = 32) -> float:
    """Compute total duration (in seconds) of all .wav files under parent_path."""
    wav_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(parent_path)
        for file in files if file.lower().endswith('.wav')
    ]

    total = 0.0
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [pool.submit(get_wav_duration, f) for f in wav_files]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Calculating duration"):
            total += f.result()

    print(f"✅ Total duration: {total:.2f} seconds over {len(wav_files)} files")
    return len(wav_files), total


def get_video_duration(path: str) -> float:
    """Get duration of a video file in seconds using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"⚠️ Failed to read {path}: {e}")
        return 0.0

def total_video_duration(parent_path: str, threads: int = 32) -> float:
    """Compute total duration (in seconds) of all video files under parent_path."""
    video_exts = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.mpeg', '.mpg')
    video_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(parent_path)
        for file in files if file.lower().endswith(video_exts)
    ]

    total = 0.0
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [pool.submit(get_video_duration, f) for f in video_files]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Calculating video duration"):
            total += f.result()

    print(f"✅ Total video duration: {total:.2f} seconds over {len(video_files)} files")
    return len(video_files), total


def split_gpt_requests(original_jsonl_file, num_requests_per_file=12012):
    with open(original_jsonl_file, 'r') as f:
        lines = f.readlines()
    
    num_files = (len(lines) + num_requests_per_file - 1) // num_requests_per_file  # Ceiling division
    print(f"Splitting {len(lines)} requests into {num_files} files with {num_requests_per_file} requests each.")
    split_files = []
    for i in range(num_files):
        start_idx = i * num_requests_per_file
        end_idx = min((i + 1) * num_requests_per_file, len(lines))
        split_file_path = f"{original_jsonl_file.split('.')[0]}_part_{i+1}.jsonl"
        with open(split_file_path, 'w') as split_file:
            split_file.writelines(lines[start_idx:end_idx])
        print(f"Created {split_file_path} with {end_idx - start_idx} requests.")
        split_files.append(split_file_path)
    return split_files

def rstrip_choice(choice):
    choice = choice.strip()
    if choice.startswith("(A)") or choice.startswith("(B)") or choice.startswith("(C)") or choice.startswith("(D)"):
        choice = choice[3:].strip()
    elif choice.startswith("A:") or choice.startswith("B:") or choice.startswith("C:") or choice.startswith("D:"):
        choice = choice[2:].strip()
    elif choice.startswith("A.") or choice.startswith("B.") or choice.startswith("C.") or choice.startswith("D."):
        choice = choice[2:].strip()
    return choice

def strip_choices_with_letters(choices):
    stripped_choices = []
    for choice in choices:
        new_choice = rstrip_choice(choice)
        stripped_choices.append(new_choice)
    return stripped_choices

def shuffle_choices(choices, correct_choice):
    correct_answer = rstrip_choice(choices[ord(correct_choice.upper()) - ord('A')])
    new_choices = strip_choices_with_letters(choices)
    random.shuffle(new_choices)
    correct_index = new_choices.index(correct_answer)
    new_correct_choice = chr(correct_index + ord('A'))
    new_choices = [f"({chr(i + ord('A'))}) {choice}" for i, choice in enumerate(new_choices)]
    return new_choices, new_correct_choice