"""
Single-video inference script for AVERE.

Usage:
    python infer.py --video_path <path> --prompt "Describe the emotion."
    python infer.py --video_path <path> --prompt "..." --no_audio
"""

import os
import sys
import argparse
import subprocess
import tempfile
import json

import torch

from avere.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_AUDIO_TOKEN, AUDIO_TOKEN_INDEX
from avere.conversation import conv_templates, SeparatorStyle
from avere.model.builder import load_pretrained_model
from avere.utils import disable_torch_init
from avere.mm_utils import (
    tokenizer_image_token,
    tokenizer_audio_token,
    tokenizer_audio_and_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _probe_streams(video_path: str) -> dict:
    """Return ffprobe stream info for the given file."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on '{video_path}':\n{result.stderr}")
    return json.loads(result.stdout)


def _has_audio_stream(video_path: str) -> bool:
    """Return True if the video file contains at least one audio stream."""
    info = _probe_streams(video_path)
    return any(s["codec_type"] == "audio" for s in info.get("streams", []))


def extract_audio_16k(video_path: str, out_wav: str) -> None:
    """Extract audio from video and resample to 16 kHz mono WAV."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",                  # no video
        "-ac", "1",             # mono
        "-ar", "16000",         # 16 kHz
        "-f", "wav",
        out_wav,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed:\n{result.stderr}")


def resample_video_24fps(video_path: str, out_video: str) -> None:
    """Re-encode video to 24 fps (copy audio stream as-is)."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", "fps=24",        # resample to 24 fps
        "-c:v", "libx264",
        "-preset", "fast",
        "-an",                  # drop audio — we handle it separately
        out_video,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg video resampling failed:\n{result.stderr}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Single-video inference with AVERE.")
    parser.add_argument("--model_path", required=False, default="./checkpoint/AVERE-7B",
                        help="Path to the AVERE model checkpoint.")
    parser.add_argument("--model_base", default=None,
                        help="Optional base model path (for LoRA / delta weights).")
    parser.add_argument("--video_path", required=True,
                        help="Path to the input video file.")
    parser.add_argument("--prompt", required=True,
                        help="Text prompt for the model.")
    parser.add_argument("--no_audio", action="store_true",
                        help="Disable audio modality (default: audio is enabled).")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum number of new tokens to generate (default: 512).")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature (default: 0.1).")
    parser.add_argument("--load_8bit", action="store_true")
    parser.add_argument("--load_4bit", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    with_audio = not args.no_audio

    # ---- validate input -----------------------------------------------------
    if not os.path.isfile(args.video_path):
        raise FileNotFoundError(f"Video not found: '{args.video_path}'")

    if with_audio and not _has_audio_stream(args.video_path):
        raise ValueError(
            f"Audio modality is enabled (--no_audio not set) but the input video "
            f"'{args.video_path}' contains no audio stream. "
            f"Pass --no_audio to run video-only inference."
        )

    # ---- load model ---------------------------------------------------------
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, _ = load_pretrained_model(
        args.model_path, args.model_base, model_name,
        args.load_8bit, args.load_4bit,
        device="cuda", cache_dir="cache_dir",
    )
    model.eval()
    speech_processor = processor["speech"]
    video_processor  = processor["video"]

    # ---- preprocess into temp files -----------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        resampled_video = os.path.join(tmpdir, "video_24fps.mp4")
        resample_video_24fps(args.video_path, resampled_video)

        if with_audio:
            resampled_audio = os.path.join(tmpdir, "audio_16k.wav")
            extract_audio_16k(args.video_path, resampled_audio)

        # ---- build tensors --------------------------------------------------
        video_tensor = video_processor([resampled_video], return_tensors="pt")["pixel_values"]
        video_tensor = [v.to(model.device, dtype=torch.float16) for v in video_tensor]

        if with_audio:
            speech_tensor = speech_processor([resampled_audio], return_tensors="pt")["spectrogram"]
            speech_tensor = [s.to(model.device, dtype=torch.float16) for s in speech_tensor]
        else:
            speech_tensor = None

        # ---- build prompt ---------------------------------------------------
        num_frames = model.get_video_tower().config.num_frames
        inp = DEFAULT_IMAGE_TOKEN * num_frames + "\n"
        if with_audio:
            inp = DEFAULT_AUDIO_TOKEN + "\n" + inp
        inp += args.prompt

        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # ---- tokenize -------------------------------------------------------
        if with_audio:
            input_ids = tokenizer_audio_and_image_token(
                prompt, tokenizer, return_tensors="pt"
            ).unsqueeze(0).cuda()
        else:
            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

        # ---- generate -------------------------------------------------------
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=video_tensor,
                audios=speech_tensor,
                do_sample=True,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                use_cache=False,
                stopping_criteria=[stopping_criteria],
            )

    output = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
    # strip trailing stop token if present
    if output.endswith(stop_str):
        output = output[: -len(stop_str)].strip()

    print(output)
    return output


if __name__ == "__main__":
    main()
