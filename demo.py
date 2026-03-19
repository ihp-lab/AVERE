"""
Gradio demo for AVERE — single-turn video + audio inference.

Usage:
    python demo.py --model_path <path> [--model_base <path>] [--load_4bit] [--load_8bit] [--port 7860]
"""

import os
import sys
import argparse
import subprocess
import tempfile
import json

import torch
import gradio as gr

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
# Example prompts sampled from EmoReAlM task categories
# ---------------------------------------------------------------------------
EXAMPLE_PROMPTS = [
    # modality_agreement
    "Do the audio and video modalities effectively convey the same emotion expressed by the person in the video?",
    # reasoning_basic_audio
    "What vocal cue indicates the speaker's emotional state in the video?",
    # reasoning_basic_video
    "What does the person's facial expression and body language convey in this video?",
    # reasoning_stress_audio
    "Does the tone of voice enhance or contradict the emotion displayed by the person in the video?",
    # reasoning_stress_video
    "Do the person's facial features reflect a positive, negative, or neutral emotional state?",
]


# ---------------------------------------------------------------------------
# ffprobe / ffmpeg helpers (same as infer.py)
# ---------------------------------------------------------------------------

def _has_audio_stream(video_path: str) -> bool:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    info = json.loads(result.stdout)
    return any(s["codec_type"] == "audio" for s in info.get("streams", []))


def extract_audio_16k(video_path: str, out_wav: str) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000", "-f", "wav",
        out_wav,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr}")


def resample_video_24fps(video_path: str, out_video: str) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", "fps=24",
        "-c:v", "libx264", "-preset", "fast",
        "-an",
        out_video,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg video resampling failed: {result.stderr}")


# ---------------------------------------------------------------------------
# Model loader (called once at startup)
# ---------------------------------------------------------------------------

def load_model(model_path, model_base, load_8bit, load_4bit):
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, _ = load_pretrained_model(
        model_path, model_base, model_name,
        load_8bit, load_4bit,
        device="cuda", cache_dir="cache_dir",
    )
    model.eval()
    return tokenizer, model, processor["speech"], processor["video"]


# ---------------------------------------------------------------------------
# Inference function — wired to the Gradio interface
# ---------------------------------------------------------------------------

def run_inference(video_path, prompt, with_audio, max_new_tokens, temperature,
                  tokenizer, model, speech_processor, video_processor):
    """Core inference callable used by Gradio."""

    if video_path is None:
        return "Please upload a video file."
    if not prompt or not prompt.strip():
        return "Please enter a prompt."

    # Audio validation
    if with_audio:
        try:
            has_audio = _has_audio_stream(video_path)
        except RuntimeError as e:
            return f"Error probing video: {e}"
        if not has_audio:
            return (
                "Error: 'Use audio' is enabled but the uploaded video has no audio stream. "
                "Please upload a video with audio, or uncheck 'Use audio'."
            )

    with tempfile.TemporaryDirectory() as tmpdir:
        resampled_video = os.path.join(tmpdir, "video_24fps.mp4")
        try:
            resample_video_24fps(video_path, resampled_video)
        except RuntimeError as e:
            return f"Video preprocessing failed: {e}"

        if with_audio:
            resampled_audio = os.path.join(tmpdir, "audio_16k.wav")
            try:
                extract_audio_16k(video_path, resampled_audio)
            except RuntimeError as e:
                return f"Audio preprocessing failed: {e}"

        # Build tensors
        video_tensor = video_processor([resampled_video], return_tensors="pt")["pixel_values"]
        video_tensor = [v.to(model.device, dtype=torch.float16) for v in video_tensor]

        if with_audio:
            speech_tensor = speech_processor([resampled_audio], return_tensors="pt")["spectrogram"]
            speech_tensor = [s.to(model.device, dtype=torch.float16) for s in speech_tensor]
        else:
            speech_tensor = None

        # Build prompt string
        num_frames = model.get_video_tower().config.num_frames
        inp = DEFAULT_IMAGE_TOKEN * num_frames + "\n"
        if with_audio:
            inp = DEFAULT_AUDIO_TOKEN + "\n" + inp
        inp += prompt.strip()

        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        # Tokenize
        if with_audio:
            input_ids = tokenizer_audio_and_image_token(
                full_prompt, tokenizer, return_tensors="pt"
            ).unsqueeze(0).cuda()
        else:
            input_ids = tokenizer_image_token(
                full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

        # Generate
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=video_tensor,
                audios=speech_tensor,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=int(max_new_tokens),
                use_cache=False,
                stopping_criteria=[stopping_criteria],
            )

    output = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
    if output.endswith(stop_str):
        output = output[: -len(stop_str)].strip()
    return output


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_demo(tokenizer, model, speech_processor, video_processor):

    def predict(video_path, prompt, with_audio, max_new_tokens, temperature):
        return run_inference(
            video_path, prompt, with_audio, max_new_tokens, temperature,
            tokenizer, model, speech_processor, video_processor,
        )

    with gr.Blocks(title="AVERE — Audio-Visual Emotion Reasoning") as demo:
        gr.Markdown(
            "# AVERE — Audio-Visual Emotion Reasoning\n"
            "Upload a video, enter a prompt, and click **Run** to get a response."
        )

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="Upload Video")
                with_audio_checkbox = gr.Checkbox(
                    label="Use audio",
                    value=True,
                    info="Uncheck to run video-only inference. "
                         "An error is raised if checked but the video has no audio.",
                )

            with gr.Column(scale=1):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your question here…",
                    lines=3,
                )
                with gr.Accordion("Generation settings", open=False):
                    max_new_tokens_slider = gr.Slider(
                        minimum=16, maximum=1024, value=512, step=16,
                        label="Max new tokens",
                    )
                    temperature_slider = gr.Slider(
                        minimum=0.01, maximum=1.0, value=0.1, step=0.01,
                        label="Temperature",
                    )
                run_btn = gr.Button("Run", variant="primary")

        output_box = gr.Textbox(label="Model response", lines=5, interactive=False)

        run_btn.click(
            fn=predict,
            inputs=[video_input, prompt_input, with_audio_checkbox,
                    max_new_tokens_slider, temperature_slider],
            outputs=output_box,
        )

        gr.Markdown("### Example prompts")
        gr.Examples(
            examples=[[p] for p in EXAMPLE_PROMPTS],
            inputs=[prompt_input],
            label="Click an example to load it into the prompt box",
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="AVERE Gradio demo.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_base", default=None)
    parser.add_argument("--load_8bit", action="store_true")
    parser.add_argument("--load_4bit", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Loading model…")
    tokenizer, model, speech_processor, video_processor = load_model(
        args.model_path, args.model_base, args.load_8bit, args.load_4bit
    )
    print("Model loaded. Starting demo…")
    demo = build_demo(tokenizer, model, speech_processor, video_processor)
    demo.launch(server_port=args.port, share=args.share)
