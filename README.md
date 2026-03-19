<div align="center">
  <img src="./assets/readme_assets/avere_logo_cropped.png" width="380">

  <h1>AVERE: Improving Audiovisual Emotion Reasoning with Preference Optimization</h1>
  <h3><em>ICLR 2026 &nbsp;·&nbsp; Rio de Janeiro, Brazil</em></h3>

  <p>
    <a href="https://arxiv.org/abs/2602.07054">
      <img src="https://img.shields.io/badge/arXiv-2602.07054-b31b1b.svg?logo=arxiv" alt="arXiv">
    </a>
    <a href="https://openreview.net/forum?id=td682AAuPr">
      <img src="https://img.shields.io/badge/OpenReview-td682AAuPr-b31b1b?logo=wechat&logoColor=f5f5f5" alt="OpenReview">
    </a>
    <a href="https://github.com/ihp-lab/AVERE">
      <img src="https://img.shields.io/badge/GitHub-AVERE-black?logo=github" alt="GitHub">
    </a>
    <a href="https://avere-iclr.github.io/">
      <img src="https://img.shields.io/badge/Website-avere--iclr.github.io-6f42c1?logo=googlechrome&logoColor=white" alt="Website">
    </a>
    <a href="https://huggingface.co/chaubeyG/AVERE-7B">
      <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Weights-AVERE--7B-orange" alt="Model Weights">
    </a>
    <a href="https://huggingface.co/datasets/chaubeyG/EmoReAlM">
      <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Benchmark-EmoReAlM-orange" alt="Dataset">
    </a>
    <a href="LICENSE.rst">
      <img src="https://img.shields.io/badge/License-USC%20Research-green" alt="License">
    </a>
    <a href="https://www.python.org/">
      <img src="https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white" alt="Python">
    </a>
  </p>
</div>

---

## Table of Contents

- [Abstract](#-abstract)
- [News](#-news)
- [Results](#-results)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Inference](#-inference)
  - [Single-Video CLI](#single-video-cli)
  - [Interactive Demo](#interactive-demo)
- [Evaluation](#-evaluation)
- [Repository Progress](#-repository-progress)
- [License](#-license)
- [Credits](#-credits)
- [Citation](#-citation)

---

## 📄 Abstract

Emotion understanding is essential for building socially intelligent agents. Although recent multimodal large language models (MLLMs) have shown strong performance on this task, two key challenges remain: **(i)** spurious associations between emotions and irrelevant audiovisual cues (reasoning errors) and **(ii)** hallucination of audiovisual cues (perception errors) driven by text priors in the language model backbone.

To quantify and understand these issues, we introduce **EmoReAlM**, a benchmark designed to evaluate MLLMs for cue–emotion associations, hallucinations, and modality agreement. We then propose **AVEm-DPO**, a preference optimization technique that aligns model responses with both audiovisual inputs and emotion-centric queries. Specifically, we construct preferences over (i) responses exhibiting spurious associations or hallucinations and (ii) audiovisual input pairs guided by textual prompts. We also include a regularization term that penalizes reliance on text priors, thereby mitigating modality-specific cue hallucinations.

Experimental results on **DFEW**, **RAVDESS**, and **EMER** demonstrate that our method significantly improves the performance of reference baseline models (**6–19% relative improvement**) in zero-shot settings.

---

## 📣 News

| Date | Update |
|------|--------|
| **Mar. 2026** | Official codebase is now public. Initial release includes evaluation scripts for EmoReAlM and other emotion benchmarks. Training code coming soon. |
| **Mar. 2026** | Model weights for AVERE-7B are live on 🤗 HuggingFace → [chaubeyG/AVERE-7B](https://huggingface.co/chaubeyG/AVERE-7B). |
| **Feb. 2026** | Pleased to announce our CVPR 2026 paper on using DPO to mitigate cross-modal hallucinations in omni-LLMs → [MoD-DPO](https://mod-dpo.github.io/). |
| **Jan. 2026** | EmoReAlM benchmark released on 🤗 HuggingFace → [chaubeyG/EmoReAlM](https://huggingface.co/datasets/chaubeyG/EmoReAlM). |
| **Jan. 2026** | AVERE accepted to **ICLR 2026**. See you in Rio de Janeiro! |

---

## 🏆 Results

Detailed results and the EmoReAlM leaderboard are available on the project website: <a href="https://avere-iclr.github.io/"><strong>avere-iclr.github.io</strong></a>

> To submit your model to the EmoReAlM leaderboard, please contact the first author at [achaubey@usc.edu](mailto:achaubey@usc.edu).

---

## 📦 Repository Structure

```
AVERE/
├── avere/                        # Core package (model, training, serving, eval utilities)
│   ├── model/                    #   Model architecture (LLaVA-based multimodal LLM)
│   ├── train/                    #   Training entry points
├── backbones/                    # Pre-trained backbone models (see Inference §1)
│   ├── bert-base-uncased/
│   ├── LanguageBind_Image/
│   ├── LanguageBind_Video_merge/
│   └── whisper-large-v3/
├── checkpoint/                   # AVERE model weights (see Inference §2)
├── data_preprocess/              # Data preprocessing scripts
├── evaluate/                     # Evaluation pipeline & benchmark metrics
├── scripts/                      # Training & evaluation shell scripts
├── infer.py                      # Single-video inference script
└── demo.py                       # Interactive Gradio demo
```

---

## 🔧 Installation

**1. Clone the repository**

```bash
git clone https://github.com/ihp-lab/AVERE.git
cd AVERE
```

**2. Create a Conda environment**

```bash
conda create -n avere python=3.10 -y
conda activate avere
```

**3. Install PyTorch**

```bash
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> **Note:** The command above targets CUDA 12.1, which has also been tested on CUDA 12.2. Adjust the `--index-url` if your machine uses a different CUDA version.

**4. Install AVERE**

```bash
pip install -e .              # core package
pip install -e ".[train]"     # additionally install training dependencies
```

**5. Install additional dependencies**

```bash
pip install flash-attn --no-build-isolation   # recommended but optional
pip install decord soundfile opencv-python \
    git+https://github.com/facebookresearch/pytorchvideo.git@28fe037d212663c6a24f373b94cc5d478c8c1a1d
```

---

## 🎯 Inference

### Preparation

**Step 1 — Download backbone models** into `backbones/`:

| Model | Source |
|-------|--------|
| `LanguageBind_Image` | [HuggingFace](https://huggingface.co/LanguageBind/LanguageBind_Image) |
| `LanguageBind_Video_merge` | [HuggingFace](https://huggingface.co/LanguageBind/LanguageBind_Video_merge) |
| `bert-base-uncased` | [HuggingFace](https://huggingface.co/google-bert/bert-base-uncased) |
| `whisper-large-v3` | [HuggingFace](https://huggingface.co/openai/whisper-large-v3) |

**Step 2 — Download AVERE weights** from [chaubeyG/AVERE-7B](https://huggingface.co/chaubeyG/AVERE-7B) into `checkpoint/` so the path becomes `./checkpoint/AVERE-7B`.

---

### Single-Video CLI

`infer.py` accepts a single video file and a text prompt. The video is automatically resampled to **24 fps** and audio to **16 kHz** before inference.

```bash
# Video + audio (default)
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --model_path "checkpoint/AVERE-7B" \
    --video_path "/path/to/input.mp4" \
    --prompt "Describe the emotion of the person in the video in detail."

# Video only
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --model_path "checkpoint/AVERE-7B" \
    --video_path "/path/to/input.mp4" \
    --prompt "What emotion does the person's facial expression convey?" \
    --no_audio
```

<details>
<summary><strong>All CLI options</strong></summary>

| Flag | Default | Description |
|------|---------|-------------|
| `--model_path` | *(required)* | Path to the AVERE model checkpoint |
| `--model_base` | `None` | Base model path for LoRA / delta weights |
| `--video_path` | *(required)* | Path to the input video file |
| `--prompt` | *(required)* | Text prompt for the model |
| `--no_audio` | `False` | Disable audio modality (video-only inference) |
| `--max_new_tokens` | `512` | Maximum tokens to generate |
| `--temperature` | `0.1` | Sampling temperature |
| `--load_4bit` | `False` | Load model in 4-bit quantization |
| `--load_8bit` | `False` | Load model in 8-bit quantization |

> **Important:** If `--no_audio` is not set and the video has no audio stream, the script raises an error before loading the model.

</details>

---

### Interactive Demo

`demo.py` launches a Gradio web interface for single-turn video + prompt inference.

```bash
# Basic launch
CUDA_VISIBLE_DEVICES=0 python demo.py \
    --model_path "checkpoint/AVERE-7B" \
    --port 7860

# With 4-bit quantization and a public share link
CUDA_VISIBLE_DEVICES=0 python demo.py \
    --model_path "checkpoint/AVERE-7B" \
    --load_4bit --share
```

The interface includes pre-loaded example prompts drawn from all five EmoReAlM task categories: `modality_agreement`, `reasoning_basic_audio`, `reasoning_basic_video`, `reasoning_stress_audio`, and `reasoning_stress_video`.

---

## 📊 Evaluation

**Step 1 & 2** — Complete the [Inference preparation](#preparation) steps above.

**Step 3** — Set the parent data directory by updating `VIDEO_EVAL_DATA_PAR` in `evaluate/eval_constants.py`.

**Step 4 — Download datasets:**

| Dataset | Source | Notes |
|---------|--------|-------|
| **EmoReAlM** | [chaubeyG/EmoReAlM](https://huggingface.co/datasets/chaubeyG/EmoReAlM) | Place inside the data directory |
| **DFEW** | [Official website](https://dfew-dataset.github.io/) | Process videos to 24 fps, audio to 16 kHz |
| **RAVDESS** | [Zenodo](https://zenodo.org/records/1188976) | Optional — only needed for RAVDESS evaluation |

<details>
<summary><strong>Expected data directory structure</strong></summary>

```
<data_dir>/
│
├── EmoReAlM/
│   └── emorealm_v1.json
│
├── dfew/
│   ├── dfew_annotation/
│   │   └── test(single-labeled)/
│   │       └── set_1.csv
│   ├── dfew_original_clips_24fps/
│   │   ├── part_1/
│   │   │   └── 1.mp4                    # 24 fps
│   │   └── ... (part_2 … part_11)
│   └── dfew_original_clips_16khz/
│       ├── part_1/
│       │   └── 1.wav                    # 16 kHz mono PCM
│       └── ... (part_2 … part_11)
│
└── RAVDESS/                             # optional
    ├── ravdess_videos_24fps/
    │   ├── Actor_01/
    │   │   └── 01-01-01-01-01-01-01.mp4
    │   └── ... (Actor_02 … Actor_24)
    └── ravdess_videos_16khz/
        ├── Actor_01/
        │   └── 01-01-01-01-01-01-01.wav
        └── ... (Actor_02 … Actor_24)
```

</details>

**Step 5 — Run evaluation:**

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 python evaluate/main.py \
    --model_path "checkpoint/AVERE-7B" \
    --task "emotion_qa-emorealm" \
    --batch_size 1

# Multi-GPU (recommended for DFEW / RAVDESS. WILL NOT WORK FOR EMOREALM.)
bash scripts/eval.sh \
    --gpus 2,4,6 \
    --task emotion-dfew-audio \
    --batch 8
```

Available tasks: `emotion_qa-emorealm` · `emotion-dfew-audio` · `emotion-ravdess-video-audio`

> **Tip:** Use `--batch_size 1` for EmoReAlM. For DFEW and RAVDESS, larger batch sizes (e.g. 8 on an H100) significantly speed up evaluation.

Results and per-sample predictions are saved to `eval_temp/{task_name}/AVERE-7B/`. Scores should closely match the *"Our Base + AVEm-DPO"* row reported in the paper.

---

## ✅ Repository Progress

| Component | Status |
|-----------|--------|
| Model Weights | ✅ Released |
| Benchmark (EmoReAlM) | ✅ Released |
| Inference Code | ✅ Released |
| Evaluation Code | ✅ Released |
| Training Script | ✅ Released |
| Training Data & Instructions | 🔜 Coming soon |
| Full Training Code & Instructions | 🔜 Coming soon |

---

## ⚖️ License

This codebase is distributed under the **USC Research License**. See [LICENSE.rst](LICENSE.rst) for details.

Portions of this codebase are derived from [Vista-DPO](https://github.com/HaroldChen19/VistaDPO) and [VideoLLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA); those portions inherit their respective licenses.

---

## 🙌 Credits

AVERE builds upon the following excellent open-source works:

- [VistaDPO](https://github.com/HaroldChen19/VistaDPO)
- [VideoLLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA)

We gratefully acknowledge their contributions to the open-source community.

---

## 🔗 Other Works

<table>
  <tr>
    <td align="center" width="50%">
      <a href="https://face-llava.github.io/">
        <strong>Face-LLaVA</strong>
      </a>
      <br>
      <em>WACV 2026 🌵</em>
      <br><br>
      A multimodal large language model for fine-grained facial understanding, leveraging face-specific visual encoders for improved expression and attribute reasoning.
      <br><br>
      <a href="https://face-llava.github.io/">
        <img src="https://img.shields.io/badge/Website-face--llava.github.io-6f42c1?logo=googlechrome&logoColor=white" alt="Face-LLaVA Website">
      </a>
    </td>
    <td align="center" width="50%">
      <a href="https://mod-dpo.github.io/">
        <strong>MoD-DPO</strong>
      </a>
      <br>
      <em>CVPR 2026 ⛰️</em>
      <br><br>
      A preference optimization approach to mitigate cross-modal hallucinations in omni-LLMs, enabling more faithful alignment between visual, audio, and textual modalities.
      <br><br>
      <a href="https://mod-dpo.github.io/">
        <img src="https://img.shields.io/badge/Website-mod--dpo.github.io-6f42c1?logo=googlechrome&logoColor=white" alt="MoD-DPO Website">
      </a>
    </td>
  </tr>
</table>

---

## 🪶 Citation

If you find AVERE or EmoReAlM useful in your research, please cite:

```bibtex
@inproceedings{chaubey2026avere,
  title     = {{AVERE}: Improving Audiovisual Emotion Reasoning with Preference Optimization},
  author    = {Ashutosh Chaubey and Jiacheng Pang and Maksim Siniukov and Mohammad Soleymani},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=td682AAuPr}
}
```
