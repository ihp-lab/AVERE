# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import sys
sys.path.append("/wekafs/ict/achaubey/emotion_reasoning/audio_exp/Video-LLaVA")

import os
import copy
import random
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List, Tuple, Union, Any
from accelerate.utils import DistributedType
import numpy as np
from datetime import datetime

import torch

import transformers

from avere.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN, DEFAULT_VIDEO_TOKEN, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN, MAX_IMAGE_LENGTH, \
    MAX_VIDEO_LENGTH, AUDIO_TOKEN_INDEX
from torch.utils.data import Dataset
from avere.train.llava_trainer import LLaVADPOTrainer

from avere import conversation as conversation_lib
from avere.model import *
from avere.mm_utils import tokenizer_image_token, tokenizer_audio_token, tokenizer_audio_and_image_token

from PIL import Image
from avere.utils import order_pick_k

from trl.trainer.utils import DPODataCollatorWithPadding

local_rank = None
# set max audio length (number of audio files that can be in a single prompt). Note that this is NOT AUDIO DURATION.
MAX_AUDIO_LENGTH=MAX_VIDEO_LENGTH

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")

    tune_speech_mlp_adapter: bool = field(default=False)
    speech_projector_type: Optional[str] = field(default='naive')
    pretrain_speech_mlp_adapter: Optional[str] = field(default=None)

    ### speech qformer params
    qformer_num_speech_query_tokens: Optional[int] = field(default=1)
    qformer_window_level: Optional[bool] = field(default=True)
    qformer_second_per_window: Optional[float] = field(default=0.33333)
    qformer_second_stride: Optional[float] = field(default=0.33333)

    ## video qformer params
    qformer_num_video_query_tokens: Optional[int] = field(default=16)
    video_qformer_window_level: Optional[bool] = field(default=True)
    video_qformer_tokens_per_window: Optional[int] = field(default=257)
    video_qformer_token_stride: Optional[int] = field(default=257)

    # ===================================================================
    image_tower: Optional[str] = field(default=None)
    video_tower: Optional[str] = field(default=None)
    speech_tower: Optional[str] = field(default=None)
    # ===================================================================

@dataclass
class DataArguments:
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_aspect_ratio: str = 'square'
    # ===================================================================
    data_path: Optional[List[str]] = field(default=None, metadata={"help": "Path to the training data."})
    image_folder: Optional[str] = field(default=None)
    video_folder: Optional[str] = field(default=None)
    audio_folder: Optional[str] = field(default=None)
    num_frames: int = 8
    # ===================================================================


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    freeze_speech_mlp_adapter:bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    speech_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    attn_implementation: str = field(default="flash_attention_2", metadata={"help": "Use transformers attention implementation."})

    # ================================================
    tokenizer_model_max_length: Optional[int] = None
    # ================================================

    fix_vit: bool = True
    dpo_alpha: float = field(default=1.0)
    beta: float = field(default=0.1)
    gamma: float = field(default=1.0)
    generate_during_eval: bool = field(default=False)
    tpd_gamma: float = field(default=0.1)
    use_tpd: bool = field(default=False)

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'speech_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    # if getattr(trainer.args, "tune_mm_mlp_adapter", False):
    #     # Only save Adapter
    #     keys_to_match = ['mm_projector']
    #     if getattr(trainer.args, "use_im_start_end", False):
    #         keys_to_match.extend(['embed_tokens', 'embed_in'])

    #     weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
    #     trainer.model.config.save_pretrained(output_dir)

    #     current_folder = output_dir.split('/')[-1]
    #     parent_folder = os.path.dirname(output_dir)
    #     if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
    #         if current_folder.startswith('checkpoint-'):
    #             mm_projector_folder = os.path.join(parent_folder, "mm_projector")
    #             os.makedirs(mm_projector_folder, exist_ok=True)
    #             torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
    #         else:
    #             torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
    #     return

    # if getattr(trainer.args, "tune_speech_mlp_adapter", False):
    #     # Only save Adapter
    #     keys_to_match = ['speech_projector']

    #     weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
    #     trainer.model.config.save_pretrained(output_dir)

    #     current_folder = output_dir.split('/')[-1]
    #     parent_folder = os.path.dirname(output_dir)
    #     if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
    #         if current_folder.startswith('checkpoint-'):
    #             speech_projector_folder = os.path.join(parent_folder, "speech_projector")
    #             os.makedirs(speech_projector_folder, exist_ok=True)
    #             torch.save(weight_to_save, os.path.join(speech_projector_folder, f'{current_folder}.bin'))
    #         else:
    #             torch.save(weight_to_save, os.path.join(output_dir, f'speech_projector.bin'))
    #     return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def save_my_lora_ckpt(output_dir, args, model):
    state_dict = get_peft_state_maybe_zero_3(
        model.named_parameters(), args.lora_bias
    )
    non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
        model.named_parameters()
    )
    if args.local_rank == 0 or args.local_rank == -1:
        model.config.save_pretrained(output_dir)
        model.save_pretrained(output_dir, state_dict=state_dict)
        torch.save(non_lora_state_dict, os.path.join(output_dir, 'non_lora_trainables.bin'))



def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources
    # print("======================= SOURCES ================================")
    # print(sources)
    # print("="*100)
    for source in sources:
        for sentence in source:

            # ======================================================================================================
            if sentence['value'].startswith(DEFAULT_IMAGE_TOKEN) or sentence['value'].startswith(DEFAULT_VIDEO_TOKEN):  # run with multi-im, multi-vid, multi-im & multi-vid
                # <video><video><image><image>\nxxxxxxxxxxxxx  # must <video> first
                # <image>\nxxxxxxxxxxxxx -> <image>\nxxxxxxxxxxxxx
                # <video>\nxxxxxxxxxxxxx -> <video>\nxxxxxxxxxxxxx

                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')

                IMAGE_TOKEN_NUM = sentence['value'].count(DEFAULT_IMAGE_TOKEN)
                if IMAGE_TOKEN_NUM > MAX_IMAGE_LENGTH:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN * IMAGE_TOKEN_NUM, DEFAULT_IMAGE_TOKEN * MAX_IMAGE_LENGTH).strip()
                VIDEO_TOKEN_NUM = sentence['value'].count(DEFAULT_VIDEO_TOKEN)
                if VIDEO_TOKEN_NUM > MAX_VIDEO_LENGTH:
                    raise ValueError(f"{sentence['value']}")
                    sentence['value'] = sentence['value'].replace(DEFAULT_VIDEO_TOKEN * VIDEO_TOKEN_NUM, DEFAULT_VIDEO_TOKEN * MAX_VIDEO_LENGTH).strip()

            # a <video> is treated as `num_frames * <image>`
            replace_token, vid_replace_token = DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_TOKEN * data_args.num_frames
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                vid_replace_token = DEFAULT_VID_START_TOKEN + vid_replace_token + DEFAULT_VID_END_TOKEN

            # <video><video><image><image>\nxxxxxxxxxxxxx -> `num_frames*<image>``num_frames*<image>`<image><image>\nxxxxxxxxxxxxx
            # <video>\nxxxxxxxxxxxxx -> `num_frames*<image>`\nxxxxxxxxxxxxx
            # print('before replace_token:', [sentence['value']])
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
            sentence['value'] = sentence['value'].replace(DEFAULT_VIDEO_TOKEN, vid_replace_token)
            # print('after replace_token:', [sentence['value']])
            # ======================================================================================================
    # print("======================= SOURCES AFTER preprocess_multimodal() ================================")
    # print(sources)
    # print("="*100)
    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, has_audio: bool = False,
                    max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)
    if has_audio:
        tokenizer.add_tokens(["<audio>"], special_tokens=True)

    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    audio_token_index = tokenizer.convert_tokens_to_ids("<audio>")
    # print("Type tokenizer -", type(tokenizer))
    # print("Tokenizer additional special tokens", tokenizer.additional_special_tokens_ids)
    im_start, im_end = tokenizer.additional_special_tokens_ids
    # unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
    unmask_tokens_idx =  [198, im_start, im_end]
    nl_tokens = tokenizer("\n").input_ids

    # Reset Qwen chat templates so that it won't include system message every time we apply
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # _system = tokenizer("system").input_ids + nl_tokens
    # _user = tokenizer("user").input_ids + nl_tokens
    # _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            # print("conv ==>", conv)
            encode_id = tokenizer.apply_chat_template(conv)
            # print("Encode_id ==>", encode_id)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
        

                    
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
            if encode_id == audio_token_index:
                input_id[idx] = AUDIO_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    has_audio:bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    # print("ROLES ==>", roles)


    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # print("CONVERSATIONS ==>", conversations)

    # Tokenize conversations

    if has_image and not has_audio:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    elif has_audio and not has_image:
        input_ids = torch.stack([tokenizer_audio_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    elif has_audio and has_image:
        input_ids = torch.stack([tokenizer_audio_and_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    # print("INPUT_IDS.shape ==>", input_ids.shape)
    # print("INPUT_IDS ==>", input_ids)
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image and not has_audio:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            elif has_audio and not has_image:
                round_len = len(tokenizer_audio_token(rou, tokenizer))
                instruction_len = len(tokenizer_audio_token(parts[0], tokenizer)) - 2
            elif has_audio and has_image:
                round_len = len(tokenizer_audio_and_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_audio_and_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    # print("FINAL INPUT_IDS", input_ids)
    # print("FINAL TARGETS", targets)
    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer_image_token(rou, tokenizer)) + len(tokenizer_image_token(conv.sep, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    has_audio: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        # print(f"INSIDE_V1_PREPROCESS - has_image: {has_image}, has_audio: {has_audio}")
        return preprocess_v1(sources, tokenizer, has_image=has_image, has_audio=has_audio)
    if conversation_lib.default_conversation.version == "qwen":
        # print("INSIDE_QWEN_PREPROCESS")
        return preprocess_qwen(sources, tokenizer, has_image=has_image, has_audio=has_audio)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)

    if has_audio:
        raise NotImplementedError("preprocess() is only implemented for version v1.0 in case there is audio...")
    
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def make_conv(prompt, answer):
    return [
        {
            "from": "human",
            "value": prompt,
        },
        {
            "from": "gpt",
            "value": answer,
        },
    ]

class AudioVisualDPODataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(AudioVisualDPODataset, self).__init__()

        # ================================================
        list_data_dict = []
        for data in data_path:
            data = json.load(open(data, "r"))
            for i in data:
                i['id'] = len(list_data_dict)
                list_data_dict.append(i)
        # ================================================

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def modality_lengths(self):
        length_list = []

        for sample in self.list_data_dict:
            has_video, has_image, has_audio = False, False, False
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            if "video" in sample:
                cur_len += 256 * 8  # assuming 8 frames of video, each frame is 256 tokens
                has_video = True
            if "image" in sample:
                cur_len += 256  # assuming 256 tokens for image
                has_image = True
            if "audio" in sample:
                cur_len += 88  # assuming 88 tokens for audio
                has_audio = True
            length_list.append((cur_len, f"v:{has_video},i:{has_image},a:{has_audio}"))
        
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # try:
        '''
        {
            "prompt": "",
            "chosen": "",
            "rejected": "",
            "rejected_2": "",
            'video': '',
            'audio': '',
            'video_l': '',
            'audio_l': '',
        }
        '''
        data_dict = self.list_data_dict[i]
        # ======================================================================================================

        if 'audio' in data_dict and 'video' in data_dict:
            # both audio and video are present here
            # processing them one by one
            if "processed_audio" not in data_dict:
                # first process audio
                audio_file = data_dict["audio"]
                audio_folder = self.data_args.video_folder  ## both are present in case of a video, hence video folder
                speech_processor = self.data_args.speech_processor
                audio_file = audio_file if isinstance(audio_file, list) else [audio_file]
                audio_file = order_pick_k(audio_file, MAX_AUDIO_LENGTH)
                audio = [os.path.join(audio_folder, file) for file in audio_file]
                audio = speech_processor(audio[0], return_tensors="pt")["spectrogram"][0]
                # print(f"Audio file shape after extracting spectrograms -->", [audio_file.shape for audio_file in audio]) ## (128,3000) for each audio file
            else:
                aud = np.load(os.path.join(self.data_args.video_folder, data_dict['processed_audio']))
                audio = torch.tensor(aud)

            ## process audio_l in the data_dict if it exists
            if 'audio_l' in data_dict:
                audio_file = data_dict['audio_l']
                audio_folder = self.data_args.video_folder
                audio_file = audio_file if isinstance(audio_file, list) else [audio_file]
                audio_file = order_pick_k(audio_file, MAX_AUDIO_LENGTH)
                audio_l = [os.path.join(audio_folder, file) for file in audio_file]
                audio_l = speech_processor(audio_l[0], return_tensors="pt")["spectrogram"][0]
            else:
                audio_l = torch.rand_like(audio)

            # now process video
            if "processed_video" not in data_dict:
                video_file = data_dict['video']
                video_folder = self.data_args.video_folder
                video_processor = self.data_args.video_processor
                video_file = video_file if isinstance(video_file, list) else [video_file]
                video_file = order_pick_k(video_file, MAX_VIDEO_LENGTH)
                video = [os.path.join(video_folder, file) for file in video_file]
                image = video_processor(video[0], return_tensors='pt')['pixel_values'][0]
                # print(f"Video file shape after extracting frames -->", [video_file.shape for video_file in image]) ## (3,8,224,224) for each video file
            else:
                print(f"Loading processed video from {os.path.join(self.data_args.video_folder, data_dict['processed_video'])}")
                video = np.load(os.path.join(self.data_args.video_folder, data_dict['processed_video']))
                image = torch.tensor(video)

            # process image_l in the data_dict if it exists
            if 'video_l' in data_dict:
                video_file = data_dict['video_l']
                video_folder = self.data_args.video_folder
                video_file = video_file if isinstance(video_file, list) else [video_file]
                video_file = order_pick_k(video_file, MAX_VIDEO_LENGTH)
                video_l = [os.path.join(video_folder, file) for file in video_file]
                image_l = video_processor(video_l[0], return_tensors='pt')['pixel_values'][0]
            else:
                image_l = torch.rand_like(image)


            ## validate prompt
            prompt = data_dict['prompt']
            if not prompt.startswith("<video>\n<audio>\n"):
                
                raise ValueError(f"Prompt should start with <video> and <audio> tags, but got: {prompt}")
        else:
            print("!!!!!!!!!!EITHER AUDIO OR VIDEO IS MISSING.......")
            raise ValueError(f"Either audio or video is missing in the data. Please check your data.")
            # sources = copy.deepcopy([e["conversations"] for e in sources])
            # data_dict = preprocess(sources, self.tokenizer, has_image=False)

        # ==========================================================================================================

        # if isinstance(i, int):
        #     data_dict = dict(input_ids=data_dict["input_ids"][0],
        #                         labels=data_dict["labels"][0])
        # image exist in the data
        if 'video' in data_dict:
            data_dict['image'] = image
            # data_dict['image_l'] = torch.rand_like(image)  # random length for each video frame
            data_dict['image_l'] = image_l
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            # crop_size = self.data_args.image_processor.crop_size
            # print("Image was none so we added a dummy image")
            # crop_size = {'height': 224, 'width': 224}  # dummy image
            # data_dict['image'] = [torch.zeros(3, 8, crop_size['height'], crop_size['width'])] # shape 8 because we only deal with videos in our case
            raise ValueError(f"Image is not present in the data, but the model is multimodal. Please check your data.")

        if 'audio' in data_dict:
            data_dict['audio'] = audio
            data_dict['audio_l'] = torch.rand_like(audio)  # random length for each audio file
        elif self.data_args.is_multimodal:
            # audio does not exist in the data, but the model is multimodal
            # crop_size = self.data_args.speech_processor.crop_size
            # print("Audio was none so we added a dummy audio")
            # crop_size = {'height': 128, 'width': 3000}
            # data_dict['audio'] = [torch.zeros(crop_size['height'], crop_size['width'])]
            raise ValueError(f"Audio is not present in the data, but the model is multimodal. Please check your data.")

        
        
        # print("The code is not reaching end = 10")
        return data_dict
        # except Exception as e:
        #     print(f'Error with {e}')
        #     return self.__getitem__(random.randint(0, self.__len__() - 1))


@dataclass
class DPODataCollator(DPODataCollatorWithPadding):
    data_args: Any = None  # new attribute

    def __init__(self, *args, data_args=None, **kwargs):
        super().__init__(*args, **kwargs)  # initialize parent with its args
        self.data_args = data_args        # store your extra argument

    def collate(self, batch):
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                # if "prompt" in k:
                #     to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                # else:
                to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith("_input_ids"):
                    padding_value = self.tokenizer.pad_token_id
                elif k.endswith("_labels"):
                    padding_value = self.label_pad_token_id
                else:
                    continue
                # elif k.endswith("_attention_mask"):
                #     padding_value = self.padding_value
                # else:
                #     raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = torch.nn.utils.rnn.pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                # for the prompt, flip back so padding is on left side
                # if "prompt" in k:
                #     padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]
        for k in ['chosen_input_ids', 'rejected_input_ids', 'rejected_ir_input_ids']:
        # for k in ['chosen_input_ids', 'rejected_input_ids']:
            attn_k = k.replace('input_ids', 'attention_mask')
            padded_batch[attn_k] = padded_batch[k].ne(self.tokenizer.pad_token_id)
        return padded_batch


    def tokenize_batch_element(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
        rejected_ir: str
    ) -> Dict:
        """Tokenize a single batch element.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
            in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        # import pdb; pdb.set_trace()
        """
        (Pdb) prompt
        '<video>\nWhat object did the person throw in the video?'
        (Pdb) chosen
        'The person in the video ate a delicious sandwich.'
        (Pdb) rejected
        'In the video, the person picked up a door instead of an object.'
        (Pdb) has_X
        'video'
        """
        batch = {} 
        
        chosen_sources = make_conv(prompt, chosen) # [{'from': 'human', 'value': '<video>\nIn the video, which object did the person take with them?'}, {'from': 'gpt', 'value': 'The person in the video took a paper notebook with them, likely for taking notes or carrying important documents.'}]
        rejected_sources = make_conv(prompt, rejected) # [{'from': 'human', 'value': '<video>\nWhich object did the person open in the video?'}, {'from': 'gpt', 'value': 'The video shows the person opening the bag, revealing its contents with a dramatic flourish.'}]
        rejected_ir_sources = make_conv(prompt, rejected_ir)

        chosen_sources = preprocess_multimodal([chosen_sources], self.data_args)
        rejected_sources = preprocess_multimodal([rejected_sources], self.data_args)
        rejected_ir_sources = preprocess_multimodal([rejected_ir_sources], self.data_args)
        # data_dict = preprocess(sources, self.tokenizer, has_image=True, has_audio=True)

        chosen_data_dict = preprocess(
            chosen_sources,
            self.tokenizer,
            has_image=True,
            has_audio=True
        )
        #chosen_data_dict['attention_mask'] = chosen_data_dict["input_ids"].ne(self.tokenizer.pad_token_id)

        rejected_data_dict = preprocess(
            rejected_sources,
            self.tokenizer,
            has_image=True,
            has_audio=True
        )

        rejected_ir_data_dict = preprocess(
            rejected_ir_sources,
            self.tokenizer,
            has_image=True,
            has_audio=True
        )

        #rejected_data_dict['attention_mask'] = rejected_data_dict["input_ids"].ne(self.tokenizer.pad_token_id)

        chosen_data_dict = {k: v[0] for k, v in chosen_data_dict.items()}
        rejected_data_dict = {k: v[0] for k, v in rejected_data_dict.items()}
        rejected_ir_data_dict = {k: v[0] for k, v in rejected_ir_data_dict.items()}

        for k, toks in {
            "chosen": chosen_data_dict,
            "rejected": rejected_data_dict,
            "rejected_ir": rejected_ir_data_dict,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}_{type_key}"] = tokens

        # import pdb; pdb.set_trace()

        return batch
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        tokenized_batch, tokenized_batch_l = [], []
        Vs, Vs_l, As, As_l = [], [], [], []
        for feature in features:
            prompt = feature["prompt"]
            chosen = feature["chosen"]
            rejected = feature["rejected"]
            rejected_ir = feature["rejected_2"]
            Vs.append(feature["image"])
            Vs_l.append(feature['image_l'])

            As.append(feature["audio"])
            As_l.append(feature['audio_l'])

            # if 'clip' in feature:
            #     Xclip.append(feature['clip'])
            #     Xclip_l.append(feature['clip_l'])
            #     Xframe.append(feature['frame'])
            #     Xframe_l.append(feature['frame_l'])
            # import pdb; pdb.set_trace()
            # (Pdb) feature.keys()
            # dict_keys(['id', 'video', 'prompt', 'chosen', 'rejected', 'rejected_2', 'video_l', 'has_X'])
             
            batch_element = self.tokenize_batch_element(prompt, chosen, rejected, rejected_ir)
            # batch_element = self.tokenize_batch_element(prompt, chosen, rejected)
            tokenized_batch.append(batch_element)

        # return collated batch
        padded_batch =  self.collate(tokenized_batch)
        padded_batch['images'] = Vs
        padded_batch['images_l'] = Vs_l
        padded_batch['audios'] = As
        padded_batch['audios_l'] = As_l
        # if Xclip != []:
        #     padded_batch['clips'] = [Xclip, keys]
        #     padded_batch['clips_l'] = [Xclip_l, keys]
        #     padded_batch['frames'] = [Xframe, keys]
        #     padded_batch['frames_l'] = [Xframe_l, keys]


        # import pdb; pdb.set_trace()
        # dict_keys(['chosen_input_ids', 'chosen_labels', 'rejected_input_ids', 'rejected_labels', 
        #           'chosen_attention_mask', 'rejected_attention_mask', 'images'])
        # dict_keys(['chosen_input_ids', 'chosen_labels', 'rejected_input_ids', 'rejected_labels', 'rejected_ir_input_ids', 'rejected_ir_labels', 
        #           'chosen_attention_mask', 'rejected_attention_mask', 'rejected_ir_attention_mask', 'images', 'images_l', 'clips', 'clips_l', 'frames', 'frames_l'])
        return padded_batch


def make_dpo_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = AudioVisualDPODataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    # data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return train_dataset


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector", "speech_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))
    # ==========================================================================
    if model_args.image_tower is not None or model_args.video_tower is not None or args.speech_tower is not None:
    # ==========================================================================
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMPTForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        elif "qwen" in model_args.model_name_or_path.lower():
            model = LlavaQwenForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **bnb_model_from_pretrained_args,
            )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                force_download=True,
                **bnb_model_from_pretrained_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    elif "qwen" in model_args.model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length, padding_side="right")
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # =============================================================================================================
    if model_args.image_tower is not None or model_args.video_tower is not None:
        # print(model_args)
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        if model_args.image_tower is not None:
            image_tower = model.get_image_tower()
            image_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

            data_args.image_processor = image_tower.image_processor
            data_args.is_multimodal = True
        if model_args.video_tower is not None:
            video_tower = model.get_video_tower()
            video_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

            data_args.video_processor = video_tower.video_processor
            data_args.is_multimodal = True
            data_args.num_frames = video_tower.config.num_frames
    # =============================================================================================================
    if model_args.speech_tower is not None:
        # print(model_args)
        model.get_model().initialize_audio_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        if model_args.speech_tower is not None:
            speech_tower = model.get_speech_tower()
            speech_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

            data_args.speech_processor = speech_tower.speech_processor
            data_args.is_multimodal = True
    # =============================================================================================================


    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side

    # =============================================================================================================
    tokenizer_model_max_length = training_args.tokenizer_model_max_length
    model.config.tokenizer_model_max_length = tokenizer.model_max_length if tokenizer_model_max_length is None else tokenizer_model_max_length
    # =============================================================================================================
    
    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
    if training_args.freeze_mm_mlp_adapter:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False

    model.config.tune_speech_mlp_adapter = training_args.tune_speech_mlp_adapter = model_args.tune_speech_mlp_adapter
    if model_args.tune_speech_mlp_adapter:
        model.requires_grad_(False)
        for p in model.get_model().speech_projector.parameters():
            p.requires_grad = True

    model.config.freeze_speech_mlp_adapter = training_args.freeze_speech_mlp_adapter
    if training_args.freeze_speech_mlp_adapter:
        for p in model.get_model().speech_projector.parameters():
            p.requires_grad = False

    if training_args.bits in [4, 8]:
        model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)
        model.get_model().speech_projector.to(dtype=compute_dtype, device=training_args.device)

    model.config.mm_use_im_start_end = model_args.mm_use_im_start_end
    data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    # print("DATA ARGS ==>", data_args)
    model.config.mm_projector_lr = training_args.mm_projector_lr
    model.config.speech_projector_lr = training_args.speech_projector_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    # data_module = make_supervised_data_module(tokenizer=tokenizer,
    #                                           data_args=data_args)
    dpo_dataset = make_dpo_data_module(tokenizer=tokenizer,
                                       data_args=data_args)

    data_collator = DPODataCollator(
            tokenizer,
            label_pad_token_id=IGNORE_INDEX,
            pad_token_id=tokenizer.pad_token_id,
            data_args=data_args
        )

    # trainer = LLaVATrainer(model=model,
    #                 tokenizer=tokenizer,
    #                 args=training_args,
    #                 **data_module)
    trainer = LLaVADPOTrainer(
        model=model,
        args=training_args,
        dpo_alpha=training_args.dpo_alpha,
        beta=training_args.beta,
        gamma=training_args.gamma,
        train_dataset=dpo_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        tokenizer=tokenizer,
        max_length=training_args.model_max_length,
        generate_during_eval=False, #training_args.generate_during_eval,
        tpd_gamma=training_args.tpd_gamma,
        use_tpd=training_args.use_tpd
    )
    trainer.save_my_lora_ckpt = save_my_lora_ckpt

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
