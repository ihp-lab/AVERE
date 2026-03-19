import os
import torch
import glob, os, tqdm, re, json, logging
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_AUDIO_TOKEN, AUDIO_TOKEN_INDEX
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, tokenizer_audio_token, tokenizer_audio_and_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from utils import strip_trailing_unk, strip_end_tag, chunk_path_list

class AVEREInference:

    def __init__(self, model_path, model_base=None):

        # initialize model
        disable_torch_init()
        cache_dir = 'cache_dir'
        device = 'cuda'
        load_4bit, load_8bit = False, False
        self.model_path = model_path
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, processor, _ = load_pretrained_model(model_path, model_base, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)
        self.model.eval()
        self.speech_processor = processor['speech']
        self.video_processor = processor['video']

    def _inference_batch(self, eval_prompt, batch, pred_path, max_new_tokens=5):
        batch_size = len(batch)
        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()
        roles = conv.roles
        # print(batch)
        cur_videos, cur_audios = None, None
        ### check the type of input batch
        if type(batch[0]) is dict:
            if "video" in batch[0]:
                cur_videos = [fpath["video"] for fpath in batch]
            if "audio" in batch[0]:
                cur_audios = [fpath["audio"] for fpath in batch]
        elif type(batch[0]) is str:
            if batch[0].endswith(('.mp4', '.avi', '.mov', '.mkv')):  # assuming video files
                cur_videos = batch
            elif batch[0].endswith(('.wav', '.flac')):  # assuming audio files
                cur_audios = batch
        else:
            raise ValueError("Unsupported batch format. Expected list of dicts or list of file paths.")

        logging.info(f"No. of cur_videos: {len(cur_videos) if cur_videos is not None else 0}, cur_audios: {len(cur_audios) if cur_audios is not None else 0}")

        if cur_audios is not None:
            # print(self.speech_processor)
            speech_tensor = [audio.to(self.model.device, dtype=torch.float16) for audio in self.speech_processor(cur_audios, return_tensors='pt')['spectrogram']]
        else:
            speech_tensor = None
        if cur_videos is not None:
            video_tensor = [video.to(self.model.device, dtype=torch.float16) for video in self.video_processor(cur_videos, return_tensors='pt')['pixel_values']]
        else:
            video_tensor = None
        
        # print("Inside batch inference 2")
        if eval_prompt != "custom_qa":
            inp = eval_prompt
            # print(f"{roles[1]}: {inp}")
            if cur_audios is not None:
                inp = DEFAULT_AUDIO_TOKEN + '\n' + inp
            if cur_videos is not None:
                inp = DEFAULT_IMAGE_TOKEN*self.model.get_video_tower().config.num_frames + '\n' + inp
        
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            if cur_videos is None and cur_audios is not None:
                input_ids = tokenizer_audio_token(prompt, self.tokenizer, AUDIO_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            elif cur_audios is None and cur_videos is not None:
                input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            elif cur_audios is not None and cur_videos is not None:
                input_ids = tokenizer_audio_and_image_token(prompt, self.tokenizer, return_tensors='pt').unsqueeze(0).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
            input_ids = torch.cat([input_ids]*batch_size, dim=0)
        else:
            custom_input_ids = []
            for sample in batch:
                if "eval_prompt" not in sample or "sample_id" not in sample:
                    raise ValueError("Each sample in the batch must contain 'eval_prompt' and 'sample_id' because prompt type is custom qa.")
                cur_eval_prompt = sample["eval_prompt"]
                sample_id = sample["sample_id"]
                inp = cur_eval_prompt
                if cur_audios is not None:
                    inp = DEFAULT_AUDIO_TOKEN + '\n' + inp
                if cur_videos is not None:
                    inp = DEFAULT_IMAGE_TOKEN*self.model.get_video_tower().config.num_frames + '\n' + inp
                conv.append_message(conv.roles[0], inp)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                if cur_videos is None and cur_audios is not None:
                    input_ids = tokenizer_audio_token(prompt, self.tokenizer, AUDIO_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                elif cur_audios is None and cur_videos is not None:
                    input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                elif cur_audios is not None and cur_videos is not None:
                    input_ids = tokenizer_audio_and_image_token(prompt, self.tokenizer, return_tensors='pt').unsqueeze(0).cuda()
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
                custom_input_ids.append(input_ids)
            input_ids = torch.cat(custom_input_ids, dim=0)
        
        
        # print("Inside batch inference 3")
        # try:
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                audios=speech_tensor,
                images=video_tensor,
                do_sample=True,
                temperature=0.1,
                max_new_tokens=max_new_tokens,
                use_cache=False,
                stopping_criteria=[stopping_criteria])
        # except Exception as e:
        #     print(f"Error during inference: {e}")
        #     print(f"Input IDs: {input_ids}")
        #     print(f"Speech Tensor: {speech_tensor}")
        #     print(f"Stopping Criteria: {stopping_criteria}")
        #     return None
                
        outputs = self.tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:])
        outputs = [strip_end_tag(strip_trailing_unk(output)).strip() for output in outputs]
        # print("Inside batch inference 5")
        if eval_prompt != "custom_qa":
            cur_files = cur_videos if cur_videos is not None else cur_audios
            for bidx, fpath in enumerate(cur_files):
                fname = fpath.split("/")[-1].split(".")[0]
                save_path = os.path.join(pred_path, f"{fname}.txt")
                with open(save_path, "w") as f:
                    f.write(outputs[bidx])
        else:
            for bidx, sample in enumerate(batch):
                if "sample_id" not in sample:
                    raise ValueError("Each sample in the batch must contain 'sample_id' because prompt type is custom qa.")
                sample_id = sample["sample_id"]
                save_path = os.path.join(pred_path, f"{sample_id}.txt")
                with open(save_path, "w") as f:
                    f.write(outputs[bidx])
        return outputs

    def inference_all(self, eval_prompt, all_files, temp_eval_path, batch_size, multi_gpu_split=False, num_gpus=1, gpu_split_idx=0, max_new_tokens=5):

        if multi_gpu_split:
            assert num_gpus>1, "num_gpus should be > 1 if multi_gpu_split is True"
            assert gpu_split_idx>=0 and gpu_split_idx<num_gpus, "gpu_split_idx should be between [0, num_gpus-1]"

        pred_path = os.path.join(temp_eval_path, "preds")
        os.makedirs(pred_path, exist_ok=True)

        # check which samples are already present in the pred_path
        new_samples = []
        for cur_fpath in all_files:
            if type(cur_fpath) is dict:
                fpath = cur_fpath["video"] if "video" in cur_fpath else cur_fpath["audio"]
            else:
                fpath = cur_fpath
            fname = fpath.split("/")[-1].split(".")[0]
            save_path = os.path.join(pred_path, f"{fname}.txt")
            if os.path.exists(save_path):
                continue
            new_samples.append(cur_fpath)

        batches = chunk_path_list(new_samples, batch_size)

        if multi_gpu_split:
            total_batches = len(batches)
            batches_per_gpu = (total_batches + num_gpus - 1) // num_gpus
            start_idx = gpu_split_idx * batches_per_gpu
            end_idx = min(start_idx + batches_per_gpu, total_batches)
            batches = batches[start_idx:end_idx]
            print(f"GPU {gpu_split_idx} processing batches from index {start_idx} to {end_idx} out of {total_batches} total batches.")
        
        for batch in tqdm.tqdm(batches, desc="Running inference on batches of files..."):
            cur_preds = self._inference_batch(eval_prompt, batch, pred_path, max_new_tokens)
        
        return pred_path
    
    def inference_all_qa(self, eval_prompts, all_files, all_sample_ids, temp_eval_path, batch_size, multi_gpu_split=False, num_gpus=1, gpu_split_idx=0, max_new_tokens=5):

        if multi_gpu_split:
            assert num_gpus>1, "num_gpus should be > 1 if multi_gpu_split is True"
            assert gpu_split_idx>=0 and gpu_split_idx<num_gpus, "gpu_split_idx should be between [0, num_gpus-1]"
            

        pred_path = os.path.join(temp_eval_path, "preds")
        os.makedirs(pred_path, exist_ok=True)

        # check which samples are already present in the pred_path
        new_samples = []
        for cur_sample_id, cur_fpath, cur_prompt in zip(all_sample_ids, all_files, eval_prompts):
            save_path = os.path.join(pred_path, f"{cur_sample_id}.txt")
            if os.path.exists(save_path):
                continue
            if type(cur_fpath) is dict:
                if "video" in cur_fpath and "audio" in cur_fpath:
                    new_samples.append({"video": cur_fpath["video"], "audio": cur_fpath["audio"], "eval_prompt": cur_prompt, "sample_id": cur_sample_id})
                elif "video" in cur_fpath:
                    new_samples.append({"video": cur_fpath["video"], "eval_prompt": cur_prompt, "sample_id": cur_sample_id})
                elif "audio" in cur_fpath:
                    new_samples.append({"audio": cur_fpath["audio"], "eval_prompt": cur_prompt, "sample_id": cur_sample_id})
            else:
                new_samples.append({"video": cur_fpath, "eval_prompt": cur_prompt, "sample_id": cur_sample_id})

        batches = chunk_path_list(new_samples, batch_size)

        if multi_gpu_split:
            total_batches = len(batches)
            batches_per_gpu = (total_batches + num_gpus - 1) // num_gpus
            start_idx = gpu_split_idx * batches_per_gpu
            end_idx = min(start_idx + batches_per_gpu, total_batches)
            batches = batches[start_idx:end_idx]
            print(f"GPU {gpu_split_idx} processing batches from index {start_idx} to {end_idx} out of {total_batches} total batches.")
        
        for batch in tqdm.tqdm(batches, desc="Running inference on batches of files..."):
            cur_preds = self._inference_batch("custom_qa", batch, pred_path, max_new_tokens)
        
        return pred_path
