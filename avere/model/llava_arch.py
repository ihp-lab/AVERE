#    Copyright 2023 Haotian Liu
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


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_image_tower, build_video_tower, build_speech_tower
from .multimodal_projector.builder import build_vision_projector, build_speech_projector

from avere.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, AUDIO_TOKEN_INDEX


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if getattr(config, "mm_image_tower", None) is not None:
            self.image_tower = build_image_tower(config, delay_load=True)
        if getattr(config, "mm_video_tower", None) is not None:
            self.video_tower = build_video_tower(config, delay_load=True)
        if getattr(config, "mm_image_tower", None) is not None or getattr(config, "mm_video_tower", None) is not None:
            self.mm_projector = build_vision_projector(config)

        if getattr(config, 'mm_speech_tower', None) is not None:
            self.speech_tower = build_speech_tower(config, delay_load=True)
        if getattr(config, "mm_speech_tower", None) is not None:
            self.speech_projector = build_speech_projector(config)

    def get_image_tower(self):
        image_tower = getattr(self, 'image_tower', None)
        if type(image_tower) is list:
            image_tower = image_tower[0]
        return image_tower

    def get_video_tower(self):
        video_tower = getattr(self, 'video_tower', None)
        if type(video_tower) is list:
            video_tower = video_tower[0]
        return video_tower

    def get_speech_tower(self):
        speech_tower = getattr(self, 'speech_tower', None)
        if type(speech_tower) is list:
            speech_tower = speech_tower[0]
        return speech_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        # ==============================================
        image_tower = model_args.image_tower
        video_tower = model_args.video_tower
        assert image_tower is not None or video_tower is not None
        # ==============================================
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        # ==========================================================================

        self.config.mm_image_tower = image_tower
        if image_tower is not None:
            if self.get_image_tower() is None:
                image_tower = build_image_tower(model_args)

                if fsdp is not None and len(fsdp) > 0:
                    self.image_tower = [image_tower]
                else:
                    self.image_tower = image_tower
            else:
                if fsdp is not None and len(fsdp) > 0:
                    image_tower = self.image_tower[0]
                else:
                    image_tower = self.image_tower
                image_tower.load_model()

        self.config.mm_video_tower = video_tower
        if video_tower is not None:
            if self.get_video_tower() is None:
                video_tower = build_video_tower(model_args)

                if fsdp is not None and len(fsdp) > 0:
                    self.video_tower = [video_tower]
                else:
                    self.video_tower = video_tower
            else:
                if fsdp is not None and len(fsdp) > 0:
                    video_tower = self.video_tower[0]
                else:
                    video_tower = self.video_tower
                video_tower.load_model()

        # ==========================================================================

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        # ==========================================================================
        if image_tower is not None and video_tower is not None:  # TODO: support different hidden_size
            assert image_tower.hidden_size == video_tower.hidden_size
            self.config.mm_hidden_size = image_tower.hidden_size
        else:
            self.config.mm_hidden_size = max(getattr(image_tower, 'hidden_size', -1),
                                             getattr(video_tower, 'hidden_size', -1))
        # ===================================================================================


        # ==========================================================================
        self.config.qformer_num_video_query_tokens = getattr(model_args, 'qformer_num_video_query_tokens', -1)
        self.config.video_qformer_window_level = getattr(model_args, 'video_qformer_window_level', None)
        self.config.video_qformer_tokens_per_window = getattr(model_args, 'video_qformer_tokens_per_window', -1)
        self.config.video_qformer_token_stride = getattr(model_args, 'video_qformer_token_stride', -1)

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
    
    def initialize_audio_modules(self, model_args, fsdp=None):
        # ==============================================
        speech_tower = model_args.speech_tower
        assert speech_tower is not None
        # ============================================== 
        pretrain_speech_mlp_adapter = model_args.pretrain_speech_mlp_adapter

        # ==========================================================================

        self.config.mm_speech_tower = speech_tower
        if speech_tower is not None:
            if self.get_speech_tower() is None:
                speech_tower = build_speech_tower(model_args)

                if fsdp is not None and len(fsdp) > 0:
                    self.speech_tower = [speech_tower]
                else:
                    self.speech_tower = speech_tower
            else:
                if fsdp is not None and len(fsdp) > 0:
                    speech_tower = self.speech_tower[0]
                else:
                    speech_tower = self.speech_tower
                speech_tower.load_model()
                # print("SPEECH TOWER DEVICE", speech_tower.device)

        # ==========================================================================

        self.config.use_speech_proj = True
        self.config.speech_projector_type = getattr(model_args, 'speech_projector_type', 'linear')
        # ==========================================================================
        
        self.config.speech_hidden_size = getattr(speech_tower, 'hidden_size', -1)

        # ==========================================================================
        self.config.qformer_num_speech_query_tokens = getattr(model_args, 'qformer_num_speech_query_tokens', -1)
        self.config.qformer_window_level = getattr(model_args, 'qformer_window_level', None)
        self.config.qformer_second_per_window = getattr(model_args, 'qformer_second_per_window', -1)
        self.config.qformer_second_stride = getattr(model_args, 'qformer_second_stride', -1)
        # ==========================================================================

        if getattr(self, 'speech_projector', None) is None:
            self.speech_projector = build_speech_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.speech_projector.parameters():
                p.requires_grad = True

        if pretrain_speech_mlp_adapter is not None:
            speech_projector_weights = torch.load(pretrain_speech_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.speech_projector.load_state_dict(get_w(speech_projector_weights, 'speech_projector'), strict=False) ## FIXME the code fails for strict=True as there are missing keys speech_Qformer.bert.embeddings.position_ids


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_image_tower(self):
        return self.get_model().get_image_tower()

    def get_video_tower(self):
        return self.get_model().get_video_tower()

    def get_speech_tower(self):
        return self.get_model().get_speech_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_image_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def encode_videos(self, videos):  # [mini_b, c, t, h, w]
        b, _, t, _, _ = videos.shape
        video_features = self.get_model().get_video_tower()(videos)  # [mini_b, t, n, c]
        video_features = self.get_model().mm_projector(video_features)
        return video_features

    def encode_speech(self, audios):
        speech_features = self.get_model().get_speech_tower()(audios)
        # print("1. Speech features", speech_features.shape)
        speech_features = self.get_model().speech_projector(speech_features)
        # print("2. Speech features", speech_features.shape)
        return speech_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images, audios
    ):
        # ====================================================================================================
        image_tower = self.get_image_tower()
        video_tower = self.get_video_tower()
        speech_tower = self.get_speech_tower()
        # print("CODE IS REACHING HERE === -3")
        if (image_tower is None and video_tower is None and speech_tower is None) or (images is None and audios is None) or input_ids.shape[1] == 1:
            if past_key_values is not None and (image_tower is not None or video_tower is not None) and images is not None and input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        # print("CODE IS REACHING HERE === -2")
        # print("prepare_inputs_labels_for_multimodal() - 1 - input_ids=", input_ids.shape)
        # print("prepare_inputs_labels_for_multimodal() - 1 - attention_mask=", attention_mask.shape)
        '''
            images is a list, if batch_size=6
            [
                image(3, 224, 224),      # sample 1
                image(3, 224, 224),      # sample 2
                video(t, 3, 224, 224),   # sample 3
                image(3, 224, 224),      # sample 4
                image(3, 224, 224),      # sample 4
                video(t, 3, 224, 224),   # sample 5
                video(t, 3, 224, 224),   # sample 5
                video(t, 3, 224, 224),   # sample 6
                image(3, 224, 224),      # sample 6
            ]
            will be converted to image_features, all video_feature will be flatten as image
            [
                [n, c],                  # sample 1
                [n, c),                  # sample 2
                *(t * [new_n, c]),       # sample 3
                [n, c],                  # sample 4
                [n, c],                  # sample 4
                *(t * [new_n, c]),       # sample 5
                *(t * [new_n, c]),       # sample 5
                *(t * [new_n, c]),       # sample 6
                [n, c],                  # sample 6
            ]
        '''
        # print("CODE IS REACHING HERE === -1")
        if images is not None:
            image_idx = [idx for idx, img in enumerate(images) if img.ndim == 3]
            is_all_image = len(image_idx) == len(images)
            video_idx = [idx for idx, vid in enumerate(images) if vid.ndim == 4]
            images_minibatch = torch.stack([images[idx] for idx in image_idx]) if len(image_idx) > 0 else []  # mini_b c h w
            videos_minibatch = torch.stack([images[idx] for idx in video_idx]) if len(video_idx) > 0 else []  # mini_b c t h w

            # print("CODE IS REACHING HERE === 0")

            tmp_image_features = [None] * (len(image_idx) + len(video_idx))
            if getattr(images_minibatch, 'ndim', 0) == 4:  # batch consists of images, [mini_b, c, h, w]
                if image_tower is not None:
                    image_features_minibatch = self.encode_images(images_minibatch)  # [mini_b, l, c]
                else:
                    image_features_minibatch = torch.randn(1).to(self.device)  # dummy feature for video-only training under tuning
                for i, pos in enumerate(image_idx):
                    tmp_image_features[pos] = image_features_minibatch[i]

            if getattr(videos_minibatch, 'ndim', 0) == 5:  # batch consists of videos, [mini_b, c, t, h, w]
                video_features_minibatch = self.encode_videos(videos_minibatch)  # fake list [mini_b, t, l, c]
                # print(f"video features minibatch shape = ", video_features_minibatch.shape)
                ## check for qformer kind video features
                if video_features_minibatch.ndim == 3:  # [mini_b, l, c]
                    video_features_minibatch = video_features_minibatch.view(video_features_minibatch.shape[0], 8, -1, video_features_minibatch.shape[2])
                for i, pos in enumerate(video_idx):
                    t = video_features_minibatch[i].shape[0]
                    tmp_image_features[pos] = [video_features_minibatch[i][j] for j in range(t)]

            new_tmp = []
            for image in tmp_image_features:
                # print(len(new_tmp), len(image))
                if isinstance(image, list):
                    t = len(image)
                    for i in range(t):
                        new_tmp.append(image[i])
                    # print('add video')
                else:
                    new_tmp.append(image)
            image_features = new_tmp
            # print(len(image_features), *[i.shape for i in image_features])
            # print("LEN OF IMAGE_FEATURES", len(image_features), image_features[0].shape)
            # print("CODE IS REACHING HERE === 1")
        # else:
        #     print("IMAGES IS NONE")
        # ====================================================================================================

        # audio related stuff
        if audios is not None:
            audios_minibatch = torch.stack(audios) if audios is not None else []
            speech_features = [None] * len(audios) if audios is not None else []
            # print("AUDIOS_MINIBATCH_SHAPE", (audios_minibatch).shape)
            if getattr(audios_minibatch, 'ndim', 0) == 3:
                speech_features_minibatch = self.encode_speech(audios_minibatch)
                # print(f"speech features minibatch shape = ", speech_features_minibatch.shape)
                for i in range(speech_features_minibatch.shape[0]):
                    speech_features[i] = speech_features_minibatch[i]
            # print("CODE IS REACHING HERE === 2")
            # print("LEN OF SPEECH_FEATURES", len(speech_features), speech_features[0].shape)
        # else:
        #     print("AUDIOS IS NONE")
        # ====================================================================================================

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        # print("prepare_inputs_labels_for_multimodal() - 2 - input_ids=", len(input_ids), input_ids[0].shape)
        # print("prepare_inputs_labels_for_multimodal() - 2 - attention_mask=", attention_mask.shape)

        # print("CODE IS REACHING HERE === 3")

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        cur_audio_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            num_audios = (cur_input_ids == AUDIO_TOKEN_INDEX).sum()
            # print("cur_input_ids", cur_input_ids)
            # print(f"batch_idx = {batch_idx}, num_images = {num_images}, num_audios = {num_audios}, cur_image_idx = {cur_image_idx}, cur_audio_idx = {cur_audio_idx}")
            if num_images == 0 and num_audios == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 8  # 8 is the number of frames in a video, so we skip 8 images
                cur_audio_idx += 1  # no audio, so we skip 1 audio
                continue
            elif num_images>0 and num_audios==0:
                image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
                cur_input_ids_noim = []
                cur_labels = labels[batch_idx]
                cur_labels_noim = []
                for i in range(len(image_token_indices) - 1):
                    cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                    cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
                split_sizes = [x.shape[0] for x in cur_labels_noim]
                cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
                cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
                cur_new_input_embeds = []
                cur_new_labels = []

                for i in range(num_images + 1):
                    cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                    cur_new_labels.append(cur_labels_noim[i])
                    if i < num_images:
                        # print(cur_image_idx)
                        cur_image_features = image_features[cur_image_idx]
                        cur_image_idx += 1
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

                cur_new_input_embeds = torch.cat(cur_new_input_embeds)
                cur_new_labels = torch.cat(cur_new_labels)
                
                ## increment cur_audio_idx by 1 to skip the dummy audio from the features
                cur_audio_idx += 1  # no audio, so we skip 1 audio

                new_input_embeds.append(cur_new_input_embeds)
                new_labels.append(cur_new_labels)
            elif num_audios>0 and num_images==0:
                audio_token_indices = [-1] + torch.where(cur_input_ids == AUDIO_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
                cur_input_ids_noaud = []
                cur_labels = labels[batch_idx]
                cur_labels_noaud = []
                for i in range(len(audio_token_indices) - 1):
                    cur_input_ids_noaud.append(cur_input_ids[audio_token_indices[i]+1:audio_token_indices[i+1]])
                    cur_labels_noaud.append(cur_labels[audio_token_indices[i]+1:audio_token_indices[i+1]])
                split_sizes = [x.shape[0] for x in cur_labels_noaud]
                cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noaud))
                cur_input_embeds_no_aud = torch.split(cur_input_embeds, split_sizes, dim=0)
                cur_new_input_embeds = []
                cur_new_labels = []

                for i in range(num_audios + 1):
                    # print(f"I = {i}, cur_input_embeds_no_aud[i] {cur_input_embeds_no_aud[i].shape}")
                    cur_new_input_embeds.append(cur_input_embeds_no_aud[i])
                    cur_new_labels.append(cur_labels_noaud[i])
                    if i < num_audios:
                        # print(cur_audio_idx)
                        cur_speech_features = speech_features[cur_audio_idx]
                        cur_audio_idx += 1
                        # print(f"I = {i}, cur_speech_features {cur_speech_features.shape}")
                        cur_new_input_embeds.append(cur_speech_features)
                        cur_new_labels.append(torch.full((cur_speech_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

                cur_new_input_embeds = torch.cat(cur_new_input_embeds)
                cur_new_labels = torch.cat(cur_new_labels)

                ## increment cur_image_idx by 8 to skip the dummy image from the features
                cur_image_idx += 8

                new_input_embeds.append(cur_new_input_embeds)
                new_labels.append(cur_new_labels)
            else:
                # Handle both image and audio tokens present in the same sequence
                image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
                audio_token_indices = [-1] + torch.where(cur_input_ids == AUDIO_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]

                # Merge and sort all special token indices (image/audio), keep track of type
                special_tokens = []
                for idx in torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist():
                    special_tokens.append((idx, 'image'))
                for idx in torch.where(cur_input_ids == AUDIO_TOKEN_INDEX)[0].tolist():
                    special_tokens.append((idx, 'audio'))
                special_tokens = sorted(special_tokens, key=lambda x: x[0])

                cur_labels = labels[batch_idx]
                cur_new_input_embeds = []
                cur_new_labels = []

                prev_idx = -1
                for i, (token_idx, token_type) in enumerate(special_tokens + [(cur_input_ids.shape[0], None)]):
                    # Add text between previous special token and current special token
                    text_slice = cur_input_ids[prev_idx+1:token_idx]
                    label_slice = cur_labels[prev_idx+1:token_idx]
                    if text_slice.shape[0] > 0:
                        text_embeds = self.get_model().embed_tokens(text_slice)
                        cur_new_input_embeds.append(text_embeds)
                        cur_new_labels.append(label_slice)
                    # Add special token embedding (image/audio features)
                    if token_type == 'image':
                        cur_image_features = image_features[cur_image_idx]
                        cur_image_idx += 1
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    elif token_type == 'audio':
                        cur_speech_features = speech_features[cur_audio_idx]
                        cur_audio_idx += 1
                        cur_new_input_embeds.append(cur_speech_features)
                        cur_new_labels.append(torch.full((cur_speech_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    prev_idx = token_idx

                cur_new_input_embeds = torch.cat(cur_new_input_embeds)
                cur_new_labels = torch.cat(cur_new_labels)

                new_input_embeds.append(cur_new_input_embeds)
                new_labels.append(cur_new_labels)

        # print("CODE IS REACHING HERE === 4")
        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
        
        # print("CODE IS REACHING HERE === 5")
        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        # print("CODE IS REACHING HERE === 6")

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        # print("CODE IS REACHING HERE === 7")

        # print("prepare_inputs_labels_for_multimodal() - 3 - input_ids=", len(input_ids), input_ids[0].shape)
        # print("prepare_inputs_labels_for_multimodal() - 3 - ___attention_mask=", _attention_mask.shape)
        # print("prepare_inputs_labels_for_multimodal() - 3 - attention_mask=", attention_mask.shape)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        
        # print("CODE IS REACHING HERE === 8")
        # print("prepare_inputs_labels_for_multimodal() - 4 - input_ids=", len(input_ids), input_ids[0].shape)
        # print("prepare_inputs_labels_for_multimodal() - 4 - ___attention_mask=", _attention_mask.shape)
        # print("prepare_inputs_labels_for_multimodal() - 4 - attention_mask=", attention_mask.shape)

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
