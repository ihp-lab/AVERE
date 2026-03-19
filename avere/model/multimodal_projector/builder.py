import torch
import torch.nn as nn
import re
from .Qformer import BertConfig, BertLMHeadModel
import torch.nn.functional as F

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)

class VideoQFormer(nn.Module):
    @classmethod
    def init_video_Qformer(cls, num_query_token, video_feature_width, num_hidden_layers=2):
        encoder_config = BertConfig.from_pretrained("./backbones/bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = video_feature_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def __init__(self, config):
        super(VideoQFormer, self).__init__()
        input_dim = config.mm_hidden_size
        num_video_query_tokens = config.qformer_num_video_query_tokens
        self.window_level_qformer = config.video_qformer_window_level
        self.video_tokens_per_window = config.video_qformer_tokens_per_window
        self.video_token_stride = config.video_qformer_token_stride

        self.ln_video = nn.LayerNorm(input_dim)
        self.video_Qformer, self.video_query_tokens = self.init_video_Qformer(
                    num_query_token=num_video_query_tokens, video_feature_width=input_dim
                )
        
        self.video_Qformer.bert.embeddings.word_embeddings = None
        self.video_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.video_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.video_Qformer.cls = None

        # following functionality is not supported right now. TODO
        # if freeze_speech_qformer:
        #     for name, param in self.speech_Qformer.named_parameters():
        #         param.requires_grad = False
        #     self.speech_Qformer.eval()
        #     self.speech_query_tokens.requires_grad = False

        self.video_proj = nn.Linear(
            self.video_Qformer.config.hidden_size, config.hidden_size
        )

    def forward(self, video_embeds):
        if video_embeds.ndim == 4:
            B, T, N, C = video_embeds.shape
            video_embeds = video_embeds.view(B, T*N, C)
        
        # print(f"Video embeds shape before ln: {video_embeds.shape}")
        video_embeds = self.ln_video(video_embeds)
        # print(f"Video embeds shape after ln: {video_embeds.shape}")
        video_atts = torch.ones(video_embeds.size()[:-1], dtype=torch.long).to(video_embeds.device)
        # print(f"Video attention shape: {video_atts.shape}")

        if self.window_level_qformer:
            B, T, C = video_embeds.shape
            kernel = self.video_tokens_per_window
            stride = self.video_token_stride
            # print(f"Video Q-former kernel = {kernel}, stride = {stride}")
            kernel = (1, kernel)
            stride = (1, stride)
            video_embeds_tr = video_embeds.transpose(1, 2).unsqueeze(2)
            video_embeds_overlap = F.unfold(video_embeds_tr, kernel_size=kernel, dilation=1, padding=0, stride=stride)
            _, _, L = video_embeds_overlap.shape
            video_embeds_overlap = video_embeds_overlap.view(B, -1, kernel[1], L)
            video_embeds_overlap = torch.permute(video_embeds_overlap, [0, 3, 2, 1])
            video_embeds = video_embeds_overlap.reshape(-1, kernel[1], C)
            video_atts = torch.ones(video_embeds.size()[:-1], dtype=torch.long, device=video_embeds.device)
        else:
            raise NotImplementedError("Only window level Q-Former is supported for video.")

        query_tokens = self.video_query_tokens.expand(video_embeds.shape[0], -1, -1)
        query_output = self.video_Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=video_embeds,
            encoder_attention_mask=video_atts,
            return_dict=True,
        )
        video_embeds = self.video_proj(query_output.last_hidden_state)

        if self.window_level_qformer:
            video_embeds = video_embeds.view(B, -1, video_embeds.size(2)).contiguous()
        
        # print(f"video embeds shape after Q-Former: {video_embeds.shape}")

        return video_embeds


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    elif projector_type == "qformer":
        return VideoQFormer(config)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

def concatenate_tokens_window(x, window_size):
    """
    Args:
        x: Tensor of shape (B, NT, 1280)
        window_size: Number of tokens to group together
    Returns:
        Tensor of shape (B, NT // window_size, 1280 * window_size)
    """
    B, NT, D = x.shape
    assert NT % window_size == 0, "NT must be divisible by window_size"

    x = x.view(B, NT // window_size, window_size* D)   # (B, num_windows, window_size, D)
    return x

class NaiveSpeechProjector(nn.Module):
    def __init__(self, config, input_tokens=1500, window_size=10):
        super(NaiveSpeechProjector, self).__init__()
        
        input_dim = config.speech_hidden_size

        mlp_depth = 2
        modules = [nn.Linear(input_dim*window_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        self.projection = nn.Sequential(*modules)

        self.window_size = window_size

    def forward(self, x):
        x = concatenate_tokens_window(x, self.window_size)
        return self.projection(x)

class SpeechQFormer(nn.Module):
    @classmethod
    def init_speech_Qformer(cls, num_query_token, speech_width, num_hidden_layers=2):
        encoder_config = BertConfig.from_pretrained("./backbones/bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = speech_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def __init__(self, config):
        super(SpeechQFormer, self).__init__()
        input_dim = config.speech_hidden_size
        num_speech_query_tokens = config.qformer_num_speech_query_tokens
        self.window_level_qformer = config.qformer_window_level
        self.second_per_window = config.qformer_second_per_window
        self.second_stride = config.qformer_second_stride

        self.ln_speech = nn.LayerNorm(input_dim)
        self.speech_Qformer, self.speech_query_tokens = self.init_speech_Qformer(
                    num_query_token=num_speech_query_tokens, speech_width=input_dim
                )
        
        self.speech_Qformer.bert.embeddings.word_embeddings = None
        self.speech_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.speech_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.speech_Qformer.cls = None

        # following functionality is not supported right now. TODO
        # if freeze_speech_qformer:
        #     for name, param in self.speech_Qformer.named_parameters():
        #         param.requires_grad = False
        #     self.speech_Qformer.eval()
        #     self.speech_query_tokens.requires_grad = False

        self.speech_proj = nn.Linear(
            self.speech_Qformer.config.hidden_size, config.hidden_size
        )

    def forward(self, speech_embeds):
        # print(f"Speech embeds shape before ln: {speech_embeds.shape}")
        speech_embeds = self.ln_speech(speech_embeds)
        # print(f"Speech embeds shape after ln: {speech_embeds.shape}")
        speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)
        # print(f"Speech attention shape: {speech_atts.shape}")

        if self.window_level_qformer:
            B, T, C = speech_embeds.shape
            kernel = round(1500 * self.second_per_window / 30.0)
            stride = round(1500 * self.second_stride / 30.0)
            # print(f"kernel = {kernel}, stride = {stride}")
            kernel = (1, kernel)
            stride = (1, stride)
            speech_embeds_tr = speech_embeds.transpose(1, 2).unsqueeze(2)
            speech_embeds_overlap = F.unfold(speech_embeds_tr, kernel_size=kernel, dilation=1, padding=0, stride=stride)
            _, _, L = speech_embeds_overlap.shape
            speech_embeds_overlap = speech_embeds_overlap.view(B, -1, kernel[1], L)
            speech_embeds_overlap = torch.permute(speech_embeds_overlap, [0, 3, 2, 1])
            speech_embeds = speech_embeds_overlap.reshape(-1, kernel[1], C)
            speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long, device=speech_embeds.device)
        else:
            raise NotImplementedError("Only window level Q-Former is supported for speech.")

        query_tokens = self.speech_query_tokens.expand(speech_embeds.shape[0], -1, -1)
        query_output = self.speech_Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=speech_embeds,
            encoder_attention_mask=speech_atts,
            return_dict=True,
        )
        speech_embeds = self.speech_proj(query_output.last_hidden_state)

        if self.window_level_qformer:
            speech_embeds = speech_embeds.view(B, -1, speech_embeds.size(2)).contiguous()
        
        # print(f"Speech embeds shape after Q-Former: {speech_embeds.shape}")

        return speech_embeds


def build_speech_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'speech_projector_type', 'naive')

    if projector_type == "naive":
        return NaiveSpeechProjector(config)
    
    elif projector_type == "qformer":
        return SpeechQFormer(config)

    raise ValueError(f'Unknown speech projector type: {projector_type}')

