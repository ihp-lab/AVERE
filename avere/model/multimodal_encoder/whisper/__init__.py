import torch.nn as nn
import torch
from .modeling_whisper import WhisperModel
from .processing_audio import WhisperAudioProcessor
from transformers.models.whisper.configuration_whisper import WhisperConfig

class WhisperSpeechTower(nn.Module):
    def __init__(self, speech_tower, args, delay_load=False, cache_dir='./cache_dir'):
        super().__init__()

        self.is_loaded = False

        self.speech_tower_name = speech_tower

        self.cache_dir = cache_dir

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = WhisperConfig.from_pretrained(self.speech_tower_name, cache_dir=self.cache_dir)

    ############################################################
    def load_model(self):
        model = WhisperModel.from_pretrained(self.speech_tower_name, cache_dir=self.cache_dir)
        self.speech_tower = model.encoder
        self.speech_tower.requires_grad_(False)
        self.speech_tower.eval()
        self.speech_processor = WhisperAudioProcessor(self.speech_tower_name)

        self.is_loaded = True

    @torch.no_grad()
    def forward(self, audios):
        if type(audios) is list:
            speech_features = []
            for audio in audios:
                speech_feature = self.speech_tower(audio.to(device=self.device, dtype=self.dtype).unsqueeze(0), return_dict=True).last_hidden_state
                speech_features.append(speech_feature)
        else:
            speech_features = self.speech_tower(audios.to(device=self.device, dtype=self.dtype), return_dict=True).last_hidden_state

        return speech_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.speech_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.speech_tower.parameters()).device

    @property
    def config(self):
        if self.is_loaded:
            return self.speech_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size