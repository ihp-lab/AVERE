import torch
import soundfile as sf
import numpy as np
from transformers import WhisperFeatureExtractor
from transformers import ProcessorMixin

def make_list_of_audios(x):
    if not isinstance(x, list):
        return [x]
    return x

def load_audio(audio_path):
    audio, sr = sf.read(audio_path)
    if len(audio.shape) == 2: # stereo to mono
        audio = audio[:, 0]
    
    if len(audio) < sr: # pad audio to at least 1s
        sil = np.zeros(sr - len(audio), dtype=float)
        audio = np.concatenate((audio, sil), axis=0)
    audio = audio[: sr * 30] # truncate audio to at most 30s
    return audio, sr


class WhisperAudioProcessor(ProcessorMixin):
    attributes = []
    tokenizer_class = ("WhisperAudioProcessor")

    def __init__(self, speech_tower_name, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        # self.config = config
        self.audio_processor = load_audio
        speech_tower = speech_tower_name
        self.spectrogram_extractor = WhisperFeatureExtractor.from_pretrained(speech_tower)
        self.tokenizer = tokenizer

    def __call__(self, audios=None, text=None, return_tensors=None, **kwargs):
        if text is None and audios is None:
            raise ValueError("You have to specify either text or audio. Both cannot be none.")

        if text is not None:
            raise NotImplementedError("Text support not present for the WhisperAudioProcessor")

        if audios is not None:
            audios = make_list_of_audios(audios)
            audio_features = []
            srs = []
            for audio in audios:
                audio_feat, sr = self.audio_processor(audio)
                audio_features.append(audio_feat)
                srs.append(sr)
            # check if the audios are at different sampling rates
            if len(set(srs))!=1:
                raise Exception("The audio files in the current batch ar at different sampling rates...")
            spectrogram_features = self.spectrogram_extractor(audio_features, sampling_rate=srs[0], return_tensors="pt")["input_features"]

        if text is not None and audios is not None:
            encoding["spectrogram"] = spectrogram_features
            return encoding
        elif text is not None:
            return encoding
        else:
            return {"spectrogram": spectrogram_features}

    def preprocess(self, audios, return_tensors):
        return self.__call__(audios=audios, return_tensors=return_tensors)

    def batch_decode(self, skip_special_tokens=True, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, skip_special_tokens=skip_special_tokens, **kwargs)

    def decode(self, skip_special_tokens=True, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, skip_special_tokens=skip_special_tokens, **kwargs)
