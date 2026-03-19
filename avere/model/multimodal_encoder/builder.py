import os
from .clip_encoder import CLIPVisionTower
from .languagebind import LanguageBindImageTower, LanguageBindVideoTower
from .whisper import WhisperSpeechTower

# ============================================================================================================

def build_image_tower(image_tower_cfg, **kwargs):
    image_tower = getattr(image_tower_cfg, 'mm_image_tower', getattr(image_tower_cfg, 'image_tower', None))
    is_absolute_path_exists = os.path.exists(image_tower)
    # if is_absolute_path_exists or image_tower.startswith("openai") or image_tower.startswith("laion"):
    #     return CLIPVisionTower(image_tower, args=image_tower_cfg, **kwargs)
    # print("IMAGE_TOWER===>", image_tower)
    if image_tower.endswith('LanguageBind_Image'):
        return LanguageBindImageTower(image_tower, args=image_tower_cfg, cache_dir='./cache_dir', **kwargs)

    raise ValueError(f'Unknown image tower: {image_tower}')

def build_video_tower(video_tower_cfg, **kwargs):
    video_tower = getattr(video_tower_cfg, 'mm_video_tower', getattr(video_tower_cfg, 'video_tower', None))
    if video_tower.endswith('LanguageBind_Video_merge'):
        return LanguageBindVideoTower(video_tower, args=video_tower_cfg, cache_dir='./cache_dir', **kwargs)
    raise ValueError(f'Unknown video tower: {video_tower}')
# ============================================================================================================


def build_speech_tower(speech_tower_cfg, **kwargs):
    speech_tower = getattr(speech_tower_cfg, "mm_speech_tower", getattr(speech_tower_cfg, 'speech_tower', None))
    if "whisper" in speech_tower:
        return WhisperSpeechTower(speech_tower, args=speech_tower_cfg, cache_dir='./cache_dir', **kwargs)
    raise ValueError(f'Unknown speech tower: {video_tower}')

# def test_speech_encoder():
#     from transformers import WhisperFeatureExtractor
#     import soundfile as sf
#     wav_file = "/wekafs/ict/achaubey/emotion_reasoning/audio_exp/data/seed_examples/sample_3.wav"
#     audio, sr = sf.read(wav_file)
#     if len(audio.shape) == 2: # stereo to mono
#         audio = audio[:, 0]
    
#     if len(audio) < sr: # pad audio to at least 1s
#         sil = np.zeros(sr - len(audio), dtype=float)
#         audio = np.concatenate((audio, sil), axis=0)
#     audio = audio[: sr * 30] # truncate audio to at most 30s
#     print(audio.shape)
#     wav_processor = WhisperFeatureExtractor.from_pretrained("/wekafs/ict/achaubey/emotion_reasoning/audio_exp/SALMONN/whisper-large-v2")

#     spectrogram = wav_processor([audio, audio], sampling_rate=sr, return_tensors="pt")["input_features"]#.squeeze()
#     speech_encoder = WhisperModel.from_pretrained("/wekafs/ict/achaubey/emotion_reasoning/audio_exp/SALMONN/whisper-large-v2").encoder
#     speech_embeds = speech_encoder(spectrogram, return_dict=True).last_hidden_state
#     print(speech_embeds.shape)

# if __name__=="__main__":
#     test_speech_encoder()