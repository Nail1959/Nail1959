#!/usr/bin/env python3
# Утилиты для создания mel спектрограмм для вокодера
import numpy as np
import librosa
from scipy.signal import lfilter
from copy import deepcopy
import sox
from PIL import Image
import soundfile
#========================================================
# Константы для Вокодера
sr = 16000
n_fft = 1304
win_len = 1304
hop_length = 326
n_mels = 80
f_min = 80
preemph = 0.97
# ========================================================
tfm = sox.Transformer()
tfm.vad(location=1)
tfm.vad(location=-1)
def load_wav(
    audio_path: str, sample_rate: int, trim: bool = False
) -> np.ndarray:
    """Load and preprocess waveform."""
    wav = librosa.load(audio_path, sr=sample_rate)[0]
    wav = wav / (np.abs(wav).max() + 1e-6)
    if trim:
        _, (start_frame, end_frame) = librosa.effects.trim(
            wav, top_db=25, frame_length=512, hop_length=128
        )
        start_frame = max(0, start_frame - 0.1 * sample_rate)
        end_frame = min(len(wav), end_frame + 0.1 * sample_rate)

        start = int(start_frame)
        end = int(end_frame)
        if end - start > 1000:  # prevent empty slice
            wav = wav[start:end]

    return wav

def log_mel_spectrogram(
    x: np.ndarray,
    preemph: float,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    f_min: int,
) -> np.ndarray:
    """Create a log Mel spectrogram from a raw audio signal."""
    x = lfilter([1, -preemph], [1], x)
    magnitude = np.abs(
        librosa.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    )
    mel_fb = librosa.filters.mel(sample_rate, n_fft, n_mels=n_mels, fmin=f_min)
    mel_spec = np.dot(mel_fb, magnitude)
    log_mel_spec = np.log(mel_spec + 1e-9)
    return log_mel_spec.T

def wav2mel(file_in, file_out, sr=16000):
    src_wav = load_wav(file_in, sample_rate=sr)
    src_wav = tfm.build_array(input_array=src_wav, sample_rate_in=sr)
    src_wav = deepcopy(src_wav)
    src_mel = log_mel_spectrogram(
        src_wav, preemph, sr, n_mels, n_fft, hop_length, win_len, f_min
    )
    np.save(file_out, src_mel)
    y, sr = librosa.load(file_in)
    x, _ = librosa.effects.trim(y)
    p_mel = log_mel_spectrogram(
        x=x, preemph=0.97, sample_rate=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_len, f_min=f_min
    )

    # pmel = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=n_fft,
    #                                       hop_length=hop_length, win_length=win_len, n_mels=n_mels)
    np.save(file_out, p_mel)

def ttf(f):
    y, sr = librosa.load(f)
    x, _ = librosa.effects.trim(y)

    pmel = librosa.feature.melspectrogram(x, sr=sr, n_fft=n_fft,
                                       hop_length=hop_length, n_mels=n_mels)

    im = Image.fromarray(pmel).convert(mode='F')
    im.show()
    im.save("/mnt/SSD_Disk/FrVC/OutPath/sp.tiff")


    # from tiff to wav
    im = Image.open("/mnt/SSD_Disk/FrVC/OutPath/sp.tiff")
    img = np.array(im)
    # img = np.array(im) * pmax/256  - abs(pmin)
    wav = librosa.feature.inverse.mel_to_audio(img)
    print(img)

    soundfile.write("/mnt/SSD_Disk/FrVC/OutPath/result.wav", wav, samplerate=sr)