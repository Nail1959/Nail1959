import numpy as np
import os
from PIL import Image
import soundfile
import librosa
sr = 16000
n_fft = 2048
win_len = 2048
hop_length = 512
n_mels = 128

# def ttf(f):
#     y, sr = librosa.load(f)
#     x, _ = librosa.effects.trim(y)
#     n_fft = 2048
#     hop_length = 512
#     n_mels = 128
#     pmel = librosa.feature.melspectrogram(x, sr=sr, n_fft=n_fft,
#                                        hop_length=hop_length, n_mels=n_mels)
#
#     # pmin = p.min()
#     # pp = p+abs(pmin)
#     # pmax = abs(pp.max())
#     # pp = (pp / pmax) *256
#
#     im = Image.fromarray(pmel).convert(mode='F')
#     im.show()
#     im.save("/mnt/SSD_Disk/FrVC/OutPath/sp.tiff")
#     #im.save("/mnt/SSD_Disk/FrVC/OutPath/sp.png")
#
#     # from tiff to wav
#     im = Image.open("/mnt/SSD_Disk/FrVC/OutPath/sp.tiff")
#     img = np.array(im)
#     # img = np.array(im) * pmax/256  - abs(pmin)
#     wav = librosa.feature.inverse.mel_to_audio(img)
#     print(img)
#
#     soundfile.write("/mnt/SSD_Disk/FrVC/OutPath/result.wav", wav, samplerate=sr)

def mel2wav(mel_f, wav_f):
    arr = np.load(mel_f)
    wav = librosa.feature.inverse.mel_to_audio(arr, sr=sr,n_fft=n_fft,hop_length=hop_length, n_iter=32)
    soundfile.write(wav_f, wav, samplerate=sr)

if __name__ == '__main__':
    mel_f = r'/home/nail/MyCorpus/mel_train/p010/p010_00001.npy'
    wav_f = r'/home/nail/MyCorpus/mel_train/p010/p010_00001.wav'
    mel2wav(mel_f, wav_f)
