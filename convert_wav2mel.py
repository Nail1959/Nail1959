#!/usr/bin/env python3
# Создание папок с mel спектрограммами для вокодера
import os
from utils import wav2mel

path_in = r'/home/nail/MyCorpus/wav_train'
path_out = r'/home/nail/MyCorpus/mel_train'

path_readers = os.listdir(path_in)
path_readers.sort()
for path_reader in path_readers:
    pmel_reader = os.path.join(path_out, path_reader)
    os.makedirs(pmel_reader,exist_ok=True)
    path_reader = os.path.join(path_in, path_reader)
    flist = os.listdir(path_reader)
    os.chdir(path_reader)
    flist.sort()
    for f in flist:
        f_out = f.replace('wav','npy')
        f_out = os.path.join(pmel_reader, f_out)
        wav2mel(f, f_out)


