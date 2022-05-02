#!/usr/bin/env python3
import os

src = r'/mnt/SSD_Disk/FrVC/manifest/path/train.tsv'
pth = r'/mnt/SSD_Disk/FrVC/MyCorpus/wav'
pth_excl = r'/mnt/SSD_Disk/FrVC/MyCorpus/excl_files.txt'
#excl_list = list()

with (open(file=src, mode='r')) as f:
    ss = f.readlines()
ss = ss[1:]
i=0
with open(pth_excl,mode='a') as fout:
    for s in ss:
        s = s[:19]
        ns = os.path.join(pth,s)
#        excl_list.append(ns)
        fout.write(ns+'\n')
        i=+1
    print(' Записано строк в файл : {}'.format(i))


