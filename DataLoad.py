

import os
import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd

from src.FST.transducers import TexNormalizer
from src.FST.transducers import Normalizer

AUDIO_DIR = "Data/audio/"
TEX_DIR = "Data/latex/"
SEQ_DIR = "Data/sequences/"

class TexData(Dataset):
    
    seq_normalizer = Normalizer()
    tex_normalizer = TexNormalizer()
    
    def __init__(self, audio_dir = AUDIO_DIR, tex_dir = TEX_DIR, seq_dir = SEQ_DIR, normalize=True):
        self.audio_dir = audio_dir
        self.tex_dir = tex_dir
        self.seq_dir = seq_dir
        self.file_names = list(self._get_common_file_names())
        self.normalize = normalize

    def _get_common_file_names(self):
        audio_files = set([fname.split('.')[0] for fname in os.listdir(self.audio_dir) if fname.endswith('.wav')])
        tex_files   = set([fname.split('.')[0] for fname in os.listdir(self.tex_dir) if fname.endswith('.txt')])
        seq_files   = set([fname.split('.')[0] for fname in os.listdir(self.seq_dir) if fname.endswith('.txt')])
        if (audio_files != tex_files) : raise Exception("audio files and tex files are different")
        if (audio_files != seq_files) : raise Exception("audio files and seq files are different")
        return audio_files

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):

        file_name = self.file_names[idx]

        audio_path = os.path.join(self.audio_dir, file_name + '.wav')
        tex_path = os.path.join(self.tex_dir, file_name + '.txt')
        seq_path = os.path.join(self.seq_dir, file_name + '.txt')

        uid = -1
        for i in range(3):
            if 'p' + str(i+1) in file_name :
                uid = i

        ac = ('dna' in file_name)

        # Load audio file
        waveform, sample_rate = torchaudio.load(audio_path)

        # Load text files
        with open(tex_path, 'r') as f : tex = f.read().split('\n')[0]
        with open(seq_path, 'r') as f : seq = f.read().split('\n')[0]
        if self.normalize : 
            tex = self.tex_normalizer.predict(tex)
            seq = self.seq_normalizer.predict(seq)

        sample = {
            "id":file_name,
            "uid":uid,
            "audio_path":audio_path,
            "sample_rate":sample_rate,
            "seq":seq,
            "tex":tex,
            "ac":ac,
        }

        return sample
    
    def get_csv(self):
        df = pd.DataFrame(columns=['id', 'uid', 'audio_path', 'sample_rate', 'seq', 'tex', 'ac'])
        for i in range(len(self)):
            new_row = self[i]
            df.loc[i] = new_row
        return df
