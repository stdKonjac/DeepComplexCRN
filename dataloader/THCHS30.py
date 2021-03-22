import os
import numpy as np
import librosa
import fnmatch

import torch
from torch.utils.data import Dataset

from config import opt

DATAPATH = os.path.join(opt.data_root, 'THCHS-30')


class THCHS30(Dataset):
    def __init__(self, phase='train', sr=16000, dimension=72000):
        assert phase in ['train', 'test'], 'non-supported phase!'

        self.data_dir = None
        self.label_dir = None

        if phase == 'train':
            self.data_dir = os.path.join(DATAPATH, 'data_synthesized/train/noisy')
            self.label_dir = os.path.join(DATAPATH, 'data_synthesized/train/clean')
        elif phase == 'test':
            self.data_dir = os.path.join(DATAPATH, 'data_synthesized/test/noisy')
            self.label_dir = os.path.join(DATAPATH, 'data_synthesized/test/clean')

        self.sr = sr
        self.dim = dimension

        # use mapper in __getitem__
        # ensure each data find corresponding label
        self.mapper = {}

        # get label
        self.label_path = []
        for file in os.listdir(self.label_dir):
            if file.endswith('.wav'):
                self.mapper[file[:-4]] = len(self.label_path)
                self.label_path.append(os.path.join(self.label_dir, file))

        # get data path
        self.data_path = []
        for file in os.listdir(self.data_dir):
            if file.endswith('.wav'):
                self.data_path.append(os.path.join(self.data_dir, file))

        assert len(self.data_path) == len(self.label_path), 'data or label is corrupted!'

    def __getitem__(self, item):
        data, _ = librosa.load(self.data_path[item], sr=self.sr)
        data_name = os.path.basename(self.data_path[item])
        data_name = data_name[:data_name.rfind('_')]
        label, _ = librosa.load(self.label_path[self.mapper[data_name]], sr=self.sr)
        # 取 帧
        if len(data) > self.dim:
            max_audio_start = len(data) - self.dim
            audio_start = np.random.randint(0, max_audio_start)
            data = data[audio_start: audio_start + self.dim]
            label = label[audio_start:audio_start + self.dim]
        else:
            data = np.pad(data, (0, self.dim - len(data)), "constant")
            label = np.pad(label, (0, self.dim - len(label)), "constant")

        return data, label

    def __len__(self):
        return len(self.data_path)


if __name__ == '__main__':
    ds = THCHS30(phase='train')
    min_dim = 1e8
    max_dim = 0
    for i in range(0, len(ds)):
        data, label = ds[i]
        min_dim = min(min_dim, len(data))
        max_dim = max(max_dim, len(data))
    print('min dim=', min_dim)
    print('max dim=', max_dim)
    print('mid dim=', int((min_dim + max_dim) / 2))
    pass
