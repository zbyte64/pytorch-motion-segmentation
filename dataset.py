import os
import torch
import numpy as np
import random
import h5py
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class TwoFrameVideoDataset(Dataset):
    def __init__(self, path):
        super(TwoFrameVideoDataset).__init__()
        self.f = h5py.File(path, 'r')
        self.ds = f['X']

    def __len__(self):
        return self.ds.shape[0] - 1

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.ds[idx] / 255.),
            torch.from_numpy(self.ds[idx+1] / 255.)
        )


def sample_folder(folder_path):
    videos = list()
    for fname in os.listdir(folder_path):
        if fname.endswith('.h5'):
            path = os.path.join(folder_path, fname)
            ds = TwoFrameVideoDataset(path)
            videos.append(ds)
    return ConcatDataset(videos)
