import torch
from torch.utils.data import Dataset, DataLoader
import random


class MFDataset(Dataset):
    def __init__(self, args, data):
        self.args = args
        self.data = data
        

    def __getitem__(self, index):
        cur_idx = self.data.iloc[index]
        cur_tensors = (
            torch.tensor(cur_idx[0], dtype=torch.long),
            torch.tensor(cur_idx[1], dtype=torch.long),
            torch.tensor(cur_idx[2], dtype=torch.float)
        )
        return cur_tensors


    def __len__(self):
        return self.data.shape[0]
