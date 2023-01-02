import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List

class CustomDataset(Dataset):
    def __init__(self, df : pd.DataFrame, src_cols : List, tar_cols : List, src_len : int, tar_len : int, stride : int = 1):
        self.data = df
        self.src_cols = src_cols
        self.tar_cols = tar_cols
        self.src_len = src_len
        self.tar_len = tar_len

        self.source_indices = []
        self.target_indices = []

        self.stride = stride

        self._preprocessing()
        self._generate_indices()

    def _preprocessing(self):
        # remove NaN values
        pass

    def _generate_indices(self):
        
        for idx in range(0, len(self.data), self.stride):
            if idx + self.tar_len + self.src_len >= len(self.data) - 1:
                break
            else:
                self.source_indices.append(idx)
                self.target_indices.append(idx + self.src_len)

    def __len__(self):
        return len(self.source_indices)
    
    def __getitem__(self, idx : int):

        src_indx = self.source_indices[idx]
        tar_indx = self.target_indices[idx]

        data = self.data[src_indx : src_indx + self.src_len][self.src_cols].values
        target = self.data[tar_indx : tar_indx + self.tar_len][self.tar_cols].values

        data = torch.from_numpy(data).float()
        target = torch.from_numpy(target).float()
    
        return data, target