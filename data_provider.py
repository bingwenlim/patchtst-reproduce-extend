import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        """
        data: numpy array of shape [T, C]
        """
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return x, y


def load_csv_dataset(file_path, target_cols=None):
    df = pd.read_csv(file_path)

    # keep only numeric columns if target_cols not provided
    if target_cols is None:
        df = df.select_dtypes(include=[np.number])
    else:
        df = df[target_cols]

    return df.values


def standardize_train_val_test(train_data, val_data, test_data):
    mean = train_data.mean(axis=0, keepdims=True)
    std = train_data.std(axis=0, keepdims=True)
    std[std == 0] = 1.0

    train_scaled = (train_data - mean) / std
    val_scaled = (val_data - mean) / std
    test_scaled = (test_data - mean) / std

    return train_scaled, val_scaled, test_scaled, mean, std