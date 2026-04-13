import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int):
        """
        data: numpy array of shape [T, C]
        seq_len: input sequence length
        pred_len: prediction horizon
        """
        if len(data) < seq_len + pred_len:
            raise ValueError(
                f"Data length ({len(data)}) is too short for "
                f"seq_len={seq_len} and pred_len={pred_len}."
            )

        self.data = np.asarray(data, dtype=np.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]

        return torch.from_numpy(x), torch.from_numpy(y)


def load_csv_dataset(file_path: str, target_cols=None) -> np.ndarray:
    df = pd.read_csv(file_path)

    if target_cols is None:
        df = df.select_dtypes(include=[np.number])
    else:
        missing_cols = [col for col in target_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in dataset: {missing_cols}")
        df = df[target_cols]

    if df.empty:
        raise ValueError(
            "No usable columns found in the dataset. "
            "Check whether the CSV contains numeric columns or valid target_cols."
        )

    return df.to_numpy(dtype=np.float32)


def standardize_train_val_test(train_data, val_data, test_data):
    train_data = np.asarray(train_data, dtype=np.float32)
    val_data = np.asarray(val_data, dtype=np.float32)
    test_data = np.asarray(test_data, dtype=np.float32)

    mean = train_data.mean(axis=0, keepdims=True)
    std = train_data.std(axis=0, keepdims=True)
    std[std == 0] = 1.0

    train_scaled = (train_data - mean) / std
    val_scaled = (val_data - mean) / std
    test_scaled = (test_data - mean) / std

    return train_scaled, val_scaled, test_scaled, mean, std