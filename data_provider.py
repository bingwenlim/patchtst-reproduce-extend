import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int,
                 label_len: int, time_marks: np.ndarray):
        """
        data: numpy array of shape [T, C]
        seq_len: input sequence length
        pred_len: prediction horizon
        label_len: overlap between encoder and decoder (for Autoformer)
        time_marks: [T, D] array of temporal features
        """
        if len(data) < seq_len + pred_len:
            raise ValueError(
                f"Data length ({len(data)}) is too short for "
                f"seq_len={seq_len} and pred_len={pred_len}."
            )

        self.data = np.asarray(data, dtype=np.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.time_marks = np.asarray(time_marks, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]

        x_mark = self.time_marks[idx : idx + self.seq_len]
        dec_start = idx + self.seq_len - self.label_len
        y_mark = self.time_marks[dec_start : dec_start + self.label_len + self.pred_len]

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(x_mark),
            torch.from_numpy(y_mark),
        )


def get_dataset(name, seq_len, pred_len, label_len):
    """Load and split a dataset by name.

    Returns: (datasets_dict, enc_in) where datasets_dict has keys
    "train", "val", "test" each containing a TimeSeriesDataset.
    """
    if name == "ETTh1":
        return _get_ett_datasets(
            file_path="data/ett.csv",
            seq_len=seq_len, pred_len=pred_len, label_len=label_len,
        )
    raise ValueError(f"Unknown dataset: {name}")


def time_features(dates: pd.Series) -> np.ndarray:
    """Extract temporal features normalized to [-0.5, 0.5].
    Returns: np.ndarray of shape [T, 4] (hour, day_of_week, day_of_month, day_of_year)."""
    dt = pd.to_datetime(dates).dt
    features = np.column_stack([
        dt.hour / 23.0 - 0.5,
        dt.dayofweek / 6.0 - 0.5,
        (dt.day - 1) / 30.0 - 0.5,
        (dt.dayofyear - 1) / 365.0 - 0.5,
    ])
    return features.astype(np.float32)


def _get_ett_datasets(file_path, seq_len, pred_len, label_len):
    """Load ETTh1 CSV, split by standard 12/4/4 month borders, return TimeSeriesDatasets."""
    df = pd.read_csv(file_path)

    # Extract numeric data and time marks
    raw_data = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
    marks = time_features(df["date"])

    # Standard ETTh1 borders: 8640 / 2880 / 2880
    train_end = 12 * 30 * 24          # 8640
    val_end = train_end + 4 * 30 * 24  # 11520
    test_end = val_end + 4 * 30 * 24   # 14400

    # Val/test start shifted back by seq_len for lookback overlap
    border1s = [0, train_end - seq_len, val_end - seq_len]
    border2s = [train_end, val_end, test_end]

    # Standardize using train portion only
    train_portion = raw_data[:train_end]
    mean = train_portion.mean(axis=0, keepdims=True)
    std = train_portion.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    scaled = (raw_data - mean) / std

    enc_in = raw_data.shape[1]

    datasets = {}
    for i, name in enumerate(["train", "val", "test"]):
        b1, b2 = border1s[i], border2s[i]
        datasets[name] = TimeSeriesDataset(
            scaled[b1:b2], seq_len, pred_len, label_len, marks[b1:b2],
        )

    return datasets, enc_in
