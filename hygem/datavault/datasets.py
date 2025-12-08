from typing import Optional, Tuple, Callable
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from datavault.utils import load_np_array


class ArrayDataset(Dataset):
    def __init__(
        self,
        features_arr: np.ndarray | str | Path,
        labels_arr: Optional[np.ndarray | str | Path] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.feats = load_np_array(features_arr)
        if labels_arr is None:
            self.labels = None
        else:
            self.labels = load_np_array(labels_arr)
            assert self.labels.shape[0] == self.feats.shape[0]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.feats.shape[0]

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray] | np.ndarray:
        feat = self.feats[index]
        if self.transform is not None:
            feat = self.transform(feat)

        if self.labels is None:
            return feat
        label = self.labels[index]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return feat, label
