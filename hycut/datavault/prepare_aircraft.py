from torchvision.datasets.vision import VisionDataset
from typing import Union, Callable, Optional
from pathlib import Path

import argparse
import os
import shutil

from tqdm import tqdm

from datavault.data_utils import download_and_extract_archive


if __name__ == "__main__":
    # import os
    # ds = FGVCAircraft(os.environ["DATASETS"]+"./aircraft/")
    from torchvision.datasets import FGVCAircraft
    from torchvision.transforms import ToTensor

    ds = FGVCAircraft(
        os.environ["DATASETS"] + "./aircraft/", transform=ToTensor()
    )
    for i in range(4):
        a, b = ds[i]
        print(a.shape, b)
