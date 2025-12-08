from typing import Optional, Union
import os
import io
import re
import urllib
import urllib.error
import urllib.request
from urllib.parse import urlparse
from tqdm import tqdm

from pathlib import Path
from datavault.utils import get_logger
from torchvision.datasets.utils import (
    check_integrity,
    download_file_from_google_drive,
    extract_archive,
    _urlretrieve,
    _get_google_drive_file_id,
    download_and_extract_archive,
    download_url,
)
