from pathlib import Path
import shutil
import logging
import numpy as np


def get_logger():
    logger = logging.getLogger("datavault")

    # so we don’t duplicate if reloaded in some environments
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            "%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def _compute_path_size(p: Path) -> int:
    """Return total size in bytes of a file or directory (recursively)."""
    if not p.exists():
        return 0
    if p.is_file():
        try:
            return p.stat().st_size
        except OSError:
            return 0
    total = 0
    # Use rglob; ignore files that raise errors (e.g., broken symlinks)
    for child in p.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                pass
    return total


def _format_size(num_bytes: int) -> str:
    """Format a byte count in a human-readable way."""
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def delete_path(path: Path | str, confirm: bool = True):
    p = Path(path)

    if not p.exists():
        logger.error(f"Path does not exist: {p}")
        return

    size_bytes = _compute_path_size(p)
    human_size = _format_size(size_bytes)

    if confirm:
        try:
            answer = (
                input(f"Delete path '{p}' (size {human_size}). confirm: y/n?")
                .strip()
                .lower()
            )
        except EOFError:
            # Non-interactive environment: treat as cancelled
            answer = ""
        if answer != "y":
            logger.info("Deletion aborted by user.")
            return
    try:
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
        else:
            # Covers regular files and symlinks
            p.unlink(missing_ok=True)
        logger.info(f"Deleted path: {p}, size {human_size}.")
    except Exception as e:
        logger.error(f"Failed to delete path {p}: {e}")
        raise


def load_np_array(arr: np.ndarray | str | Path) -> np.ndarray:
    if isinstance(arr, str) or isinstance(arr, Path):
        return np.load(arr)
    return arr


logger = get_logger()

if __name__ == "__main__":
    target = "/tmp/test_file.txt"
    with Path(target).open("w") as f:
        for _ in range(1000000):
            f.write("Hi there !")
    delete_path(target, True)
    target = "/tmp/test_file.txt"
    with Path(target).open("w") as f:
        for _ in range(1000000):
            f.write("Hi there !")
    delete_path(target, False)
    delete_path(target, False)
