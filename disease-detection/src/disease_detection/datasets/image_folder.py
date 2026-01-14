from kedro.io import AbstractDataset
import cv2
import numpy as np
from pathlib import Path

class ImageFolderDataset(AbstractDataset):
    """Stores a single image as a file on disk."""

    def __init__(self, filepath: str):
        self._filepath = Path(filepath)

    def _load(self) -> np.ndarray:
        img = cv2.imread(str(self._filepath))
        if img is None:
            raise IOError(f"Failed to read image {self._filepath}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _save(self, data: np.ndarray) -> None:
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        img_bgr = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(self._filepath), img_bgr)

    def _describe(self):
        return {"filepath": str(self._filepath)}