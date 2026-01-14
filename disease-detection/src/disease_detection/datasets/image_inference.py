from kedro.io import AbstractDataset
from pathlib import Path
from PIL import Image


class ImageInferenceDataset(AbstractDataset):
    def __init__(self, filepath: str, extensions=(".jpg", ".jpeg", ".png")):
        self._filepath = Path(filepath)
        self._extensions = extensions

    def _load(self):
        images = {}
        for img_path in sorted(self._filepath.iterdir()):
            if img_path.suffix.lower() in self._extensions:
                with Image.open(img_path) as img:
                    images[img_path.name] = img.convert("RGB")
        return images

    def _save(self, data):
        raise NotImplementedError("Read-only dataset")

    def _describe(self):
        return {"filepath": str(self._filepath)}
