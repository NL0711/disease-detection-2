import os
from pathlib import Path
import fiftyone as fo
import fiftyone.brain as fob
from fiftyone import ViewField as F
import mlflow

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def fiftyone_similarity_node(
    train_augmented_ds: dict,
    embeddings: dict,
    dataset_name: str = "train_augmented_ds",
):
    samples = []
    skipped = 0

    for img_path in train_augmented_ds.values():
        p = Path(img_path)

        if (
            p.exists()
            and p.is_file()
            and p.suffix.lower() in VALID_EXTS
        ):
            samples.append(fo.Sample(filepath=str(p)))
        else:
            skipped += 1

    if not samples:
        raise RuntimeError("No valid image files found for FiftyOne similarity")

    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)

    fo_dataset = fo.Dataset(dataset_name)
    fo_dataset.add_samples(samples)

    fo_dataset.media_type = "image"

    fob.compute_similarity(
        fo_dataset,
        model=embeddings["model"],
        brain_key="similarity",
    )

    view = fo_dataset.match(
        F("similarity") > embeddings["similarity_threshold"]
    )

    mlflow.log_metrics({
        "deduplication/total_images": len(fo_dataset),
        "deduplication/similar_images": len(view),
        "deduplication/unique_images": len(fo_dataset) - len(view),
        "deduplication/skipped_invalid": skipped,
    })

    duplicate_ids = view.values("id")

    return fo_dataset.exclude(duplicate_ids).to_dicts()