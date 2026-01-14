import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob
from fiftyone import ViewField as F

import tempfile
from pathlib import Path
import cv2


def fiftyone_similarity_node(
    images: dict,
    dataset_name: str = "train_augmented_ds",
    similarity_threshold: float = 0.92,
):
    """
    images: Dict[str, np.ndarray]  (RGB images)
    """

    # Create / overwrite dataset
    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)

    fo_dataset = fo.Dataset(dataset_name)

    tmp_dir = Path(tempfile.mkdtemp())

    samples = []

    for name, img in images.items():
        if img is None:
            continue

        # OPTIONAL: infer class from filename
        class_name = name.split("_")[0]

        out_path = tmp_dir / f"{name}.jpg"
        cv2.imwrite(
            str(out_path),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        )

        samples.append(
            fo.Sample(
                filepath=str(out_path),
                ground_truth=fo.Classification(label=class_name),
            )
        )

    fo_dataset.add_samples(samples)

    # Load model
    model = foz.load_zoo_model(
        "mobilenet-v2-imagenet-torch",
        embeddings=True,
    )

    # Compute embeddings
    fo_dataset.compute_embeddings(
        model,
        embeddings_field="embeddings",
    )

    # Compute similarity
    fob.compute_similarity(
        fo_dataset,
        embeddings="embeddings",
        metric="cosine",
    )

    # Filter similar samples
    view = fo_dataset.match(F("similarity") > similarity_threshold)

    # Return metadata (Kedro-safe)
    return {
        "dataset_name": dataset_name,
        "num_samples": len(fo_dataset),
        "num_similar": len(view),
    }