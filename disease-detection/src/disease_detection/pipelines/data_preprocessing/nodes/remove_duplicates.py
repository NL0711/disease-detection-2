import fiftyone as fo
import fiftyone.brain as fob
from fiftyone import ViewField as F

from pathlib import Path
import mlflow

def fiftyone_similarity_node(
    train_augmented_ds: dict,
    embeddings,
    dataset_name: str = "train_augmented_ds",
):
    
    dir_path = Path(train_augmented_ds)
    samples = []

    for class_name in dir_path.iterdir():
        if not class_name.is_dir():
            continue

        for img_path in class_name.iterdir():
            samples.append(
                fo.Sample(
                    filepath=str(img_path),
                    ground_truth=fo.Classification(label=class_name),
                )
            )

    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)

    fo_dataset = fo.Dataset(dataset_name)
    fo_dataset.add_samples(samples)

    fo_dataset.compute_embeddings(
        embeddings["model"],
        embeddings_field="embeddings",
    )

    fob.compute_similarity(
        fo_dataset,
        embeddings="embeddings",
        metric="cosine",
    )

    view = fo_dataset.match(F("similarity") > embeddings["similarity_threshold"])

    mlflow.log_metric({
        "deduplication/total_images": len(fo_dataset),
        "deduplication/similar_images": len(view),
        "deduplication/unique_images": len(fo_dataset) - len(view),
    })


    return fo_dataset.exclude(samples).to_dicts()