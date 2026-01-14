from torchvision.datasets import ImageFolder
from sklearn.model_selection import StratifiedKFold
import numpy as np
import mlflow

def split_data_node(preprocessed_ds, data, k_fold):
    keys = sorted(preprocessed_ds.keys())

    # Extract class labels from keys: class_name/image.jpg
    labels = np.array([key.split("/")[0] for key in keys])

    class_names = sorted(set(labels))

    indices = np.arange(len(keys))

    skf = StratifiedKFold(
        n_splits=k_fold["n_folds"],
        shuffle=True,
        random_state=k_fold["seed"],
    )

    train_folds = []
    val_folds = []

    for train_idx, val_idx in skf.split(indices, labels):
        train_folds.append(train_idx.tolist())
        val_folds.append(val_idx.tolist())

    mlflow.log_params({
        "image_size": data["image_size"],   
        "n_folds": k_fold["n_folds"],
        "seed": k_fold["seed"],
        "num_classes": len(class_names),
    })
    mlflow.log_text("\n".join(class_names), artifact_file="classes.txt")

    return train_folds, val_folds, class_names