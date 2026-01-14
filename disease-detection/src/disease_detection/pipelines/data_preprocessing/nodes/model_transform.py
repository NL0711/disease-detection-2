from torchvision import transforms
import torch
import cv2
import numpy as np

def get_model_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

def model_transform_node(
    train_augmented_ds,
    preprocessed_ds,
    train_folds,
    val_folds,
):
    transform = get_model_transform()

    keys = sorted(preprocessed_ds.keys())

    class_names = sorted({k.split("/")[0] for k in keys})
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}

    train_folds_tensors = []
    val_folds_tensors = []

    for fold_idx, train_indices in enumerate(train_folds):
        X_train, y_train = [], []

        for idx in train_indices:
            key = keys[idx]
            fold_key = f"fold_{fold_idx}/{key}"

            loader = train_augmented_ds.get(fold_key)
            if loader is None:
                continue

            img = loader()  
            if img is None:
                continue

            if img.dtype != np.uint8:
                img = img.astype(np.uint8)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = transform(img)

            label = class_to_idx[key.split("/")[0]]

            X_train.append(tensor)
            y_train.append(label)

        train_folds_tensors.append(
            (torch.stack(X_train), torch.tensor(y_train, dtype=torch.long))
        )

    for fold_idx, val_indices in enumerate(val_folds):
        X_val, y_val = [], []

        for idx in val_indices:
            key = keys[idx]

            loader = preprocessed_ds[key]   
            img = loader()                  

            if img is None:
                continue

            if img.dtype != np.uint8:
                img = img.astype(np.uint8)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = transform(img)

            label = class_to_idx[key.split("/")[0]]

            X_val.append(tensor)
            y_val.append(label)

        val_folds_tensors.append(
            (torch.stack(X_val), torch.tensor(y_val, dtype=torch.long))
        )

    return train_folds_tensors, val_folds_tensors