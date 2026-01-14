from torchvision import transforms
import numpy as np
import mlflow

def get_augmentation_transform(image_size):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
    ])

def augmentation_node(train_preprocessed_ds, data):
    augmented = {}
    image_size = data["image_size"]
    transform = get_augmentation_transform(image_size)

    original_count = 0
    augmented_count = 0

    for key, loader in train_preprocessed_ds.items():
        img = loader()
        if img is None:
            continue

        if not isinstance(img, np.ndarray):
            raise TypeError(f"{key}: expected np.ndarray, got {type(img)}")

        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        # keep original
        augmented[key] = img
        original_count += 1

        # augmentation
        pil_img = transforms.ToPILImage()(img)
        aug_img = transform(pil_img)
        aug_img = np.array(aug_img)

        augmented[f"{key}_aug"] = aug_img
        augmented_count += 1

    final_count = len(augmented)

    mlflow.log_metrics({
        "augmentation/original_images": original_count,
        "augmentation/augmented_images": augmented_count,
        "augmentation/final_image_count": final_count,
        "augmentation/ratio": augmented_count / max(original_count, 1),
    })

    return augmented
