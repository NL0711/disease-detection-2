from unittest import loader
import cv2
import numpy as np

def check_blur_and_noise(img, params):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    gray = img_yuv[:, :, 0]

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    score = lap.var()

    if score < params["blur"]["reject_threshold"]:
        return None, "blur"

    elif score <= params["blur"]["processing_threshold"]:
        gray = cv2.fastNlMeansDenoising(gray, None, 3, 7, 21)
        img_sharpened = cv2.filter2D(
            gray, -1,
            np.array([[0, -1, 0],
                      [-1, 5, -1],
                      [0, -1, 0]])
        )
        img_yuv[:, :, 0] = img_sharpened
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB), None

    else:
        img_blur = cv2.GaussianBlur(img, (5, 5), 2.0)
        noise_score = np.std(img.astype(np.float32) - img_blur.astype(np.float32))

        if noise_score > params["noise_max"]:
            return None, "noise"

        return img, None

def check_contrast(img, params):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    gray = img_yuv[:, :, 0]
    contrast = gray.std()

    if contrast < params["contrast"]["min"]:
        clipLimit = 2.5
    elif contrast < params["contrast"]["mid"]:
        clipLimit = 1.8
    elif contrast > params["contrast"]["max"]:
        return None, "contrast"
    else:
        return img, None

    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
    img_yuv[:, :, 0] = clahe.apply(gray)
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB), None

def check_brightness(img, params):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    gray = img_yuv[:, :, 0]
    brightness = gray.mean()

    if brightness < params["brightness"]["min"]:
        return None, "brightness"

    elif brightness < params["brightness"]["mid"]:
        inv = 1.0 / 0.8
        table = (np.arange(256) / 255.0) ** inv * 255
        img_yuv[:, :, 0] = cv2.LUT(gray, table.astype(np.uint8))
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB), None

    elif brightness > params["brightness"]["max"]:
        inv = 1.0 / 1.3
        table = (np.arange(256) / 255.0) ** inv * 255
        img_yuv[:, :, 0] = cv2.LUT(gray, table.astype(np.uint8))
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB), None

    return img, None

def preprocess_train_node(preprocessed_ds, train_folds, preprocess_params):
    accepted = {}
    rejected = {}

    # Stable ordering is mandatory
    keys = sorted(preprocessed_ds.keys())

    for fold_idx, train_indices in enumerate(train_folds):
        for idx in train_indices:
            partition_key = keys[idx]
            loader = preprocessed_ds[partition_key]
            img = loader() # np.ndarray (RGB)

            if img.dtype != np.uint8:
                img = img.astype(np.uint8)

            if img is None:
                continue

            img, reason = check_blur_and_noise(img, preprocess_params)
            if img is None:
                rejected[f"{reason}/{partition_key}"] = preprocessed_ds[partition_key]
                continue

            img, reason = check_brightness(img, preprocess_params)
            if img is None:
                rejected[f"{reason}/{partition_key}"] = preprocessed_ds[partition_key]
                continue

            img, reason = check_contrast(img, preprocess_params)
            if img is None:
                rejected[f"{reason}/{partition_key}"] = preprocessed_ds[partition_key]
                continue

            class_name, img_name = partition_key.split("/", 1)
            accepted_key = f"fold_{fold_idx}/{class_name}/{img_name}"

            accepted[accepted_key] = img

    return accepted, rejected