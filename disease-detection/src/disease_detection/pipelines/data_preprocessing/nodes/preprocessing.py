from pathlib import Path
import cv2

def preprocessing_node(data):
    raw_dir = Path(data["raw_image_dir"])
    image_size = data["image_size"]

    partitions = {}

    for class_name in raw_dir.iterdir():
        if not class_name.is_dir():
            continue

        for img_path in class_name.iterdir():
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (image_size, image_size))

            key = f"{class_name.name}/{img_path.name}"
            partitions[key] = img

    return partitions