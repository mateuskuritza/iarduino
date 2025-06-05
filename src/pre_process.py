import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from shared import PROCESSED_DATA_DIR, CAPTURES_DIR, IMG_SIZE


def create_processed_folder_structure(labels):
    for label in labels:
        path = os.path.join(PROCESSED_DATA_DIR, label)
        os.makedirs(path, exist_ok=True)


def load_and_process_images():
    X = []
    y = []
    labels = os.listdir(CAPTURES_DIR)

    create_processed_folder_structure(labels)

    for label in labels:
        class_path = os.path.join(CAPTURES_DIR, label)
        save_path = os.path.join(PROCESSED_DATA_DIR, label)
        if not os.path.isdir(class_path):
            continue

        for file in os.listdir(class_path):
            img_path = os.path.join(class_path, file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            processed_img_path = os.path.join(save_path, file)
            cv2.imwrite(processed_img_path, img_resized)

            img_normalized = img_resized / 255.0
            X.append(img_normalized)
            y.append(label)

            img_flipped = cv2.flip(img_resized, 1)
            X.append(img_flipped / 255.0)
            y.append(label)

            h, w = img_resized.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), 10, 1)
            img_rotated = cv2.warpAffine(
                img_resized, M, (w, h), borderMode=cv2.BORDER_REFLECT
            )
            X.append(img_rotated / 255.0)
            y.append(label)

    return np.array(X), np.array(y), labels


def encode_labels(y, label_names):
    label_to_index = {name: i for i, name in enumerate(label_names)}
    encoded = np.array([label_to_index[label] for label in y])
    one_hot = np.eye(len(label_names))[encoded]
    return one_hot


def preprocess():
    print("[INFO] Carregando e processando imagens...")
    X, y, label_names = load_and_process_images()

    print("[INFO] Codificando rótulos...")
    y_encoded = encode_labels(y, label_names)

    print("[INFO] Dividindo dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[INFO] Total de imagens processadas: {len(X)}")
    print(f"[INFO] Labels: {label_names}")
    return X_train, X_test, y_train, y_test, label_names


if __name__ == "__main__":
    preprocess()
