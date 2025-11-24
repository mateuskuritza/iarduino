import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from shared import PROCESSED_DATA_DIR, CAPTURES_DIR, IMG_SIZE
import random


def create_processed_folder_structure(labels):
    for label in labels:
        path = os.path.join(PROCESSED_DATA_DIR, label)
        os.makedirs(path, exist_ok=True)


def apply_augmentation(img):
    augmented_images = []

    augmented_images.append(img.copy())
    augmented_images.append(cv2.flip(img, 1))

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 10, 1)
    img_rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    augmented_images.append(img_rotated)

    return augmented_images


def load_and_process_images():
    X = []
    y = []
    labels = [
        d
        for d in os.listdir(CAPTURES_DIR)
        if os.path.isdir(os.path.join(CAPTURES_DIR, d))
    ]

    create_processed_folder_structure(labels)

    print(f"[INFO] Processando imagens com augmentation...")

    for label in labels:
        class_path = os.path.join(CAPTURES_DIR, label)
        save_path = os.path.join(PROCESSED_DATA_DIR, label)

        image_count = 0
        for file in os.listdir(class_path):
            img_path = os.path.join(class_path, file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            processed_img_path = os.path.join(save_path, file)
            cv2.imwrite(processed_img_path, img_resized)

            augmented_images = apply_augmentation(img_resized)

            for aug_img in augmented_images:
                img_normalized = aug_img / 255.0
                X.append(img_normalized)
                y.append(label)

            image_count += 1

        print(
            f"[INFO] Classe '{label}': {image_count} imagens → {image_count * len(augmented_images)} após augmentação"
        )

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
