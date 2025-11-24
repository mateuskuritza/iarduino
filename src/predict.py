import cv2
import numpy as np
import tensorflow as tf
import sys
import os
import json
from collections import deque
from shared import IMG_SIZE, MODEL_PREFIX, MODEL_EXT, MODELS_FOLDER

CONFIDENCE_THRESHOLD = 0.70
BUFFER_SIZE = 10
STABLE_FRAMES_REQUIRED = 5


class PredictionBuffer:
    def __init__(self, buffer_size=BUFFER_SIZE, stable_frames=STABLE_FRAMES_REQUIRED):
        self.buffer = deque(maxlen=buffer_size)
        self.stable_frames = stable_frames
        self.current_label = None
        self.stable_count = 0

    def add_prediction(self, prediction):
        self.buffer.append(prediction)

    def get_smoothed_prediction(self):
        if len(self.buffer) == 0:
            return None
        return np.mean(self.buffer, axis=0)

    def get_stable_label(self, current_label):
        if current_label == self.current_label:
            self.stable_count += 1
        else:
            self.stable_count = 1
            if self.stable_count >= self.stable_frames:
                self.current_label = current_label

        return (
            self.current_label
            if self.stable_count >= self.stable_frames
            else self.current_label
        )


def load_labels_from_model(model_filename):
    label_file = model_filename.replace(MODEL_EXT, ".labels.json")
    label_path = os.path.join(MODELS_FOLDER, label_file)
    if not os.path.exists(label_path):
        print(f"[ERRO] Arquivo de labels '{label_file}' não encontrado.")
        sys.exit(1)
    with open(label_path, "r") as f:
        return json.load(f)


def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame_normalized = frame_resized / 255.0
    return np.expand_dims(frame_normalized, axis=0)


def draw_label_with_background(
    frame,
    text,
    position,
    font,
    font_scale,
    thickness,
    bg_color=(0, 0, 0),
    text_color=(255, 255, 255),
):
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size

    x, y = position
    cv2.rectangle(
        frame, (x - 5, y - text_h - 10), (x + text_w + 5, y + 10), bg_color, -1
    )

    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness)


def main(model_filename):
    labels = load_labels_from_model(model_filename)
    model_path = os.path.join(MODELS_FOLDER, model_filename)

    if not os.path.exists(model_path):
        print(f"[ERRO] Modelo '{model_path}' não encontrado.")
        sys.exit(1)

    print(f"[INFO] Carregando modelo '{model_path}'...")
    model = tf.keras.models.load_model(model_path)

    cap = cv2.VideoCapture(0)
    print("[INFO] Pressione 'q' ou clique no botão de fechar da janela para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERRO] Não foi possível acessar a câmera.")
            break

        input_data = preprocess_frame(frame)
        prediction = model.predict(input_data, verbose=0)[0]

        top_indices = prediction.argsort()[-3:][::-1]

        for i, idx in enumerate(top_indices):
            label = labels[idx]
            confidence = prediction[idx]
            text = f"{label}: {confidence * 100:.1f}%"
            draw_label_with_background(
                frame, text, (30, 60 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
            )

        cv2.imshow("Reconhecimento em tempo real", frame)

        if (
            cv2.getWindowProperty("Reconhecimento em tempo real", cv2.WND_PROP_VISIBLE)
            < 1
        ):
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def get_latest_model():
    models = [
        f
        for f in os.listdir(MODELS_FOLDER)
        if f.startswith(MODEL_PREFIX) and f.endswith(MODEL_EXT)
    ]
    if not models:
        return None
    models.sort(reverse=True)
    return models[0]


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        model_filename = sys.argv[1]
    else:
        model_filename = get_latest_model()
        if not model_filename:
            print("[ERRO] Nenhum modelo encontrado no diretório atual.")
            sys.exit(1)
        print(
            f"[INFO] Nenhum modelo informado. Usando o mais recente: {model_filename}"
        )

    main(model_filename)
