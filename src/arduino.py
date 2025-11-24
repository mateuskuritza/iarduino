import cv2
import numpy as np
import tensorflow as tf
import serial
import time
import sys
import os
import json
import argparse
from shared import MODEL_EXT, MODELS_FOLDER, IMG_SIZE


def load_labels_from_model(model_filename):
    label_file = model_filename.replace(MODEL_EXT, ".labels.json")
    label_path = os.path.join(MODELS_FOLDER, label_file)
    if not os.path.exists(label_path):
        print(f"[ERRO] Labels '{label_file}' não encontrados.")
        sys.exit(1)
    with open(label_path, "r") as f:
        return json.load(f)


def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame_normalized = frame_resized / 255.0
    return np.expand_dims(frame_normalized, axis=0)


def draw_label(frame, text, position):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    cv2.rectangle(
        frame,
        (x - 5, y - text_size[1] - 10),
        (x + text_size[0] + 5, y + 10),
        (0, 0, 0),
        -1,
    )
    cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness)


def main(model_filename, port_map, serial_port="/dev/ttyACM0"):
    labels = load_labels_from_model(model_filename)
    model_path = os.path.join(MODELS_FOLDER, model_filename)

    model = tf.keras.models.load_model(model_path)
    ser = serial.Serial(serial_port, 9600)
    time.sleep(2)  # Arduino must restart after serial init

    cap = cv2.VideoCapture(0)

    last_sent = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_data = preprocess_frame(frame)
        prediction = model.predict(input_data, verbose=0)[0]

        top_idx = prediction.argmax()
        label = labels[top_idx]
        confidence = prediction[top_idx]
        port = port_map.get(label)

        if port and label != last_sent:
            print("[ARDUINO] Comando enviado:", label, "-> Porta", f"LED:{port}")
            ser.write(f"LED:{port}\n".encode())
            last_sent = label

        draw_label(frame, f"{label}: {confidence*100:.1f}%", (30, 60))
        cv2.imshow("Detecção com Arduino", frame)

        if cv2.getWindowProperty("Detecção com Arduino", cv2.WND_PROP_VISIBLE) < 1:
            break
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    ser.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--port", required=True)

    args = parser.parse_args()

    with open(args.map, "r") as f:
        port_map = json.load(f)

    main(args.model, port_map, args.port)
