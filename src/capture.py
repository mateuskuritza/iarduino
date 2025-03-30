import cv2
import os
import shutil
import sys


def capture_images(label, num_images=50):
    save_path = f"data/captures/{label}"

    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0

    print(f"[INFO] Capturando imagens para: {label}")

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break

        img_name = os.path.join(save_path, f"{label}_{count}.jpg")
        cv2.imwrite(img_name, frame)
        count += 1

        cv2.imshow("Capturando...", frame)
        cv2.waitKey(100)  # tira uma foto a cada 100ms

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Captura finalizada. Total: {count}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python capture.py <nome_item>")
        sys.exit(1)

    label = sys.argv[1]
    capture_images(label)
