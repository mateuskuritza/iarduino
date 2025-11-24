import cv2
import os
import shutil
import sys
import time


def draw_progress(frame, label, count, total):
    height, width = frame.shape[:2]

    progress = count / total
    bar_width = width - 40
    bar_height = 30

    cv2.rectangle(frame, (20, 20), (20 + bar_width, 20 + bar_height), (50, 50, 50), -1)
    cv2.rectangle(
        frame,
        (20, 20),
        (20 + int(bar_width * progress), 20 + bar_height),
        (0, 255, 0),
        -1,
    )

    progress_text = f"{label}: {count}/{total} ({progress * 100:.0f}%)"
    cv2.putText(
        frame,
        progress_text,
        (30, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    return frame


def capture_images(label, num_images=200):
    save_path = f"data/captures/{label}"

    if os.path.exists(save_path):
        print(f"[AVISO] Já existem imagens para '{label}'")
        response = input("Deseja substituir as imagens existentes? (s/n): ")
        if response.lower() == "s":
            shutil.rmtree(save_path)
        else:
            print("[INFO] Captura cancelada.")
            return

    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0

    print(f"[INFO] Capturando {num_images} imagens para: {label}")
    print("[INFO] Iniciando em 2 segundos...")
    time.sleep(2)

    last_capture_time = time.time()
    capture_interval = 0.08

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("[ERRO] Não foi possível acessar a câmera.")
            break

        display_frame = draw_progress(frame.copy(), label, count, num_images)

        cv2.imshow("Captura", display_frame)

        current_time = time.time()
        if current_time - last_capture_time >= capture_interval:
            img_name = os.path.join(save_path, f"{label}_{count:04d}.jpg")
            cv2.imwrite(img_name, frame)
            count += 1
            last_capture_time = current_time

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print(f"\n[INFO] Captura interrompida pelo usuário.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Captura finalizada. Total: {count} imagens salvas em '{save_path}'")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python capture.py <nome_item> [num_imagens]")
        print("\nExemplos:")
        print("  python capture.py banana")
        print("  python capture.py maca 150")
        sys.exit(1)

    label = sys.argv[1]
    num_images = 200

    if len(sys.argv) >= 3:
        try:
            num_images = int(sys.argv[2])
        except ValueError:
            print("[ERRO] Número de imagens inválido.")
            sys.exit(1)

    capture_images(label, num_images)
