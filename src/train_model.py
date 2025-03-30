import datetime
import tensorflow as tf
from tensorflow.keras import layers, models
from pre_process import preprocess
import json


def build_model(input_shape, num_classes):
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def main():
    print("[INFO] Iniciando pré-processamento com todas as classes disponíveis...")
    X_train, X_test, y_train, y_test, labels = preprocess()

    print(f"[INFO] Labels encontradas: {labels}")
    print("[INFO] Construindo modelo...")
    model = build_model(X_train.shape[1:], len(labels))

    print("[INFO] Iniciando treino...")
    model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.1)

    print("[INFO] Avaliando modelo...")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[RESULT] Acurácia no teste: {acc:.2%}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"models/model_{timestamp}.keras"

    print(f"[INFO] Salvando modelo como {save_path}...")
    model.save(save_path)

    label_file = f"models/model_{timestamp}.labels.json"
    with open(label_file, "w") as f:
        json.dump(labels, f)

    print(f"[INFO] Labels salvos em {label_file}")

    print("[INFO] Concluído!")


if __name__ == "__main__":
    main()
