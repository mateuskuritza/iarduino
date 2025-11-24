import datetime
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from pre_process import preprocess
import json
from shared import MODEL_EXT, MODEL_PREFIX


def build_model(input_shape, num_classes):
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.5),
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
    print(f"[INFO] Total de amostras de treino: {len(X_train)}")
    print(f"[INFO] Total de amostras de teste: {len(X_test)}")

    print("[INFO] Construindo modelo...")
    model = build_model(X_train.shape[1:], len(labels))
    model.summary()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = f"models/{MODEL_PREFIX}{timestamp}{MODEL_EXT}"

    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True, verbose=1
        )
    ]

    print("[INFO] Iniciando treino...")

    history = model.fit(
        X_train,
        y_train,
        epochs=25,
        batch_size=16,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    print("\n[INFO] Avaliando modelo no conjunto de teste...")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[RESULT] Acurácia no teste: {acc:.2%}")

    model.save(checkpoint_path)

    label_file = f"models/{MODEL_PREFIX}{timestamp}.labels.json"
    with open(label_file, "w") as f:
        json.dump(labels, f)

    print(f"[INFO] Labels salvos em: {label_file}")
    print("[INFO] Concluído!")


if __name__ == "__main__":
    main()
