import os
import glob
import pandas as pd
from predict import get_latest_model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json
from shared import IMG_SIZE as SHARED_IMG_SIZE

# Caminhos
CAPTURES_DIR = "data/captures"
MODEL_NAME = get_latest_model()
MODEL_PATH = os.path.join("models", MODEL_NAME)
LABELS_PATH = MODEL_PATH.replace(".keras", ".labels.json")

# Carrega modelo e labels
model = load_model(MODEL_PATH)
with open(LABELS_PATH, "r") as f:
    labels = json.load(f)
    idx2label = {i: label for i, label in enumerate(labels)}

IMG_SIZE = (SHARED_IMG_SIZE, SHARED_IMG_SIZE)


# Função para prever uma imagem
def predict_img(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0
    pred = model.predict(x)
    return idx2label[np.argmax(pred)]


# Avaliação
resultados = []
for item in os.listdir(CAPTURES_DIR):
    pasta = os.path.join(CAPTURES_DIR, item)
    if not os.path.isdir(pasta):
        continue
    imagens = glob.glob(os.path.join(pasta, "*"))
    total = len(imagens)
    acertos = 0
    for img_path in imagens:
        pred = predict_img(img_path)
        if pred == item:
            acertos += 1
    taxa = acertos / total if total > 0 else 0
    resultados.append(
        {"item": item, "total": total, "acertos": acertos, "taxa_acerto": taxa}
    )

# Salva resultados em CSV
df = pd.DataFrame(resultados)
df.to_csv("acuracia.csv", index=False)

# Gera gráfico
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.bar(df["item"], df["taxa_acerto"], color="#4a4e69")
plt.ylabel("Taxa de Acerto")
plt.xlabel("Item")
plt.title("Taxa de Acerto por Item")
plt.ylim(0, 1)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("acuracia.png")
plt.show()
print(df)
