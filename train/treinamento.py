# ==============================
# 📌 1. Instalar o Ultralytics - V_50 épocas
# ==============================
# Instalar o Ultralytics
!pip install ultralytics
import os
from ultralytics import YOLO
import torch

# Verificar se o dataset está presente
dataset_path = "C:/YV8SKFRAME/dataset/train/images"
if os.path.exists(dataset_path) and len(os.listdir(dataset_path)) > 0:
    print("✅ Dataset C:/YV8SKFRAME/dataset encontrado e pronto para treino!")
else:
    print("❌ Dataset NÃO encontrado ou vazio! Verifique o caminho e a estrutura.")

# ==============================
# 📌 2. Treinar o Modelo YOLOv8 para detectar propagandas ("Foguetes") - V_50 épocas
# ==============================
# Carregar o modelo pré-treinado YOLOv8 (ex.: yolov8n.pt)
model = YOLO("yolov8n.pt")

# Iniciar o treinamento
results = model.train(
    data="C:/YV8SKFRAME/data.yaml",  # Caminho absoluto para o arquivo YAML do dataset
    epochs=50,
    imgsz=(640, 360),    # Redimensiona as imagens para 640x360 (ajuste conforme necessário)
    batch=16,
   project="C:YV8SKFRAME/dataset/training_results",  # Diretório onde os resultados serão salvos
    name="yolov8n_Soccer-Key-Frames"  # Nome do experimento/modelo treinado
)

# ==============================
# 📌 3. Avaliar no Conjunto de Teste - V_50 épocas
# ==============================
# Realizar avaliação no conjunto de teste
metrics = model.val(data="C:/YV8SKFRAME/data.yaml", split="test")
print("Métricas de avaliação:")
print(metrics)

# Testar com uma imagem do conjunto de teste (ajuste o caminho da imagem conforme necessário)
model("C:/YV8SKFRAME/dataset/test/images/CFOG28T_01-28-13_mp4-0195_jpg.rf.28429ae826fb7040d2a63283c789e2e6.jpg", save=True)

# ==============================
# 📌 4. Resultados do Treinamento YOLOv8 - V_50 épocas
# ==============================
# Resultados do Treinamento YOLOv8
%matplotlib inline
import matplotlib.pyplot as plt
import cv2

# Supondo que o YOLOv8 salvou um gráfico de resultados em "results.png"
results_image_path = "C:YV8SKFRAME/dataset/training_results/yolov8n_Soccer-Key-Frames13/results.png"
if os.path.exists(results_image_path):
    img = cv2.imread(results_image_path)
    # cv2 lê a imagem em BGR, converta para RGB para exibir corretamente com matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 6))
    plt.imshow(img_rgb)
    plt.title("Resultados do Treinamento YOLOv8")
    plt.axis('off')
    plt.show()
else:
    print("Arquivo de resultados não encontrado.")

# ==============================
# 📌 5. Loses de Treino por Época - V_50 épocas
# ==============================
# Loses de Treino por Época
import pandas as pd
import matplotlib.pyplot as plt

# Ler o CSV
results_csv_path = r"C:YV8SKFRAME\dataset\training_results\yolov8n_Soccer-Key-Frames13\results.csv"
df = pd.read_csv(results_csv_path)

# Exibir as colunas disponíveis
print("Colunas disponíveis:", df.columns)

plt.figure(figsize=(10, 6))
plt.plot(df["epoch"], df["train/box_loss"], label="train/box_loss")
plt.plot(df["epoch"], df["train/cls_loss"], label="train/cls_loss")
plt.plot(df["epoch"], df["train/dfl_loss"], label="train/dfl_loss")
plt.plot(df["epoch"], df["metrics/precision(B)"], label="metrics/precision(B)")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Losses de Treino por Época')
plt.legend()
plt.grid(True)
plt.show()

