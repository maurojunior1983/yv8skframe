# ==============================
# 📌 1. Instalar o Ultralytics - YOLO26 - V_50 épocas
# ==============================

# Instalar ou atualizar o Ultralytics
!pip install -U ultralytics

# Imports principais
import os
import torch
from ultralytics import YOLO

print("Tudo OK")

# ==============================
# Verificar ambiente
# ==============================

print("PyTorch version:", torch.__version__)
print("CUDA disponível:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("⚠️ GPU não disponível. O treinamento será feito em CPU.")

# ==============================
# Verificar se o dataset está presente
# ==============================

dataset_path = r"C:/Fr_DataSet_S_K_Frame/TRAIN/images"

if os.path.exists(dataset_path) and len(os.listdir(dataset_path)) > 0:
    print("✅ Dataset C:/Fr_DataSet_S_K_Frame/TRAIN/images encontrado e pronto para treino!")
else:
    print("❌ Dataset NÃO encontrado ou vazio! Verifique o caminho e a estrutura.")

# ==============================
# Testar carregamento do modelo YOLO26
# ==============================

model = YOLO("yolo26n.pt")

print("✅ Modelo YOLO26 carregado com sucesso!")


# ==============================
# 📌 2. Treinar o Modelo YOLO26 para detectar PiP - V_50 épocas
# ==============================

from ultralytics import YOLO
import torch

# Carregar o modelo pré-treinado YOLO26
# Equivalente ao yolov8n.pt, usando a versão nano do YOLO26
model = YOLO("yolo26n.pt")

# Iniciar o treinamento
results = model.train(
    data="C:/Fr_DataSet_S_K_Frame/data.yaml",  # Caminho absoluto para o arquivo YAML do dataset
    epochs=50,
    imgsz=(640, 360),  # Mantém o mesmo tamanho usado no YOLOv8
    batch=16,
    project="C:/Fr_DataSet_S_K_Frame/training_results",  # Diretório onde os resultados serão salvos
    name="yolo26n_Soccer-Key-Frames",  # Nome do experimento/modelo treinado
    pretrained=True,
    device=0 if torch.cuda.is_available() else "cpu"
)

print("✅ Treinamento YOLO26 concluído!")


# ==============================
# 📌 3. Avaliar o Modelo YOLO26 no Conjunto de Teste PiP - V_50 épocas
# ==============================

from ultralytics import YOLO

# Carregar o melhor peso treinado do YOLO26
model = YOLO("C:/Fr_DataSet_S_K_Frame/training_results/yolo26n_Soccer-Key-Frames/weights/best.pt")

# Realizar avaliação no conjunto de teste
metrics = model.val(
    data="C:/Fr_DataSet_S_K_Frame/data.yaml",
    split="test"
)

print("Métricas de avaliação YOLO26:")
print(metrics)

# Testar com uma imagem do conjunto de teste
model(
    "C:/Fr_DataSet_S_K_Frame/test/images/CFOG29T_01-30-11_mp4-0193_jpg.rf.51bab918498e7fcebb4ace4c0cb36978.jpg",
    save=True
)

print("✅ Avaliação YOLO26 no conjunto de teste concluída!")


# ==============================
# 📌 4. Resultados do Treinamento YOLO26 - V_50 épocas
# ==============================

# Resultados do Treinamento YOLO26
%matplotlib inline
import os
import matplotlib.pyplot as plt
import cv2

# Caminho do gráfico de resultados salvo durante o treinamento do YOLO26
results_image_path = "C:/Fr_DataSet_S_K_Frame/training_results/yolo26n_Soccer-Key-Frames/results.png"

if os.path.exists(results_image_path):
    img = cv2.imread(results_image_path)

    # Verifica se a imagem foi lida corretamente
    if img is not None:
        # cv2 lê a imagem em BGR; converter para RGB para exibir corretamente no matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 6))
        plt.imshow(img_rgb)
        plt.title("Resultados do Treinamento YOLO26")
        plt.axis("off")
        plt.show()
    else:
        print("❌ O arquivo foi encontrado, mas não pôde ser lido pelo OpenCV.")
else:
    print("❌ Arquivo de resultados YOLO26 não encontrado.")
    print("Verifique o caminho:")
    print(results_image_path)


# ==============================
# 📌 5. Losses de Treino por Época - YOLO26 - V_50 épocas
# ==============================

import os
import pandas as pd
import matplotlib.pyplot as plt

# Ler o CSV de resultados do YOLO26
results_csv_path = r"C:\Fr_DataSet_S_K_Frame\training_results\yolo26n_Soccer-Key-Frames\results.csv"

if os.path.exists(results_csv_path):
    df = pd.read_csv(results_csv_path)

    # Remover espaços extras nos nomes das colunas, caso existam
    df.columns = df.columns.str.strip()

    # Exibir as colunas disponíveis
    print("Colunas disponíveis:", df.columns)

    plt.figure(figsize=(10, 6))

    plt.plot(df["epoch"], df["train/box_loss"], label="train/box_loss")
    plt.plot(df["epoch"], df["train/cls_loss"], label="train/cls_loss")
    plt.plot(df["epoch"], df["train/dfl_loss"], label="train/dfl_loss")
    plt.plot(df["epoch"], df["metrics/precision(B)"], label="metrics/precision(B)")

    plt.xlabel("Epoch")
    plt.ylabel("Loss / Precision")
    plt.title("Losses de Treino por Época - YOLO26")
    plt.legend()
    plt.grid(True)
    plt.show()

else:
    print("❌ Arquivo results.csv do YOLO26 não encontrado.")
    print("Verifique o caminho:")
    print(results_csv_path)