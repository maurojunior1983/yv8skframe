# =========================
# PASSO 1 — Setup limpo
# =========================
import os
import random
import numpy as np
import torch
import torchvision

from PIL import Image
import matplotlib.pyplot as plt

from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader

from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer
)

import evaluate

print("Torch:", torch.__version__)
print("Torchvision:", torchvision.__version__)

metric = evaluate.load("accuracy")


#PASSO 2 V1
import transformers
print(transformers.__version__)
from transformers import ViTImageProcessor
from transformers import AutoImageProcessor
import torchvision.transforms as T
from transformers import EarlyStoppingCallback
import os
import re
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from transformers import TrainingArguments

# =========================
# PASSO 2 (versão completa e corrigida)
# =========================
import os
import re
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from transformers import AutoImageProcessor
import evaluate

# -----------------------------
# Parâmetros / caminhos (edite conforme necessário)
# -----------------------------
model_id = "google/vit-base-patch16-224-in21k"
base_dir = r"C:\Fr_DataSet_S_K_Frame"   # ajuste se necessário
train_imgs = os.path.join(base_dir, "train", "images")
train_lbls = os.path.join(base_dir, "train", "labels")
val_imgs   = os.path.join(base_dir, "valid", "images")
val_lbls   = os.path.join(base_dir, "valid", "labels")
test_imgs  = os.path.join(base_dir, "test", "images")
test_lbls  = os.path.join(base_dir, "test", "labels")

# -----------------------------
# Processor (use_fast=True para evitar aviso)
# -----------------------------
processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)

# -----------------------------
# Transforms (PIL-level) — aplicadas *antes* do processor
# -----------------------------
train_transforms = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    T.RandomHorizontalFlip(p=0.5),
    # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # opcional
])

val_transforms = T.Compose([
    T.Resize((224, 224)),
])

# -----------------------------
# Funções utilitárias para labels
# -----------------------------
def parse_label_content(content, max_pairs=6):
    """
    Recebe o conteúdo bruto do .txt e retorna:
      - label (int): 0 = ad (PiP presente), 1 = no-ad (null / vazio)
      - coords (list of (x,y) floats) ou None se não houver coords
    Observações:
      - aceita 'null' (case-insensitive) ou string vazia -> sem propaganda
      - espera que a primeira token seja a classe (ex: "0 y1 x1 y2 x2 ...")
      - pares vêm como (Y,X) segundo sua descrição, então converte para (X,Y)
      - retorna no máximo `max_pairs` pares
    """
    if content is None:
        return 1, None

    s = content.strip()
    if not s or s.lower().startswith("null"):
        return 1, None

    tokens = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not tokens:
        return 1, None

    # primeira token = classe
    cls_token = tokens[0]
    try:
        label = 0 if int(float(cls_token)) == 0 else 1
    except:
        label = 1

    # resto -> coordenadas (Y X Y X ...)
    rest = tokens[1:]
    coords = None
    if rest:
        needed = max_pairs * 2
        vals = [float(x) for x in rest[:needed]]
        if len(vals) >= 2 and len(vals) % 2 == 0:
            pairs = []
            for i in range(0, len(vals), 2):
                y = vals[i]
                x = vals[i+1]
                pairs.append((x, y))  # reordena para (x, y)
            coords = pairs
        else:
            coords = None

    return label, coords

def convert_origin_bottom_right_to_top_left(coords, normalized=True):
    """
    Converte coords de origem bottom-right para top-left.
    Se normalized=True assume que coords estão em [0,1] e aplica (1 - v).
    """
    if coords is None:
        return None
    conv = []
    for (x, y) in coords:
        if normalized:
            conv.append((1.0 - x, 1.0 - y))
        else:
            conv.append((-x, -y))  # apenas fallback; preferível fornecer w/h se não normalizado
    return conv

# -----------------------------
# Dataset (aplica transforms PIL antes do processor)
# -----------------------------
class PiPDataset(Dataset):
    def __init__(self, images_dir, labels_dir, image_processor,
                 transforms=None, keep_coords=False,
                 coords_origin_bottom_right=True, coords_normalized=True):
        """
        transforms: PIL-level transforms (torchvision.transforms) aplicadas ANTES do processor.
        keep_coords: se True guarda coords parseadas em self.coords_dict
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.processor = image_processor
        self.transforms = transforms
        self.keep_coords = keep_coords
        self.coords_origin_bottom_right = coords_origin_bottom_right
        self.coords_normalized = coords_normalized

        self.files = sorted([
            f for f in os.listdir(images_dir)
            if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        ])

        if keep_coords:
            self.coords_dict = {}

    def __len__(self):
        return len(self.files)

    def _read_label_file(self, img_name):
        lbl_name = os.path.splitext(img_name)[0] + ".txt"
        path = os.path.join(self.labels_dir, lbl_name)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        return content

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        content = self._read_label_file(img_name)
        image = Image.open(img_path).convert("RGB")
        label, coords = parse_label_content(content)
        if coords and self.coords_origin_bottom_right:
            coords = convert_origin_bottom_right_to_top_left(
                coords, normalized=self.coords_normalized
            )
        inputs = self.processor(image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        if self.transforms:
            pixel_values = self.transforms(pixel_values)
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(label, dtype=torch.long)
        }


print("PASSO 2 concluído com sucesso")


# =========================
# PASSO 3 (compatível com PASSO 2)
# =========================
import os
import torch
from transformers import AutoImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# --- model_id (mesmo do PASSO 2) ---
model_id = "google/vit-base-patch16-224-in21k"

# --- Reusar processor do PASSO 2 se já existir, senão carrega com use_fast=True ---
try:
    processor  # se definido no PASSO 2
    print("Usando processor já carregado do PASSO 2.")
except NameError:
    processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
    print("Processor carregado com use_fast=True.")

# ---------------------------
# Instanciar datasets (usa PiPDataset definido no PASSO 2)
# ---------------------------
# Atenção: train_transforms / val_transforms também foram definidos no PASSO 2
train_dataset = PiPDataset(train_imgs, train_lbls, processor, transforms=train_transforms, keep_coords=False)
val_dataset   = PiPDataset(val_imgs,   val_lbls,   processor, transforms=val_transforms,   keep_coords=False)
test_dataset  = PiPDataset(test_imgs,  test_lbls,  processor, transforms=val_transforms,   keep_coords=False)

print("Tamanhos (antes de qualquer redução): train =", len(train_dataset), " val =", len(val_dataset), " test =", len(test_dataset))

# ---------------------------
# OPÇÃO 3 (debug): reduzir dataset para rodar rápido no CPU
# ---------------------------
# Descomente/ajuste se quiser rodar uma versão curta de teste.
MAX_TRAIN = None   # ex: 500   -> se None, usa tudo
MAX_VAL   = None   # ex: 200

if MAX_TRAIN is not None:
    train_dataset.files = train_dataset.files[:MAX_TRAIN]
if MAX_VAL is not None:
    val_dataset.files = val_dataset.files[:MAX_VAL]

if MAX_TRAIN is not None or MAX_VAL is not None:
    print("Após redução: train =", len(train_dataset), " val =", len(val_dataset))


# ------------------------------------------------
# Class weights (opcional)
# ------------------------------------------------
class_weights = None

# ---------------------------
# Modelo (binário)
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTForImageClassification.from_pretrained(model_id, num_labels=2).to(device)

# Se quiser testar fine-tune rápido no CPU, congele o backbone e treine só a cabeça:
# for param in model.vit.parameters():
#     param.requires_grad = False
# (Descomente para congelar; muito rápido para debug, mas limita o aprendizado)

# ---------------------------
# Collate e compute_metrics
# ---------------------------
# collate_fn e compute_metrics já foram definidos no PASSO 2 (se não, defina como abaixo)
# def collate_fn(batch):
#     pxs, labels = zip(*batch)
#     return {"pixel_values": torch.stack(pxs), "labels": torch.tensor(labels)}
# metric = evaluate.load("accuracy")
# def compute_metrics(p): ...
# (se você manteve PASSO 2, não precisa redefinir)

# ---------------------------
# TrainingArguments (ajustados para visualização/CPU)
# ---------------------------
training_args = TrainingArguments(
    output_dir=os.path.join(base_dir, "ViT_training_results", "vit_Soccer-Key-Frames"),
    per_device_train_batch_size=8,    # ajuste para sua RAM/CPU
    per_device_eval_batch_size=8,
    eval_strategy="steps",            # ver métricas durante a época
    eval_steps=200,                   # a cada N batches avalia
    save_strategy="steps",
    save_steps=200,
    num_train_epochs=20,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_strategy="steps",
    logging_steps=5,                  # ver loss frequentemente
    report_to="none"
)

# ---------------------------
# Instanciar Trainer
# ---------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# ---------------------------
# Pronto: rode quando quiser
# ---------------------------
# Para começar o treino (rode manualmente):
# trainer.train()
#
# Se estiver em CPU e quiser debug rápido:
# - mantenha MAX_TRAIN pequeno
# - mantenha logging_steps=5 e eval_steps baixo
#
print("PASSO 3 pronto. Trainer instanciado. Rode trainer.train() quando quiser.")


# =========================
# PASSO 4 (instancia datasets + modelo) — adaptado
# =========================
import torch
from transformers import AutoImageProcessor, ViTForImageClassification

# --- Reusar processor do PASSO 2 (se não existir, carrega com use_fast=True) ---
try:
    processor
    print("Usando processor já carregado do PASSO 2.")
except NameError:
    processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
    print("Processor carregado com use_fast=True.")

# --- Instanciar datasets usando a PiPDataset já definida (que aplica transforms PIL antes do processor) ---
# Se PiPDataset do PASSO 2 aceita argumento `transforms`, passe-os para treino/val
train_dataset = PiPDataset(train_imgs, train_lbls, processor, transforms=train_transforms, keep_coords=False)
val_dataset   = PiPDataset(val_imgs,   val_lbls,   processor, transforms=val_transforms,   keep_coords=False)
test_dataset  = PiPDataset(test_imgs,  test_lbls,  processor, transforms=val_transforms,   keep_coords=False)

print("Tamanhos: train =", len(train_dataset), " val =", len(val_dataset), " test =", len(test_dataset))

# --- collate_fn e compute_metrics
# Se já definidos no PASSO 2, não replique aqui. Só garanta que existem:
# def collate_fn(batch): ...
# def compute_metrics(p): ...

# --- Modelo (binário) — use device consistente
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTForImageClassification.from_pretrained(model_id, num_labels=2).to(device)

print("Modelo e datasets prontos.")


# =========================
# PASSO 5 (ajustado / CPU-friendly / logs frequentes)
# =========================
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
import os

# Ajuste prático para CPU: batches menores, logs mais frequentes, avaliação por steps
training_args = TrainingArguments(
    output_dir=os.path.join(base_dir, "ViT_training_results", "vit_Soccer-Key-Frames"),
    per_device_train_batch_size=8,      # reduzir em CPU; ajuste para 4 se faltar RAM
    per_device_eval_batch_size=8,
    eval_strategy="steps",              # avaliar durante a época
    eval_steps=200,                     # a cada N batches avalia (ajuste conforme dataset/batch)
    save_strategy="steps",
    save_steps=200,
    num_train_epochs=50,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=200,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",  # mantemos eval_loss como critério
    greater_is_better=False,
    logging_strategy="steps",
    logging_steps=5,                    # ver progresso frequentemente
    report_to="none",
    # gradient_accumulation_steps=2,    # opcional: acumula grads para simular batch maior
    remove_unused_columns=False
)

# Early stopping (mantém, mas não interrompe antes do número mínimo de épocas)
early_stop = EarlyStoppingCallback(
    early_stopping_patience=20,
    early_stopping_threshold=0.0
)

# Trainer (usa collate_fn e compute_metrics definidos anteriormente)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)


print("PASSO 5 pronto. Para iniciar o treino rode: trainer.train()")
# Dica: para debug rápido em CPU, defina MAX_TRAIN no PASSO 3 e use logging_steps=5 / eval_steps baixo.
# Se quiser acelerar teste inicial: descomente a linha abaixo para treinar só a cabeça do modelo
# for param in model.vit.parameters(): param.requires_grad = False


# =========================
# PASSO 6 (rodar treino, salvar melhor modelo, avaliar e exportar previsões)
# =========================
import os
import json
import csv
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

out_dir = os.path.join(base_dir, "ViT_training_results", "vit_Soccer-Key-Frames")
os.makedirs(out_dir, exist_ok=True)
best_model_dir = os.path.join(out_dir, "best_model")
os.makedirs(best_model_dir, exist_ok=True)

# 1) Treino (garantir save mesmo em KeyboardInterrupt)
try:
    train_result = trainer.train()
except KeyboardInterrupt:
    print("Treino interrompido manualmente (KeyboardInterrupt). Salvando modelo atual...")
finally:
    # salva o modelo atual (com load_best_model_at_end=True, trainer.model será o melhor)
    trainer.save_model(best_model_dir)
    print(f"Modelo salvo em: {best_model_dir}")

# 2) Avaliação (val + test)
val_metrics = trainer.evaluate(eval_dataset=val_dataset)
print("Val metrics:", val_metrics)

test_metrics = trainer.evaluate(eval_dataset=test_dataset)
print("Test metrics:", test_metrics)

# salvar métricas em JSON
metrics_path = os.path.join(out_dir, "metrics_after_train.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump({"val": val_metrics, "test": test_metrics}, f, indent=2)
print("Métricas salvas em:", metrics_path)

# 3) Predições no test_dataset (logits -> probs)
pred_out = trainer.predict(test_dataset)   # retorna PredictionOutput(predictions, label_ids, metrics)
logits = pred_out.predictions               # (N, num_labels)
labels = pred_out.label_ids                 # pode ser None se não houver labels
probs = torch.nn.functional.softmax(torch.from_numpy(logits), dim=1).numpy()
preds = probs.argmax(axis=1)

# 4) Relatório e matriz de confusão (se labels presentes)
if labels is not None:
    labels = np.array(labels)
    print("Confusion matrix:")
    print(confusion_matrix(labels, preds))
    print("\nClassification report:")
    print(classification_report(labels, preds, digits=4))

    # salvar relatório em arquivo txt
    report_path = os.path.join(out_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Confusion matrix:\n")
        f.write(np.array2string(confusion_matrix(labels, preds)))
        f.write("\n\nClassification report:\n")
        f.write(classification_report(labels, preds, digits=4))
    print("Relatório salvo em:", report_path)
else:
    print("Labels ausentes no test_dataset — pulei relatório/CM.")

# 5) Exportar CSV com previsões (filename, true_label, pred, prob_class_0, prob_class_1)
csv_out = os.path.join(out_dir, "test_predictions.csv")
with open(csv_out, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "true_label", "pred", "prob_class_0", "prob_class_1"])
    # assumimos que test_dataset.files existe (PiPDataset tem .files)
    for i, fname in enumerate(test_dataset.files):
        true_lbl = int(labels[i]) if (labels is not None) else ""
        writer.writerow([fname, true_lbl, int(preds[i]), float(probs[i,0]), float(probs[i,1])])

print("Predições salvas em:", csv_out)
print("PASSO 6 concluído.")


# ------------------------------
# Encontrar melhor checkpoint e gerar gráfico por época
# ------------------------------
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# ------------------------------
# Ajuste este caminho se necessário
# ------------------------------
base_results_dir = Path(r"C:\Fr_DataSet_S_K_Frame\ViT_training_results\vit_Soccer-Key-Frames")

# ------------------------------
# Localizar arquivos trainer_state.json em checkpoints
# ------------------------------
trainer_state_paths = sorted(base_results_dir.glob("checkpoint-*/trainer_state.json"))
if not trainer_state_paths:
    raise FileNotFoundError(f"Nenhum trainer_state.json encontrado em {base_results_dir}/checkpoint-*")

summary = []
for p in trainer_state_paths:
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        summary.append((p, math.inf, None))
        continue

    logs = data.get("log_history", [])
    # extrair o menor eval_loss e a epoch correspondente
    min_eval_loss = math.inf
    min_epoch = None
    for entry in logs:
        if "eval_loss" in entry:
            try:
                val = float(entry["eval_loss"])
                if val < min_eval_loss:
                    min_eval_loss = val
                    # some entries use 'epoch'
                    min_epoch = entry.get("epoch", None)
            except Exception:
                pass
    summary.append((p, min_eval_loss, min_epoch))

# imprimir tabela resumo
print("Resumo de checkpoints (arquivo, min_eval_loss, epoch_of_min):")
for p, loss, ep in summary:
    loss_str = f"{loss:.6f}" if loss != math.inf else "N/A"
    print(f" - {p.parent.name}: min_eval_loss={loss_str}, epoch={ep}")

# escolher o checkpoint com menor eval_loss
# filtra aqueles com loss != inf
valid = [s for s in summary if s[1] != math.inf]
if valid:
    best_path, best_loss, best_epoch = min(valid, key=lambda t: (t[1], float(t[2]) if t[2] is not None else 0.0))
else:
    # se nenhum tiver eval_loss, escolhe o último checkpoint por nome (maior número)
    best_path = trainer_state_paths[-1]
    best_loss = None
    best_epoch = None

print("\nMelhor checkpoint selecionado:", best_path.parent.name)
print("min_eval_loss:", best_loss, " epoch:", best_epoch)

# ------------------------------
# Carregar logs do best_path e construir dataframe por epoch
# ------------------------------
state = json.loads(best_path.read_text(encoding="utf-8"))
logs = pd.DataFrame(state.get("log_history", []))
if logs.empty:
    raise RuntimeError(f"log_history vazio em {best_path}")

logs = logs.dropna(subset=['epoch']).copy()
logs['epoch'] = logs['epoch'].astype(float)
logs = logs.sort_values('epoch')

# train loss por epoch (média dos steps)
train_steps = logs[logs['loss'].notna()].copy()
if not train_steps.empty:
    train_by_epoch = train_steps.groupby('epoch')['loss'].mean()
else:
    train_by_epoch = pd.Series(dtype=float)

# validação: último valor por epoch (se houver)
eval_steps = logs[logs['eval_loss'].notna()].copy()
if not eval_steps.empty:
    val_by_epoch = eval_steps.groupby('epoch').agg({'eval_loss':'last', 'eval_accuracy':'last'})
else:
    val_by_epoch = pd.DataFrame()

# combinar índices de época
all_epochs = sorted(set(train_by_epoch.index.tolist() + val_by_epoch.index.tolist()))
df_plot = pd.DataFrame(index=all_epochs)
if not train_by_epoch.empty:
    df_plot = df_plot.join(train_by_epoch.rename("train_loss"), how='left')
if not val_by_epoch.empty:
    df_plot = df_plot.join(val_by_epoch.rename(columns={'eval_loss':'val_loss','eval_accuracy':'val_acc'}), how='left')

df_plot.index.name = 'epoch'
df_plot = df_plot.sort_index()

# preencher gaps (opcional) - mantém NaNs visíveis
# df_plot = df_plot.interpolate(method='linear', limit_direction='both')

# ------------------------------
# Preparar limites e plot
# ------------------------------
loss_vals = df_plot[['train_loss','val_loss']].values.flatten()
loss_vals = loss_vals[~np.isnan(loss_vals)]
if len(loss_vals) == 0:
    ymin, ymax = 0.0, 1.0
else:
    ymin = 0.0
    ymax = float(np.nanmax(loss_vals)) * 1.15
    if ymax == 0.0:
        ymax = 1.0

# aumentar figura para evitar sobreposição e colocar legendas fora
fig, ax1 = plt.subplots(figsize=(12, 6.5))

# losses (eixo esquerdo)
line_train = None
line_val = None
if 'train_loss' in df_plot:
    line_train, = ax1.plot(df_plot.index, df_plot['train_loss'], '-o', label='Train Loss', linewidth=1.5)
if 'val_loss' in df_plot:
    line_val, = ax1.plot(df_plot.index, df_plot['val_loss'], '-s', label='Validation Loss', linewidth=1.5)

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_ylim(ymin, ymax)
ax1.grid(True, axis='y', linestyle='--', alpha=0.35)

# accuracy (eixo direito)
ax2 = ax1.twinx()
line_acc = None
if 'val_acc' in df_plot:
    line_acc, = ax2.plot(df_plot.index, df_plot['val_acc'], '--^', label='Validation Accuracy', linewidth=1.5)
ax2.set_ylabel('Validation Accuracy')
ax2.set_ylim(0.0, 1.0)

# legendas: colocar fora, acima do plot, alinhadas
left_handles = [h for h in (line_train, line_val) if h is not None]
left_labels = [h.get_label() for h in left_handles]
right_handles = [h for h in (line_acc,) if h is not None]
right_labels = [h.get_label() for h in right_handles]

if left_handles:
    ax1.legend(left_handles, left_labels, loc='upper left',
               bbox_to_anchor=(0.0, 1.18), ncol=max(1, len(left_handles)), frameon=True, fancybox=True)
if right_handles:
    ax2.legend(right_handles, right_labels, loc='upper right',
               bbox_to_anchor=(1.0, 1.18), frameon=True, fancybox=True)

plt.title(f"Training evolution — best checkpoint: {best_path.parent.name} (min_eval_loss={best_loss})")
plt.tight_layout(rect=[0, 0, 1, 0.95])

# salvar arquivo PNG no diretório base_results_dir
out_png = base_results_dir / "training_evolution_best_checkpoint.png"
fig.savefig(out_png, dpi=200, bbox_inches='tight')
print("\nGráfico salvo em:", out_png)

plt.show()


# =========================
# PASSO 8 — Identificar melhor checkpoint (via trainer_state.json)
# =========================
import json
from pathlib import Path
import math

base_results_dir = Path(
    r"C:\Fr_DataSet_S_K_Frame\ViT_training_results\vit_Soccer-Key-Frames"
)

trainer_states = sorted(base_results_dir.glob("checkpoint-*/trainer_state.json"))
if not trainer_states:
    raise FileNotFoundError("Nenhum trainer_state.json encontrado em checkpoint-*")

best_checkpoint = None
best_eval_loss = math.inf
best_epoch = None

for state_path in trainer_states:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    for entry in state.get("log_history", []):
        if "eval_loss" in entry:
            try:
                val = float(entry["eval_loss"])
                if val < best_eval_loss:
                    best_eval_loss = val
                    best_checkpoint = state_path.parent
                    best_epoch = entry.get("epoch", None)
            except Exception:
                pass

if best_checkpoint is None:
    # fallback: escolhe o último checkpoint por número
    best_checkpoint = sorted(
        base_results_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1])
    )[-1]
    print("⚠️ Nenhum eval_loss encontrado. Usando último checkpoint.")

print("Melhor checkpoint selecionado:")
print(" → path :", best_checkpoint)
print(" → min eval_loss :", best_eval_loss)
print(" → epoch :", best_epoch)