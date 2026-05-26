import torch
import cv2
import os
import pandas as pd
import numpy as np
from PIL import Image
from transformers import ViTForImageClassification, AutoImageProcessor

print("Tudo OK")

# ================================
# CONFIGURAÇÕES
# ================================
valid_DIR  = r"C:/Vi_DataSet_S_K_Frame/valid"
MODEL_ID   = "google/vit-base-patch16-224-in21k"
MODEL_DIR  = r"C:/Fr_DataSet_S_K_Frame/ViT_training_results/vit_Soccer-Key-Frames/checkpoint-16400"

BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Thresholds
THR_ENTRY = 0.40
THR_EXIT  = 0.94

# Suavização
SMOOTH_WIN = 5

# Persistência temporal
MIN_FRAMES_ENTRY = 15
MIN_FRAMES_EXIT  = 3

# Offsets
START_OFFSET = 3
END_OFFSET   = 2

# Regras de pós-processamento
MIN_SEG_SECONDS = 2.0
MIN_GAP_SECONDS = 2.0

# ================================
# LOAD MODEL
# ================================
processor = AutoImageProcessor.from_pretrained(MODEL_ID)

model = ViTForImageClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()

print("Modelo carregado com sucesso.")

# ================================
# TIMECODE DROP FRAME
# ================================
def to_timecode_df(frame_idx):

    fps_int = 30
    drop_frames = 2

    frames_per_10_minutes = 17982
    frames_per_minute = 1798

    frame_number = int(frame_idx)

    d = frame_number // frames_per_10_minutes
    m = frame_number % frames_per_10_minutes

    total_minutes = d * 10 + m // frames_per_minute

    dropped = drop_frames * (total_minutes - total_minutes // 10)

    frame_number += dropped

    hh = frame_number // (fps_int * 3600)
    mm = (frame_number // (fps_int * 60)) % 60
    ss = (frame_number // fps_int) % 60
    ff = frame_number % fps_int

    return f"{hh:02d};{mm:02d};{ss:02d};{ff:02d}"

# ================================
# FILTRO / MERGE SEGMENTOS
# ================================
def filter_and_merge_segments(segments, fps):

    if not segments:
        return []

    min_seg_frames = int(MIN_SEG_SECONDS * fps)
    min_gap_frames = int(MIN_GAP_SECONDS * fps)

    # Remove segmentos curtos
    filtered = []

    for st, en in segments:
        if (en - st) >= min_seg_frames:
            filtered.append((st, en))

    if not filtered:
        return []

    # Merge segmentos próximos
    merged = [filtered[0]]

    for st, en in filtered[1:]:

        prev_st, prev_en = merged[-1]

        if (st - prev_en) < min_gap_frames:
            merged[-1] = (prev_st, en)

        else:
            merged.append((st, en))

    return merged

# ================================
# INFERÊNCIA
# ================================
def run_inference(video_path):

    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 29.97)

    frames = []
    probs_list = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append(pil)

        if len(frames) >= BATCH_SIZE:

            with torch.no_grad():
                inputs = processor(images=frames, return_tensors="pt").to(DEVICE)
                outputs = model(**inputs)

                logits = outputs.logits.cpu().numpy()
                probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

                probs_list.extend(probs[:, 0].tolist())

            frames = []

    # Último batch residual
    if frames:

        with torch.no_grad():
            inputs = processor(images=frames, return_tensors="pt").to(DEVICE)
            outputs = model(**inputs)

            logits = outputs.logits.cpu().numpy()
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

            probs_list.extend(probs[:, 0].tolist())

    cap.release()

    n = len(probs_list)

    if n == 0:
        return []

    # ================================
    # SUAVIZAÇÃO
    # ================================
    kernel = np.ones(SMOOTH_WIN) / SMOOTH_WIN
    smooth = np.convolve(probs_list, kernel, mode='same')

    # ================================
    # DETECÇÃO DE SEGMENTOS COM
    # PERSISTÊNCIA TEMPORAL
    # ================================
    segments = []

    state = "NO_PIP"
    entry_counter = 0
    exit_counter = 0
    start = None

    for i in range(n):

        if state == "NO_PIP":

            if smooth[i] >= THR_ENTRY:
                entry_counter += 1

                if entry_counter >= MIN_FRAMES_ENTRY:
                    start = max(0, i - MIN_FRAMES_ENTRY + 1 + START_OFFSET)
                    state = "IN_PIP"
                    exit_counter = 0
            else:
                entry_counter = 0

        elif state == "IN_PIP":

            if smooth[i] < THR_EXIT:
                exit_counter += 1

                if exit_counter >= MIN_FRAMES_EXIT:
                    end = max(0, i - MIN_FRAMES_EXIT + END_OFFSET)
                    segments.append((start, end))
                    state = "NO_PIP"
                    entry_counter = 0
                    exit_counter = 0
                    start = None
            else:
                exit_counter = 0

    # Fecha último segmento, se necessário
    if state == "IN_PIP" and start is not None:
        end = max(0, n - 1 + END_OFFSET)
        segments.append((start, end))

    # ================================
    # FILTRAGEM / MERGE
    # ================================
    segments = filter_and_merge_segments(segments, fps)

    # Converter para TC
    segments_tc = [
        (to_timecode_df(st), to_timecode_df(en))
        for st, en in segments
    ]

    return segments_tc

# ================================
# PROCESSAMENTO EM LOTE
# ================================
print("\n==== DETECÇÕES ====\n")

for vf in sorted(os.listdir(valid_DIR)):

    if not vf.lower().endswith((".mp4",".avi",".mov")):
        continue

    video_path = os.path.join(valid_DIR,vf)

    segments = run_inference(video_path)

    print(f"Vídeo: {vf}")

    if not segments:

        print("   Sem PiP detectado.\n")

    else:

        for st,en in segments:
            print(f"   PiP de {st} até {en}")

        print()