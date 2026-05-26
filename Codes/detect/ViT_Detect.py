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
test_DIR  = r"C:/Vi_DataSet_S_K_Frame/test"
MODEL_ID   = "google/vit-base-patch16-224-in21k"
MODEL_DIR  = r"C:/Fr_DataSet_S_K_Frame/ViT_training_results/vit_Soccer-Key-Frames/checkpoint-16400"

BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Parâmetros da tabela 2
# Thresholds
THR_ENTRY = 0.40
THR_EXIT  = 0.94

# Suavização
SMOOTH_WIN = 5

# Persistência temporal
MIN_FRAMES_ENTRY = 15
MIN_FRAMES_EXIT  = 3

# Offsets (Tabela 2)
START_OFFSET = 0
END_OFFSET   = 0

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

results = {}

for vf in sorted(os.listdir(test_DIR)):

    if not vf.lower().endswith((".mp4",".avi",".mov")):
        continue

    video_path = os.path.join(test_DIR,vf)

    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 29.97)
    cap.release()

    segments = run_inference(video_path)

    results[vf] = {
        "segments_tc": segments,
        "fps": fps
    }

    print(f"Vídeo: {vf}")

    if not segments:
        print("   Sem PiP detectado.\n")

    else:
        for st,en in segments:
            print(f"   PiP de {st} até {en}")

        print()
test_DIR  = r"C:/Vi_DataSet_S_K_Frame/test"
META_PATH  = r"C:/DataSet_SoccerKeyFrame - Org/DataSet_SoccerKeyFrame.xlsx"
OUT_DIR    = r"C:/Fr_DataSet_S_K_Frame/ViT_Detect_results"
MODEL_ID   = "google/vit-base-patch16-224-in21k"
MODEL_DIR  = r"C:/Fr_DataSet_S_K_Frame/ViT_training_results/vit_Soccer-Key-Frames/checkpoint-16400"
# (1) carregar meta_df
meta_df = pd.read_excel(META_PATH)

# (2) definir funções auxiliares
FRAME_BASE_REAL = 30

def tc_to_frame_real(tc, fps=30):
    tc = str(tc).strip()

    if ';' in tc:
        hh, mm, ss, ff = map(int, tc.split(';'))
        drop_frames = 2
        total_minutes = hh * 60 + mm
        dropped = drop_frames * (total_minutes - total_minutes // 10)
        total_frames = ((hh * 3600 + mm * 60 + ss) * 30) + ff
        return total_frames - dropped

    tc_norm = tc.replace('.', ':')
    parts = tc_norm.split(':')

    if len(parts) == 3:
        mm, ss, ff = map(int, parts)
        hh = 0
        return int(round((hh * 3600 + mm * 60 + ss) * fps)) + ff
    elif len(parts) == 4:
        hh, mm, ss, ff = map(int, parts)
        return int(round((hh * 3600 + mm * 60 + ss) * fps)) + ff
    else:
        raise ValueError(f"Formato de timecode não reconhecido: {tc}")

def tc_detect_to_idx(tc_str, fps):
    return tc_to_frame_real(tc_str, fps)
    # ------------------------
# Exportar Planilha 1: comparação start/end (mesma lógica que você tinha)
# ------------------------
rows1 = []
for vf, info in results.items():
    real = meta_df[meta_df["video_path"].str.contains(vf, na=False)]
    if real.empty or pd.isna(real.iloc[0]["start_time"]):
        continue

    real_start_tc = str(real.iloc[0]["start_time"])
    real_end_tc   = str(real.iloc[0]["end_time"])
    fps_i = info["fps"]

    for det_start_tc, det_end_tc in info["segments_tc"]:
        det_start_f  = tc_to_frame_real(det_start_tc, int(round(fps_i)))
        det_end_f    = tc_to_frame_real(det_end_tc,   int(round(fps_i)))
        real_start_f = tc_to_frame_real(real_start_tc, FRAME_BASE_REAL)
        real_end_f   = tc_to_frame_real(real_end_tc,   FRAME_BASE_REAL)
        rows1.append({
            "video_file":    vf,
            "start_detect":  det_start_tc,
            "start_real":    real_start_tc,
            "match_start":   (det_start_f == real_start_f),
            "end_detect":    det_end_tc,
            "end_real":      real_end_tc,
            "match_end":     (det_end_f   == real_end_f),
        })

df1 = pd.DataFrame(rows1)
out1 = os.path.join(OUT_DIR, "comparison_start_end_vit.xlsx")
df1.to_excel(out1, index=False)
print(f"✅ Planilha #1 salva em: {out1}\n")

# ------------------------
# Exportar Planilha 2
# ------------------------
rows2 = []

for vf, info in results.items():
    real = meta_df[meta_df["video_path"].astype(str).str.contains(vf, na=False)]
    if real.empty or pd.isna(real.iloc[0]["start_time"]):
        continue

    segs = info["segments_tc"]
    if not segs:
        continue

    det_start_tc, det_end_tc = segs[0]

    det_start_idx = tc_to_frame_real(det_start_tc, info["fps"])
    det_end_idx   = tc_to_frame_real(det_end_tc,   info["fps"])

    start_real_tc = str(real.iloc[0]["start_time"]).strip()
    end_real_tc   = str(real.iloc[0]["end_time"]).strip()

    real_start_idx_30 = tc_to_frame_real(start_real_tc, FRAME_BASE_REAL)
    real_end_idx_30   = tc_to_frame_real(end_real_tc,   FRAME_BASE_REAL)

    scale = info["fps"] / FRAME_BASE_REAL
    real_start_idx = int(round(real_start_idx_30 * scale))
    real_end_idx   = int(round(real_end_idx_30   * scale))

    delta_start = det_start_idx - real_start_idx
    delta_end   = det_end_idx   - real_end_idx

    FP_start = abs(delta_start) if delta_start < 0 else 0
    FN_start = delta_start      if delta_start > 0 else 0
    FP_end   = delta_end        if delta_end   > 0 else 0
    FN_end   = abs(delta_end)   if delta_end   < 0 else 0

    frame_diff = FP_start + FN_start + FP_end + FN_end
    REAL_PIP_FRAMES = 232
    error_pct = (frame_diff / REAL_PIP_FRAMES) * 100.0

    rows2.append({
        "video_file":   vf,
        "start_detect": det_start_tc,
        "start_real":   start_real_tc,
        "FP_start":     FP_start,
        "FN_start":     FN_start,
        "end_detect":   det_end_tc,
        "end_real":     end_real_tc,
        "FP_end":       FP_end,
        "FN_end":       FN_end,
        "frame_diff":   frame_diff,
        "error_pct":    error_pct
    })

df2 = pd.DataFrame(rows2)
out2 = os.path.join(OUT_DIR, "frame_fp_fn_vit_adjusted.xlsx")
df2.to_excel(out2, index=False)
print(f"✅ Planilha #2 salva em: {out2}\n")