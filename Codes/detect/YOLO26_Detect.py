# ===========================
# IMPORTS
# ===========================
import os
import pandas as pd
import numpy as np
from ultralytics import YOLO
import torch
import cv2
from collections import deque

print(np.__version__)
print("Tudo OK")


# ==========================================
# CONVERSÃO FRAME → TIMECODE DROP FRAME
# ==========================================
def frame_to_dropframe_tc(frame_number, fps=29.97):

    nominal_fps = 30
    drop_frames = 2

    frames_per_hour = 107892
    frames_per_24_hours = frames_per_hour * 24
    frames_per_10_minutes = 17982
    frames_per_minute = 1798

    frame_number = int(round(frame_number))
    frame_number = frame_number % frames_per_24_hours

    d = frame_number // frames_per_10_minutes
    m = frame_number % frames_per_10_minutes

    total_minutes = d * 10 + m // frames_per_minute

    dropped = drop_frames * (total_minutes - total_minutes // 10)

    frame_number_adjusted = frame_number + dropped

    hours = frame_number_adjusted // (nominal_fps * 60 * 60)
    minutes = (frame_number_adjusted // (nominal_fps * 60)) % 60
    seconds = (frame_number_adjusted // nominal_fps) % 60
    frames = frame_number_adjusted % nominal_fps

    return f"{hours:02d}:{minutes:02d}:{seconds:02d};{frames:02d}"


FRAME_BASE_REAL = 30


def tc_to_frame_real(tc, fps=30):
    tc = str(tc).strip()

    # Caso DF: HH:MM:SS;FF
    if ";" in tc:
        time_part, ff = tc.split(";")
        hh, mm, ss = map(int, time_part.split(":"))
        ff = int(ff)

        drop_frames = 2
        total_minutes = hh * 60 + mm
        dropped = drop_frames * (total_minutes - total_minutes // 10)

        total_frames = ((hh * 3600 + mm * 60 + ss) * 30) + ff
        return total_frames - dropped

    # Padroniza formatos restantes
    tc = tc.replace(".", ":")
    parts = tc.split(":")

    # Caso MM:SS:FF
    if len(parts) == 3:
        mm, ss, ff = map(int, parts)
        return int(round((mm * 60 + ss) * fps)) + ff

    # Caso HH:MM:SS:FF
    elif len(parts) == 4:
        hh, mm, ss, ff = map(int, parts)
        return int(round((hh * 3600 + mm * 60 + ss) * fps)) + ff

    else:
        raise ValueError(f"Formato inválido: {tc}")


# ==========================================
# PROCESSAMENTO DO VÍDEO - YOLO26
# ==========================================
def process_video_for_PiP(
    video_path,
    model,
    threshold_entry=0.3,
    threshold_exit=0.98,
    min_frames_entry=2,
    min_frames_exit=3,
    smoothing_window=5
):

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_detections = []
    frame_idx = 0

    state = "NO_PIP"
    entry_counter = 0
    exit_counter = 0

    conf_buffer = deque(maxlen=smoothing_window)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_resized = cv2.resize(frame, (640, 360))
        results = model(frame_resized, verbose=False)

        max_conf = 0.0

        for result in results:

            boxes = result.boxes

            if boxes is None:
                continue

            for box in boxes:

                # Classe 0 = PiP / propaganda
                if int(box.cls.item()) == 0:

                    conf = float(box.conf.item())

                    if conf > max_conf:
                        max_conf = conf

        # Suavização temporal
        conf_buffer.append(max_conf)
        smooth_conf = sum(conf_buffer) / len(conf_buffer)

        detected = False

        # Máquina de estados
        if state == "NO_PIP":

            if smooth_conf >= threshold_entry:

                entry_counter += 1

                if entry_counter >= min_frames_entry:

                    state = "IN_PIP"
                    detected = True
                    exit_counter = 0

            else:

                entry_counter = 0

        elif state == "IN_PIP":

            detected = True

            if smooth_conf < threshold_exit:

                exit_counter += 1

                if exit_counter >= min_frames_exit:

                    state = "NO_PIP"
                    detected = False
                    entry_counter = 0

            else:

                exit_counter = 0

        timecode = frame_to_dropframe_tc(frame_idx, fps)

        frame_detections.append((frame_idx, detected, timecode))

        frame_idx += 1

    cap.release()

    return frame_detections, fps


# ==========================================
# EXTRAÇÃO DOS SEGMENTOS DETECTADOS
# ==========================================
def get_detected_segments(
    detections,
    fps,
    start_frame_offset=0,
    end_frame_offset=0,
    min_duration_sec=2.0,
    min_gap_sec=2.0,
    print_segments=True,
):
    segments = []
    in_segment = False
    start_idx = None

    # -----------------------------
    # 1. Criar segmentos brutos com offsets
    # -----------------------------
    for i, (idx, detected, _) in enumerate(detections):

        if detected and not in_segment:
            in_segment = True
            start_idx = max(0, idx - start_frame_offset)

        elif not detected and in_segment:
            end_idx = max(0, detections[i - 1][0] - end_frame_offset)
            segments.append((start_idx, end_idx))
            in_segment = False

    if in_segment:
        end_idx = detections[-1][0]
        segments.append((start_idx, end_idx))

    if not segments:
        if print_segments:
            print("Without PiP")
        return []

    # -----------------------------
    # 2. Juntar segmentos próximos
    # -----------------------------
    merged_segments = []
    current_start, current_end = segments[0]

    for next_start, next_end in segments[1:]:

        gap_frames = next_start - current_end
        gap_sec = gap_frames / fps if fps > 0 else 0

        if gap_sec < min_gap_sec:
            current_end = next_end
        else:
            merged_segments.append((current_start, current_end))
            current_start, current_end = next_start, next_end

    merged_segments.append((current_start, current_end))

    # -----------------------------
    # 3. Filtrar por duração mínima
    # -----------------------------
    final_segments = []

    for start, end in merged_segments:

        duration_sec = (end - start) / fps if fps > 0 else 0

        if duration_sec >= min_duration_sec:
            final_segments.append((start, end))

    if not final_segments:
        if print_segments:
            print("Without PiP")
        return []

    # -----------------------------
    # 4. Imprimir segmentos finais
    # -----------------------------
    if print_segments:
        for start, end in final_segments:

            start_tc = frame_to_dropframe_tc(start, fps)
            end_tc = frame_to_dropframe_tc(end, fps)

            duration_frames = end - start
            duration_tc = frame_to_dropframe_tc(duration_frames, fps)

            print(f"PiP from {start_tc} to {end_tc} - Duration: {duration_tc}")

    return final_segments


# ==========================================
# MAIN - INFERÊNCIA YOLO26 NO CONJUNTO DE TESTES
# ==========================================

# Peso treinado do YOLO26
model_path = r"C:/Fr_DataSet_S_K_Frame/training_results/yolo26n_Soccer-Key-Frames/weights/best.pt"

# Diretório com vídeos do conjunto de teste
test_dir = r"C:\Vi_DataSet_S_K_Frame\test"

# Diretório de saída dos resultados do YOLO26
OUT_DIR = r"C:/Fr_DataSet_S_K_Frame/YOLO26_Detect_results"
os.makedirs(OUT_DIR, exist_ok=True)

# Planilha com os timecodes reais
META_PATH = r"C:/DataSet_SoccerKeyFrame - Org/DataSet_SoccerKeyFrame.xlsx"

# Carregar modelo YOLO26
model = YOLO(model_path)

# Carregar metadados reais
meta_df = pd.read_excel(META_PATH)

arquivos = os.listdir(test_dir)
results = {}


# ===========================
# Parâmetros da Tabela 2 - YOLO26
# ===========================
THRESHOLD_ENTRY = 0.3
THRESHOLD_EXIT = 0.98
MIN_FRAMES_ENTRY = 2
MIN_FRAMES_EXIT = 3
SMOOTHING_WINDOW = 5
START_FRAME_OFFSET = 0
END_FRAME_OFFSET = 0
MIN_DURATION_SEC = 2.0
MIN_GAP_SEC = 2.0


# ==========================================
# LOOP DE INFERÊNCIA NOS VÍDEOS DE TESTE
# ==========================================
for video_file in arquivos:

    if video_file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):

        video_path = os.path.join(test_dir, video_file)

        print(f"\nProcessando vídeo: {video_file}")

        detections, fps = process_video_for_PiP(
            video_path=video_path,
            model=model,
            threshold_entry=THRESHOLD_ENTRY,
            threshold_exit=THRESHOLD_EXIT,
            min_frames_entry=MIN_FRAMES_ENTRY,
            min_frames_exit=MIN_FRAMES_EXIT,
            smoothing_window=SMOOTHING_WINDOW
        )

        final_segments = get_detected_segments(
            detections=detections,
            fps=fps,
            start_frame_offset=START_FRAME_OFFSET,
            end_frame_offset=END_FRAME_OFFSET,
            min_duration_sec=MIN_DURATION_SEC,
            min_gap_sec=MIN_GAP_SEC,
            print_segments=True,
        )

        segments_tc = [
            (
                frame_to_dropframe_tc(st, fps),
                frame_to_dropframe_tc(en, fps)
            )
            for st, en in final_segments
        ]

        results[video_file] = {
            "segments_tc": segments_tc,
            "segments_frames": final_segments,
            "fps": fps
        }


# ==========================================
# EXPORTAR PLANILHA 1
# COMPARAÇÃO START/END DETECTADO VS REAL
# ==========================================
rows1 = []

for vf, info in results.items():

    real = meta_df[meta_df["video_path"].str.contains(vf, na=False)]

    if real.empty or pd.isna(real.iloc[0]["start_time"]):
        continue

    real_start_tc = str(real.iloc[0]["start_time"])
    real_end_tc = str(real.iloc[0]["end_time"])
    fps_i = info["fps"]

    for det_start_tc, det_end_tc in info["segments_tc"]:

        det_start_f = tc_to_frame_real(det_start_tc, int(round(fps_i)))
        det_end_f = tc_to_frame_real(det_end_tc, int(round(fps_i)))

        real_start_f = tc_to_frame_real(real_start_tc, FRAME_BASE_REAL)
        real_end_f = tc_to_frame_real(real_end_tc, FRAME_BASE_REAL)

        rows1.append({
            "video_file": vf,
            "start_detect": det_start_tc,
            "start_real": real_start_tc,
            "match_start": det_start_f == real_start_f,
            "end_detect": det_end_tc,
            "end_real": real_end_tc,
            "match_end": det_end_f == real_end_f,
        })

df1 = pd.DataFrame(rows1)

out1 = os.path.join(OUT_DIR, "comparison_start_end_yolo26.xlsx")
df1.to_excel(out1, index=False)

print(f"\n✅ Planilha #1 salva em: {out1}")


# ==========================================
# EXPORTAR PLANILHA 2
# FP/FN POR BORDA - START E END
# ==========================================
rows2 = []

for vf, info in results.items():

    real = meta_df[meta_df["video_path"].str.contains(vf, na=False)]

    if real.empty or pd.isna(real.iloc[0]["start_time"]):
        continue

    segs = info["segments_tc"]

    if not segs:
        continue

    # Usa o primeiro segmento detectado
    det_start_tc, det_end_tc = segs[0]

    def tc_detect_to_idx(tc_str, fps):
        hh, mm, ss, ff = map(int, tc_str.replace(";", ":").split(":"))
        return int(round((hh * 3600 + mm * 60 + ss) * fps)) + ff

    det_start_idx = tc_detect_to_idx(det_start_tc, info["fps"])
    det_end_idx = tc_detect_to_idx(det_end_tc, info["fps"])

    start_real_tc = str(real.iloc[0]["start_time"])
    end_real_tc = str(real.iloc[0]["end_time"])

    real_start_idx_30 = tc_to_frame_real(start_real_tc, FRAME_BASE_REAL)
    real_end_idx_30 = tc_to_frame_real(end_real_tc, FRAME_BASE_REAL)

    scale = info["fps"] / FRAME_BASE_REAL

    real_start_idx = int(round(real_start_idx_30 * scale))
    real_end_idx = int(round(real_end_idx_30 * scale))

    delta_start = det_start_idx - real_start_idx
    delta_end = det_end_idx - real_end_idx

    # START:
    # detecção antes do real = FP_start
    # detecção depois do real = FN_start
    FP_start = abs(delta_start) if delta_start < 0 else 0
    FN_start = delta_start if delta_start > 0 else 0

    # END:
    # detecção depois do real = FP_end
    # detecção antes do real = FN_end
    FP_end = delta_end if delta_end > 0 else 0
    FN_end = abs(delta_end) if delta_end < 0 else 0

    frame_diff = FP_start + FN_start + FP_end + FN_end

    REAL_PIP_FRAMES = 232
    error_pct = (frame_diff / REAL_PIP_FRAMES) * 100.0

    rows2.append({
        "video_file": vf,
        "start_detect": det_start_tc,
        "start_real": start_real_tc,
        "FP_start": FP_start,
        "FN_start": FN_start,
        "end_detect": det_end_tc,
        "end_real": end_real_tc,
        "FP_end": FP_end,
        "FN_end": FN_end,
        "frame_diff": frame_diff,
        "error_pct": error_pct
    })

df2 = pd.DataFrame(rows2)

out2 = os.path.join(OUT_DIR, "frame_fp_fn_yolo26_adjusted.xlsx")
df2.to_excel(out2, index=False)

print(f"✅ Planilha #2 salva em: {out2}")

print("\n✅ Inferência YOLO26 no conjunto de testes concluída.")