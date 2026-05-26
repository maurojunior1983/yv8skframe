# ===========================
# IMPORTS
# ===========================
import os
import numpy as np
print(np.__version__)

import pandas as pd
from ultralytics import YOLO
import torch
import cv2
from collections import deque

print("Tudo OK")


# ===========================
# CONVERSÃO PARA TIMECODE DROP-FRAME
# ===========================
def frame_to_dropframe_tc(frame_number, fps):
    """
    Converte número de frame para timecode drop-frame no padrão HH:MM:SS;FF.
    Adequado para vídeos ~29.97 fps, usando base nominal de 30 fps.
    """

    nominal_fps = 30
    drop_frames = 2

    frames_per_hour = nominal_fps * 60 * 60
    frames_per_24_hours = frames_per_hour * 24
    frames_per_10_minutes = nominal_fps * 60 * 10 - drop_frames * 9
    frames_per_minute = nominal_fps * 60 - drop_frames

    # Ajusta frame para ciclo de 24h
    frame_number = int(round(frame_number))
    frame_number = frame_number % frames_per_24_hours

    d = frame_number // frames_per_10_minutes
    m = frame_number % frames_per_10_minutes

    # Correção drop-frame
    frame_number += drop_frames * 9 * d
    if m >= drop_frames:
        frame_number += drop_frames * ((m - drop_frames) // frames_per_minute)

    hours = frame_number // frames_per_hour
    frame_number %= frames_per_hour

    minutes = frame_number // (nominal_fps * 60)
    frame_number %= nominal_fps * 60

    seconds = frame_number // nominal_fps
    frames = frame_number % nominal_fps

    return f"{hours:02d}:{minutes:02d}:{seconds:02d};{frames:02d}"


# ===========================
# PROCESSAMENTO DO VÍDEO - YOLO26
# ===========================
def process_video_for_PiP(
    video_path,
    model,
    threshold_entry=0.40,
    threshold_exit=0.95,
    min_frames_entry=3,
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

        # -----------------------------
        # Maior confiança da classe PiP
        # Classe 0 = PiP / propaganda
        # -----------------------------
        for result in results:
            boxes = result.boxes

            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())

                if cls_id == 0:
                    if conf > max_conf:
                        max_conf = conf

        # -----------------------------
        # Suavização temporal
        # -----------------------------
        conf_buffer.append(max_conf)
        smooth_conf = sum(conf_buffer) / len(conf_buffer)

        detected = False

        # -----------------------------
        # Máquina de estados
        # -----------------------------
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

        # -----------------------------
        # Timecode simples MM:SS.FF
        # -----------------------------
        time_sec = frame_idx / fps if fps > 0 else 0
        minutes = int(time_sec // 60)
        seconds = int(time_sec % 60)
        frames_in_sec = int(round((time_sec - int(time_sec)) * fps))
        timecode = f"{minutes:02d}:{seconds:02d}.{frames_in_sec:02d}"

        frame_detections.append((frame_idx, detected, timecode, max_conf, smooth_conf))

        frame_idx += 1

    cap.release()

    return frame_detections, fps


# ===========================
# PÓS-PROCESSAMENTO DOS SEGMENTOS
# ===========================
def get_detected_segments(
    detections,
    fps,
    start_frame_offset=0,
    end_frame_offset=0,
    min_duration_sec=2.0,
    min_gap_sec=2.0,
    print_segments=True
):
    segments = []
    in_segment = False
    start_idx = None

    # -----------------------------
    # 1. Criar segmentos brutos
    # -----------------------------
    for i, detection in enumerate(detections):
        idx = detection[0]
        detected = detection[1]

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
# MAIN - INFERÊNCIA YOLO26
# ==========================================

# Peso treinado do YOLO26
model_path = r"C:/Fr_DataSet_S_K_Frame/training_results/yolo26n_Soccer-Key-Frames/weights/best.pt"

# Diretório com vídeos de validação
valid_dir = r"C:\Vi_DataSet_S_K_Frame\valid"

# Carregar modelo YOLO26 treinado
model = YOLO(model_path)

# Arquivos do diretório de validação
arquivos = os.listdir(valid_dir)

# Base real para comparação posterior, se necessário
FRAME_BASE_REAL = 30

# Dicionário para armazenar resultados
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


# ===========================
# Loop de inferência nos vídeos
# ===========================
for video_file in arquivos:

    if video_file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):

        video_path = os.path.join(valid_dir, video_file)

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
            print_segments=True
        )

        segments_tc = [
            (
                frame_to_dropframe_tc(start, fps),
                frame_to_dropframe_tc(end, fps)
            )
            for start, end in final_segments
        ]

        results[video_file] = {
            "fps": fps,
            "segments_frames": final_segments,
            "segments_tc": segments_tc
        }


print("\n✅ Inferência YOLO26 concluída!")