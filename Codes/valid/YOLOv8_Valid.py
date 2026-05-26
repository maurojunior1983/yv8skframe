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
# PROCESSAMENTO DO VÍDEO (COM MELHORIAS)
# ===========================
def process_video_for_PiP(
    video_path,
    model,
    threshold_entry=0.90,
    threshold_exit=0.98,
    min_frames_entry=15,
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
        # Pegando maior confiança da classe PiP
        # -----------------------------
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                if box.cls.item() == 0:
                    conf = box.conf.item()
                    if conf > max_conf:
                        max_conf = conf

        # -----------------------------
        # Suavização temporal
        # -----------------------------
        conf_buffer.append(max_conf)
        smooth_conf = sum(conf_buffer) / len(conf_buffer)

        detected = False

        # -----------------------------
        # MÁQUINA DE ESTADOS
        # -----------------------------
        if state == "NO_PIP":
            if smooth_conf >= threshold_entry:
                entry_counter += 1
                if entry_counter >= min_frames_entry:
                    state = "IN_PIP"
                    detected = True
            else:
                entry_counter = 0

        elif state == "IN_PIP":
            detected = True
            if smooth_conf < threshold_exit:
                exit_counter += 1
                if exit_counter >= min_frames_exit:
                    state = "NO_PIP"
                    detected = False
            else:
                exit_counter = 0

        # -----------------------------
        # TIMECODE
        # -----------------------------
        time_sec = frame_idx / fps if fps > 0 else 0
        minutes = int(time_sec // 60)
        seconds = int(time_sec % 60)
        frames_in_sec = int(round((time_sec - int(time_sec)) * fps))
        timecode = f"{minutes:02d}:{seconds:02d}.{frames_in_sec:02d}"

        frame_detections.append((frame_idx, detected, timecode))

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
    print_segments=True,
):
    segments = []
    in_segment = False

    # 1) Criar segmentos brutos com offsets
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

    # 2) Juntar segmentos próximos
    merged_segments = []
    current_start, current_end = segments[0]

    for next_start, next_end in segments[1:]:
        gap_frames = next_start - current_end
        gap_sec = gap_frames / fps

        if gap_sec < min_gap_sec:
            current_end = next_end
        else:
            merged_segments.append((current_start, current_end))
            current_start, current_end = next_start, next_end

    merged_segments.append((current_start, current_end))

    # 3) Filtrar por duração mínima
    final_segments = []
    for start, end in merged_segments:
        duration_sec = (end - start) / fps
        if duration_sec >= min_duration_sec:
            final_segments.append((start, end))

    if not final_segments:
        if print_segments:
            print("Without PiP")
        return []

    # 4) Imprimir, se desejado
    if print_segments:
        for start, end in final_segments:
            start_tc = frame_to_dropframe_tc(start, fps)
            end_tc = frame_to_dropframe_tc(end, fps)

            duration_frames = end - start
            duration_tc = frame_to_dropframe_tc(duration_frames, fps)

            print(f"PiP from {start_tc} to {end_tc} - Duration: {duration_tc}")

    return final_segments

    # -----------------------------
    # 3. Filtrar por duração mínima
    # -----------------------------
    final_segments = []
    for start, end in merged_segments:
        duration_sec = (end - start) / fps
        if duration_sec >= min_duration_sec:
            final_segments.append((start, end))

    if not final_segments:
        print("Without PiP")
        return

    # -----------------------------
    # 4. Converter para timecode
    # -----------------------------
    def tc(frame):
        t = frame / fps
        m = int(t // 60)
        s = int(t % 60)
        f = int(round((t - int(t)) * fps))
        return f"{m:02d}:{s:02d}.{f:02d}"

    for start, end in final_segments:
        duration_frames = end - start
        duration_sec = duration_frames / fps

        m = int(duration_sec // 60)
        s = int(duration_sec % 60)
        f = int(round((duration_sec - int(duration_sec)) * fps))
        duration_tc = f"{m:02d}:{s:02d}.{f:02d}"

        print(f"PiP from {tc(start)} to {tc(end)} - Duration: {duration_tc}")


# ==========================================
# MAIN
# ==========================================
model_path = r"C:/Fr_DataSet_S_K_Frame/training_results/yolov8n_Soccer-Key-Frames/weights/best.pt"
test_dir = r"C:\Vi_DataSet_S_K_Frame\test"

model = YOLO(model_path)

arquivos = os.listdir(test_dir)
FRAME_BASE_REAL = 30
results = {}

# parâmetros da Tabela 2
THRESHOLD_ENTRY = 0.90
THRESHOLD_EXIT = 0.98
MIN_FRAMES_ENTRY = 15
MIN_FRAMES_EXIT = 3
SMOOTHING_WINDOW = 5
START_FRAME_OFFSET = 0
END_FRAME_OFFSET = 0
MIN_DURATION_SEC = 2.0
MIN_GAP_SEC = 2.0

for video_file in arquivos:

    if video_file.lower().endswith(('.mp4', '.avi', '.mov')):

        video_path = os.path.join(test_dir, video_file)

        print(f"\nProcessando vídeo: {video_file}")

        detections, fps = process_video_for_PiP(
            video_path,
            model,
            threshold_entry=THRESHOLD_ENTRY,
            threshold_exit=THRESHOLD_EXIT,
            min_frames_entry=MIN_FRAMES_ENTRY,
            min_frames_exit=MIN_FRAMES_EXIT,
            smoothing_window=SMOOTHING_WINDOW
        )

        # usa exatamente o mesmo pós-processamento da tabela
        final_segments = get_detected_segments(
            detections,
            fps,
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
            "fps": fps
        }