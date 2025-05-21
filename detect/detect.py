#python
from ultralytics import YOLO
import torch
import os
import cv2
import numpy as np

def process_video_for_propaganda(video_path, model, detection_threshold=0.93):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_detections = []  # lista de (frame_idx, detected, timecode)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (640, 360))
        results = model(frame_resized, verbose=False)
        detection_found = False
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                cls_id = box.cls.item()
                conf = box.conf.item()
                if cls_id == 0 and conf >= detection_threshold:
                    detection_found = True
                    break
            if detection_found:
                break
        time_sec = frame_idx / fps if fps > 0 else 0
        minutes = int(time_sec // 60)
        seconds = int(time_sec % 60)
        frames_in_sec = int(round((time_sec - int(time_sec)) * fps))
        timecode = f"{minutes:02d}:{seconds:02d}.{frames_in_sec:02d}"
        frame_detections.append((frame_idx, detection_found, timecode))
        frame_idx += 1
    cap.release()
    return frame_detections, fps

def print_detected_segments(detections, fps):
    segments = []  # lista de (start_idx, end_idx, start_time, end_time)
    in_segment = False
    start_idx = end_idx = None
    for idx, detected, timecode in detections:
        if detected and not in_segment:
            in_segment = True
            start_idx = idx
            start_time = timecode
        elif not detected and in_segment:
            in_segment = False
            end_idx = prev_idx
            end_time = prev_timecode
            segments.append((start_idx, end_idx, start_time, end_time))
        prev_idx, prev_timecode = idx, timecode
    # se terminar ainda dentro de um segmento
    if in_segment:
        segments.append((start_idx, prev_idx, start_time, prev_timecode))

    if not segments:
        print("Sem Foguetes")
        return

    for start, end, start_tc, end_tc in segments:
        duration_frames = end - start
        duration_sec = duration_frames / fps
        minutes = int(duration_sec // 60)
        seconds = int(duration_sec % 60)
        frames = int(round((duration_sec - int(duration_sec)) * fps))
        duration_tc = f"{minutes:02d}:{seconds:02d}.{frames:02d}"
        print(f"Foguete de {start_tc} até {end_tc} - Duração: {duration_tc}")

# Configuração do modelo e diretório de testes
model = YOLO("C:/YV8SKFRAME/dataset/training_results/yolov8n_Soccer-Key-Frames13/weights/best.pt")
test_dir = r"C:\YV8SKFRAME\dataset\test"

# Processa cada vídeo na pasta test e exibe segmentos com propaganda
for video_file in os.listdir(test_dir):
    if video_file.lower().endswith(('.mp4', '.avi', '.mov')):
        video_path = os.path.join(test_dir, video_file)
        print(f"\nProcessando vídeo: {video_file}")
        detections, fps = process_video_for_propaganda(video_path, model, detection_threshold=0.93)
        print_detected_segments(detections, fps)
import pandas as pd

# --- Carrega a planilha de timecodes reais ---
meta_df = pd.read_excel(r"C:/YV8SKFRAME/DataSet_SoccerKeyFrame - Org/DataSet_SoccerKeyFrame.xlsx")

# lista temporária para os resultados
rows = []

for video_file in os.listdir(test_dir):
    if not video_file.lower().endswith(('.mp4', '.avi', '.mov')):
        continue

    video_path = os.path.join(test_dir, video_file)
    detections, fps = process_video_for_propaganda(video_path, model, detection_threshold=0.93)

    # Reconstrói segmentos detectados
    segments = []
    in_seg = False
    prev_tc = None
    for idx, detected, tc in detections:
        if detected and not in_seg:
            in_seg = True
            start_tc = tc
        elif not detected and in_seg:
            in_seg = False
            end_tc = prev_tc
            segments.append((start_tc, end_tc))
        prev_tc = tc
    if in_seg:
        segments.append((start_tc, prev_tc))

    # Encontra timecodes reais na planilha
    real_row = meta_df[meta_df['video_path'].str.contains(video_file)]
    if not real_row.empty:
        real_start = real_row.iloc[0]['start_time']
        real_end   = real_row.iloc[0]['end_time']
    else:
        real_start = real_end = None

    # Adiciona linhas à lista
    for det_start, det_end in segments:
        rows.append({
            'video_file'  : video_file,
            'start_detect': det_start,
            'end_detect'  : det_end,
            'start_real'  : real_start,
            'end_real'    : real_end,
            'match'       : (det_start == real_start and det_end == real_end)
        })

# Cria DataFrame e salva
df_results = pd.DataFrame(rows, columns=[
    'video_file','start_detect','end_detect','start_real','end_real','match'
])
out_path = r"C:/YV8SKFRAME/DataSet_SoccerKeyFrame - Org/RESULTADO.xlsx"
df_results.to_excel(out_path, index=False)
print(f"✅ Resultados salvos em: {out_path}")