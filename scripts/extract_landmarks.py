import os
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from common import (
    get_base_dir, extract_and_normalize_spatial, pad_sequence,
    FRAMES_PER_SEQUENCE, FEATURES_PER_FRAME
)

# Configurações — usa caminhos relativos ao projeto
BASE_DIR = get_base_dir()
DATASETS_ROOT = BASE_DIR / "datasets"
OUTPUT_DIR = BASE_DIR / "processed_data"

# MediaPipe
mp_holistic = mp.solutions.holistic

def process_video(video_path, output_path):
    """
    Processa um único vídeo do começo ao fim:
    1. Amostra frames uniformemente via OpenCV (ao invés de ler todos os frames).
    2. Aplica MediaPipe Holistic.
    3. Normaliza e empacota os 159 atributos temporais via common.py.
    4. Limita/padroniza as informações numa janela temporal de 30 frames cravados.
    5. Salva o pacote em disco via numpy (.npy).
    """
    try:
        if os.path.exists(output_path):
            return f"Já processado: {video_path}"
            
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return f"Erro ao abrir vídeo: {video_path}"
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return f"Vídeo sem frames: {video_path}"
        
        # Amostragem uniforme ao invés de ler todos os frames (consistente com extract_features.py)
        if total_frames >= FRAMES_PER_SEQUENCE:
            sample_indices = np.linspace(0, total_frames - 1, FRAMES_PER_SEQUENCE, dtype=int)
        else:
            sample_indices = np.arange(total_frames)
            
        frames_features = []
        
        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
            
            for frame_idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                    
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                
                if results.pose_landmarks:
                    features = extract_and_normalize_spatial(results)
                    frames_features.append(features)
                
        cap.release()
        
        if len(frames_features) == 0:
            return f"Nenhum frame extraído: {video_path}"
        
        # Padding consistente (repeat_last) via common.py
        final_data = pad_sequence(frames_features, pad_mode='repeat_last')
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, final_data)
        return f"Sucesso: {video_path}"
        
    except Exception as e:
        return f"Erro em {video_path}: {str(e)}"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    video_paths = []
    
    # Varredura de vídeos
    search_pattern = str(DATASETS_ROOT / "**" / "*.mp4")
    
    for path in glob(search_pattern, recursive=True):
        # Ignorar Healthcare Corpus inteiramente conforme solicitado
        if "Healthcare Corpus Dataset" in path:
            continue
            
        video_paths.append(path)
        
    print(f"Total de vídeos encontrados para extração: {len(video_paths)}")
    
    tasks = []
    for vp in video_paths:
        vp_obj = Path(vp)
        # Tentar extrair a classe como a pasta pai do vídeo
        label = vp_obj.parent.name
        
        out_path = OUTPUT_DIR / label / f"{vp_obj.stem}.npy"
        tasks.append((vp, out_path))
        
    # Usando ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_video, p, o) for p, o in tasks]
        for idx, f in enumerate(futures):
            res = f.result()
            if idx % 50 == 0:
                print(f"[{idx}/{len(tasks)}] {res}")

if __name__ == "__main__":
    main()
