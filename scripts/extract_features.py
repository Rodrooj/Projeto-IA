import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import multiprocessing
from tqdm import tqdm
from common import (
    get_base_dir, get_shoulder_center, extract_and_normalize_spatial,
    pad_sequence, FRAMES_PER_SEQUENCE, FEATURES_PER_FRAME
)

mp_holistic = mp.solutions.holistic

# Global per-worker holistic instance
global_holistic = None

def worker_init():
    """
    Inicializa a instância global do MediaPipe Holistic para o worker atual.
    
    Como o MediaPipe pode consumir muita memória e vazar em ambientes de processamento 
    maciçamente paralelo, instanciá-lo apenas uma vez por worker (processo filho) 
    é uma boa prática de gestão de recursos do SO.
    """
    global global_holistic
    global_holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    )

def process_video(video_path, output_path):
    if os.path.exists(output_path):
        return True # Already processed
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
        
    global global_holistic
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return False
        
    # Determine which frames to sample (amostragem uniforme)
    if total_frames >= FRAMES_PER_SEQUENCE:
        indices = np.linspace(0, total_frames - 1, FRAMES_PER_SEQUENCE, dtype=int)
    else:
        indices = np.arange(total_frames)
        
    frames_features = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        results = global_holistic.process(image)
        
        if results.pose_landmarks:
            features = extract_and_normalize_spatial(results)
            frames_features.append(features)
            
    cap.release()
    
    if len(frames_features) == 0:
        return False
    
    # Padding consistente: repete o último frame (melhor para dados gestuais)
    final_sequence = pad_sequence(frames_features, pad_mode='repeat_last')
            
    np.save(output_path, final_sequence)
    return True

def worker(item):
    idx, row, dataset_dir, output_dir = item
    video_name = row['video_name']
    
    video_path = os.path.join(dataset_dir, 'data', video_name)
    if not os.path.exists(video_path):
        return False, f"Not found: {video_name}"
        
    out_name = video_name.replace('.mp4', '.npy')
    output_path = os.path.join(output_dir, out_name)
    
    success = process_video(video_path, output_path)
    return success, video_name

def main():
    """
    Ponto de entrada central.
    Responsável por mapear o arquivo `annotations.csv` do dataset V-LIBRASIL, 
    gerar tarefas para cada vídeo e orquestrar a extração em paralelo utilizando 
    todos os núcleos lógicos (`multiprocessing.Pool`) da máquina.
    """
    base_dir = get_base_dir()
    dataset_dir = os.path.join(base_dir, "datasets", "V-LIBRASIL Dataset")
    output_dir = os.path.join(base_dir, "datasets", "features")
    
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(os.path.join(dataset_dir, 'annotations.csv'))
    print(f"Total entries in CSV: {len(df)}")
    
    tasks = []
    for idx, row in df.iterrows():
        tasks.append((idx, row, dataset_dir, output_dir))
        
    print(f"Processing {len(tasks)} videos using multiprocessing.Pool (maxtasksperchild=10)...")
    success_count = 0
    
    with multiprocessing.Pool(processes=os.cpu_count() or 4, initializer=worker_init, maxtasksperchild=10) as pool:
        for success, msg in tqdm(pool.imap_unordered(worker, tasks), total=len(tasks)):
            if success:
                success_count += 1
                
    print(f"Finished processing! Successfully extracted features for {success_count}/{len(tasks)} videos.")

if __name__ == "__main__":
    main()
