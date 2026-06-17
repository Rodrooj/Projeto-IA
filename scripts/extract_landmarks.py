import os
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from glob import glob

# Configurações
DATASETS_ROOT = Path("../datasets")
OUTPUT_DIR = Path("../processed_data")
FRAMES_PER_SEQUENCE = 30
FEATURES_PER_FRAME = 159 # 53 landmarks * 3 coords

# MediaPipe
mp_holistic = mp.solutions.holistic

def get_shoulder_center(landmarks):
    """Calcula o ponto médio entre os ombros (landmarks 11 e 12 da pose)"""
    # Em mediapipe holistic, pose_landmarks 11 e 12 são os ombros
    if not landmarks.pose_landmarks:
        return np.array([0.0, 0.0, 0.0])
    
    l_shoulder = landmarks.pose_landmarks.landmark[11]
    r_shoulder = landmarks.pose_landmarks.landmark[12]
    
    cx = (l_shoulder.x + r_shoulder.x) / 2
    cy = (l_shoulder.y + r_shoulder.y) / 2
    cz = (l_shoulder.z + r_shoulder.z) / 2
    
    return np.array([cx, cy, cz])

def extract_features_from_results(results, origin):
    """Extrai e normaliza espacialmente os 53 landmarks (159 valores)"""
    features = []
    
    # 1. Pose (Apenas Tronco Superior: 11 a 24, mas vamos pegar os primeiros 11 principais ou mapear 11 específicos)
    # A doc diz "Tronco Superior: 11 pontos". No mediapipe pose, face=0-10, tronco/braços=11-24. 
    # Usaremos 11 a 21 (ombros, cotovelos, pulsos, etc).
    pose_indices = [11, 12, 13, 14, 15, 16, 23, 24, 19, 20, 21] # 11 pontos
    
    if results.pose_landmarks:
        for idx in pose_indices:
            lm = results.pose_landmarks.landmark[idx]
            features.extend([lm.x - origin[0], lm.y - origin[1], lm.z - origin[2]])
    else:
        features.extend([0.0] * (11 * 3))
        
    # 2. Left Hand (21 pontos)
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            features.extend([lm.x - origin[0], lm.y - origin[1], lm.z - origin[2]])
    else:
        features.extend([0.0] * (21 * 3))
        
    # 3. Right Hand (21 pontos)
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            features.extend([lm.x - origin[0], lm.y - origin[1], lm.z - origin[2]])
    else:
        features.extend([0.0] * (21 * 3))
        
    return np.array(features)

def process_video(video_path, output_path):
    """Processa um vídeo, extrai landmarks, calcula deltas e salva em .npy"""
    try:
        if os.path.exists(output_path):
            return f"Já processado: {video_path}"
            
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return f"Erro ao abrir vídeo: {video_path}"
            
        frames_features = []
        
        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                
                origin = get_shoulder_center(results)
                features = extract_features_from_results(results, origin)
                frames_features.append(features)
                
        cap.release()
        
        if len(frames_features) == 0:
            return f"Nenhum frame extraído: {video_path}"
            
        frames_features = np.array(frames_features)
        
        # Calcular deltas (t - (t-1))
        # O primeiro frame tem delta 0
        deltas = np.zeros_like(frames_features)
        deltas[1:] = frames_features[1:] - frames_features[:-1]
        
        # Combinar posições absolutas normalizadas e deltas (opcional: a documentação foca nos deltas + posições ou apenas os 159 atributos consolidados)
        # Vamos assumir que a rede recebe a posição e possivelmente os deltas se houver mais de 159. 
        # A doc diz "Entrada (30, 159)", então os deltas podem substituir as posições ou serem computados na rede. 
        # A documentação menciona: "Além das posições absolutas, o sistema calcula a diferença... Δ = posição(t) − posição(t−1)". 
        # Para caber em (30, 159), talvez seja só a posição normalizada enviada e o delta calculado internamente, ou são 159 atributos no total que MISTURAM pos e deltas?
        # A conta "53 landmarks × 3 coordenadas = 159 atributos" indica que apenas as posições SÃO os 159 atributos da entrada!
        # O delta pode ser processado localmente no pipeline e enviado no lugar das posições brutas? Não, a doc diz "159 atributos". Vamos armazenar posições normais, o modelo lida com isso ou enviamos concatenado.
        # Por hora, salvaremos as posições.
        
        final_data = frames_features # Shape (N, 159)
        
        # Interpolação/Padding para FRAMES_PER_SEQUENCE (30 frames)
        if len(final_data) > FRAMES_PER_SEQUENCE:
            # Seleciona frames uniformemente
            indices = np.linspace(0, len(final_data)-1, FRAMES_PER_SEQUENCE, dtype=int)
            final_data = final_data[indices]
        elif len(final_data) < FRAMES_PER_SEQUENCE:
            # Padding com o último frame
            padding = np.tile(final_data[-1], (FRAMES_PER_SEQUENCE - len(final_data), 1))
            final_data = np.vstack((final_data, padding))
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, final_data)
        return f"Sucesso: {video_path}"
        
    except Exception as e:
        return f"Erro em {video_path}: {str(e)}"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    video_paths = []
    
    # Varredura de vídeos (exemplo estrutural)
    # Assumindo: datasets/<nome_dataset>/<classe>/<video>.mp4 ou extraídos
    # Healthcare extraído: datasets/Healthcare Corpus Dataset/extracted_clips/<classe>/<video>.mp4
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
