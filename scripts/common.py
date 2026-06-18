"""
Módulo compartilhado de Engenharia de Atributos para LIBRAS.

Centraliza constantes e funções de extração/normalização de landmarks
utilizadas por todos os scripts de processamento de dados, evitando
duplicação de código entre extract_features.py, extract_landmarks.py,
e extract_features_photos.py.
"""
import numpy as np
from pathlib import Path


# ─── Constantes do Pipeline ───────────────────────────────────────────────────

FRAMES_PER_SEQUENCE = 30
"""Quantidade de frames (janela temporal) para a rede LSTM."""

FEATURES_PER_FRAME = 159
"""53 landmarks × 3 coordenadas (x, y, z) = 159 atributos por frame."""

POSE_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 19, 20, 21]
"""Índices dos 11 landmarks de pose relevantes para LIBRAS (tronco superior)."""


# ─── Funções Utilitárias ──────────────────────────────────────────────────────

def get_base_dir() -> Path:
    """
    Retorna o diretório raiz do projeto usando caminho relativo ao script.
    Substitui caminhos absolutos hardcoded (c:\\Users\\Rodrigo\\...) para
    portabilidade entre máquinas e ambientes CI.
    """
    return Path(__file__).resolve().parent.parent


# ─── Engenharia de Atributos ─────────────────────────────────────────────────

def get_shoulder_center(pose_landmarks):
    """
    Calcula o baricentro (ponto central) entre o ombro esquerdo e direito.
    Funciona como a "âncora" para normalizar todas as demais coordenadas,
    garantindo invariância à translação.
    
    Args:
        pose_landmarks: Objeto de landmarks da pose retornado pelo MediaPipe.
                        Deve ter atributo `.landmark` com pelo menos 13 elementos.
    
    Returns:
        dict com coordenadas (x, y, z) do centro dos ombros.
    """
    if not pose_landmarks or len(pose_landmarks.landmark) <= 12:
        return {'x': 0.0, 'y': 0.0, 'z': 0.0}
    
    l_shoulder = pose_landmarks.landmark[11]
    r_shoulder = pose_landmarks.landmark[12]
    
    return {
        'x': (l_shoulder.x + r_shoulder.x) / 2.0,
        'y': (l_shoulder.y + r_shoulder.y) / 2.0,
        'z': (l_shoulder.z + r_shoulder.z) / 2.0,
    }


def extract_and_normalize_spatial(results):
    """
    Extrai e normaliza espacialmente os 53 landmarks (159 valores no total).
    
    Recebe o objeto bruto do MediaPipe Holistic e gera um vetor numérico
    unidimensional focado nos pontos relevantes para LIBRAS, transladado
    para usar o centro dos ombros como origem (0,0,0).
    
    Layout do vetor de saída:
    - [0:33]   → 11 pontos do tronco superior × 3 coords
    - [33:96]  → 21 pontos da mão esquerda × 3 coords
    - [96:159] → 21 pontos da mão direita × 3 coords
    
    Args:
        results: Objeto Results do MediaPipe Holistic.
    
    Returns:
        list de 159 floats normalizados.
    """
    features = []
    origin = get_shoulder_center(results.pose_landmarks)
    
    # 1. Pose (11 pontos do tronco superior)
    if results.pose_landmarks:
        for idx in POSE_INDICES:
            if idx < len(results.pose_landmarks.landmark):
                lm = results.pose_landmarks.landmark[idx]
                features.extend([lm.x - origin['x'], lm.y - origin['y'], lm.z - origin['z']])
            else:
                features.extend([0.0, 0.0, 0.0])
    else:
        features.extend([0.0] * (11 * 3))
        
    # 2. Mão Esquerda (21 pontos)
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            features.extend([lm.x - origin['x'], lm.y - origin['y'], lm.z - origin['z']])
    else:
        features.extend([0.0] * (21 * 3))
        
    # 3. Mão Direita (21 pontos)
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            features.extend([lm.x - origin['x'], lm.y - origin['y'], lm.z - origin['z']])
    else:
        features.extend([0.0] * (21 * 3))
        
    return features


def pad_sequence(frames_features, pad_mode='repeat_last'):
    """
    Ajusta uma sequência para ter exatamente FRAMES_PER_SEQUENCE frames.
    
    Se a sequência for mais longa, amostra frames uniformemente (subsampling).
    Se for mais curta, faz padding conforme o modo escolhido.
    
    Padronizado para usar 'repeat_last' em todo o pipeline, garantindo
    consistência entre extract_features.py e extract_landmarks.py.
    
    Args:
        frames_features: Lista ou np.ndarray de feature vectors com shape (N, 159).
        pad_mode: Estratégia de padding:
                  - 'repeat_last': repete o último frame (padrão, melhor para gestos)
                  - 'zeros': preenche com zeros
    
    Returns:
        np.ndarray com shape (FRAMES_PER_SEQUENCE, FEATURES_PER_FRAME) em float32.
    """
    frames = np.array(frames_features, dtype=np.float32) if not isinstance(frames_features, np.ndarray) else frames_features.astype(np.float32)
    
    if len(frames) == 0:
        return np.zeros((FRAMES_PER_SEQUENCE, FEATURES_PER_FRAME), dtype=np.float32)
    
    if len(frames) > FRAMES_PER_SEQUENCE:
        # Amostragem uniforme para manter distribuição temporal
        indices = np.linspace(0, len(frames) - 1, FRAMES_PER_SEQUENCE, dtype=int)
        frames = frames[indices]
    elif len(frames) < FRAMES_PER_SEQUENCE:
        deficit = FRAMES_PER_SEQUENCE - len(frames)
        if pad_mode == 'repeat_last':
            padding = np.tile(frames[-1], (deficit, 1))
        else:
            padding = np.zeros((deficit, FEATURES_PER_FRAME), dtype=np.float32)
        frames = np.vstack((frames, padding))
    
    return frames
