import os
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from common import get_base_dir, FRAMES_PER_SEQUENCE, FEATURES_PER_FRAME

PROCESSED_DIR = get_base_dir() / "datasets" / "features"


def augment_sequence(data):
    """
    Aplica técnicas matemáticas de Data Augmentation para séries temporais de landmarks.
    
    Técnicas aplicadas:
    1. Ruído Gaussiano: Simula imprecisões e pequenos ruídos naturais gerados pela
       captura e processamento do MediaPipe.
    2. Alteração de Escala (Zoom): Simula diferentes distâncias da pessoa em relação à câmera.
    3. Distorção Temporal: Varia a velocidade do sinal para simular articuladores mais
       rápidos ou mais lentos.
    4. Espelhamento (Mirror): Troca mão esquerda com direita e inverte o eixo X para
       simular sinais feitos por canhotos/destros.
    
    Isso força a rede neural a generalizar melhor, combatendo o overfitting em classes minoritárias.
    
    Args:
        data (np.ndarray): Matriz base extraída no formato (30 frames, 159 features).
        
    Returns:
        dict: Mapeamento {sufixo: dados_aumentados} com todas as variações geradas.
    """
    augmented = {}
    
    # 1. Ruído Gaussiano
    noise = np.random.normal(0, 0.005, data.shape)
    augmented['noise'] = (data + noise).astype(np.float32)
    
    # 2. Alteração de Escala (Zoom in/out de 90% a 110%)
    scale = np.random.uniform(0.9, 1.1)
    augmented['scale'] = (data * scale).astype(np.float32)
    
    # 3. Distorção Temporal — varia a velocidade reamostrado os frames
    n_frames = data.shape[0]
    orig_time = np.linspace(0, 1, n_frames)
    # Adiciona variação aleatória na posição temporal de cada frame
    warp_offsets = np.random.normal(0, 0.1, n_frames)
    warp_offsets[0] = 0   # Mantém início fixo
    warp_offsets[-1] = 0  # Mantém fim fixo
    warped_time = orig_time + warp_offsets
    warped_time = np.sort(warped_time)
    # Normaliza para [0, 1]
    warped_time = (warped_time - warped_time[0]) / (warped_time[-1] - warped_time[0])
    warped_indices = warped_time * (n_frames - 1)
    
    temporal_warped = np.zeros_like(data)
    for feat in range(data.shape[1]):
        temporal_warped[:, feat] = np.interp(warped_indices, np.arange(n_frames), data[:, feat])
    augmented['temporal'] = temporal_warped.astype(np.float32)
    
    # 4. Espelhamento (Mirror) — troca mão esquerda ↔ direita e inverte eixo X
    # Layout do vetor de features:
    # [0:33]   → 11 pontos pose × 3 coords (x,y,z)
    # [33:96]  → 21 pontos mão esquerda × 3 coords
    # [96:159] → 21 pontos mão direita × 3 coords
    mirrored = data.copy()
    
    # Negar todas as coordenadas X (posições 0, 3, 6, ...) para refletir horizontalmente
    for i in range(0, FEATURES_PER_FRAME, 3):
        mirrored[:, i] = -mirrored[:, i]
    
    # Trocar bloco da mão esquerda com mão direita
    left_hand = mirrored[:, 33:96].copy()
    right_hand = mirrored[:, 96:159].copy()
    mirrored[:, 33:96] = right_hand
    mirrored[:, 96:159] = left_hand
    
    # Trocar pares de pose esquerda ↔ direita
    # Pares no vetor: (0-2)↔(3-5) ombros, (6-8)↔(9-11) cotovelos,
    #                 (12-14)↔(15-17) pulsos, (18-20)↔(21-23) quadris,
    #                 (24-26)↔(27-29) dedos indicadores
    pose_swap_pairs = [(0, 3), (6, 9), (12, 15), (18, 21), (24, 27)]
    for left_start, right_start in pose_swap_pairs:
        left = mirrored[:, left_start:left_start + 3].copy()
        right = mirrored[:, right_start:right_start + 3].copy()
        mirrored[:, left_start:left_start + 3] = right
        mirrored[:, right_start:right_start + 3] = left
    
    augmented['mirror'] = mirrored.astype(np.float32)
    
    return augmented


def process_augmentation(npy_path):
    """
    Função de trabalhador (Worker) executada em paralelo para processar a "Aumentação" 
    de um único arquivo `.npy` (que representa um vídeo).
    
    Ele lê os dados originais, aplica a função `augment_sequence` gerando as variações 
    (ruído, escala, temporal, mirror), e salva fisicamente as novas amostras no disco.
    
    Args:
        npy_path (str): Caminho absoluto/relativo para o arquivo matriz .npy original.
        
    Returns:
        str: Mensagem de status confirmando o sucesso ou o motivo da falha.
    """
    try:
        data = np.load(npy_path)
        
        # Ignorar se já foi augumentado (evitar recursão infinita caso o script rode múltiplas vezes)
        if "_aug_" in npy_path:
            return "Ignorado (já é arquivo aug)"
            
        augmented = augment_sequence(data)
        
        path_obj = Path(npy_path)
        saved = 0
        
        for suffix, aug_data in augmented.items():
            aug_path = path_obj.with_name(f"{path_obj.stem}_aug_{suffix}.npy")
            if not aug_path.exists():
                np.save(str(aug_path), aug_data)
                saved += 1
            
        return f"Augmented ({saved} novos): {npy_path}"
    except Exception as e:
        return f"Erro em {npy_path}: {e}"


def main():
    """
    Ponto de entrada do script de Augmentation.
    
    Busca de forma recursiva todos os arquivos de landmarks `.npy` originais 
    (ignorando aqueles que já sofreram augmentation prévia).
    Em seguida, despacha o processamento em um pool de múltiplos núcleos da CPU
    usando `ProcessPoolExecutor` para acelerar exponencialmente a execução do processo.
    """
    search_pattern = str(PROCESSED_DIR / "**" / "*.npy")
    npy_files = [f for f in glob(search_pattern, recursive=True) if "_aug_" not in f]
    
    print(f"Total de arquivos base para augmentation: {len(npy_files)}")
    print(f"Estratégias: ruído gaussiano, escala, distorção temporal, espelhamento (mirror)")
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        for idx, result in enumerate(executor.map(process_augmentation, npy_files)):
            if idx % 50 == 0:
                print(f"[{idx}/{len(npy_files)}] {result}")

if __name__ == "__main__":
    main()
