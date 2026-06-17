import os
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from glob import glob

PROCESSED_DIR = Path(r"c:\Users\Rodrigo\Downloads\Tradutor-Libras\datasets\features")

def augment_sequence(data):
    """
    Aplica técnicas matemáticas de Data Augmentation para séries temporais de landmarks.
    
    Técnicas aplicadas:
    1. Ruído Gaussiano: Simula imprecisões e pequenos ruídos naturais gerados pela
       captura e processamento do MediaPipe.
    2. Alteração de Escala (Zoom): Simula diferentes distâncias da pessoa em relação à câmera.
    
    Isso força a rede neural a generalizar melhor, combatendo o overfitting em classes minoritárias.
    
    Args:
        data (np.ndarray): Matriz base extraída no formato (30 frames, 159 features).
        
    Returns:
        tuple: (data_noisy, data_scaled) contendo duas novas variações do dado original.
    """
    # 1. Ruído Gaussiano
    noise = np.random.normal(0, 0.005, data.shape)
    data_noisy = data + noise
    
    # 2. Alteração de Escala (Zoom in/out de 90% a 110%)
    scale = np.random.uniform(0.9, 1.1)
    data_scaled = data * scale
    
    return data_noisy, data_scaled

def process_augmentation(npy_path):
    """
    Função de trabalhador (Worker) executada em paralelo para processar a "Aumentação" 
    de um único arquivo `.npy` (que representa um vídeo).
    
    Ele lê os dados originais, aplica a função `augment_sequence` gerando os cenários 
    ruidosos e escalados, e salva fisicamente as duas novas amostras no disco.
    
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
            
        data_noisy, data_scaled = augment_sequence(data)
        
        path_obj = Path(npy_path)
        noisy_path = path_obj.with_name(f"{path_obj.stem}_aug_noise.npy")
        scaled_path = path_obj.with_name(f"{path_obj.stem}_aug_scale.npy")
        
        if not noisy_path.exists():
            np.save(str(noisy_path), data_noisy)
        if not scaled_path.exists():
            np.save(str(scaled_path), data_scaled)
            
        return f"Augmented: {npy_path}"
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
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        for idx, result in enumerate(executor.map(process_augmentation, npy_files)):
            if idx % 50 == 0:
                print(f"[{idx}/{len(npy_files)}] {result}")

if __name__ == "__main__":
    main()
