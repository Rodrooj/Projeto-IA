"""
⚠️ DEPRECADO: Este script é mantido apenas para re-exportação standalone de modelos .h5.
A exportação TFLite agora é feita automaticamente no final de `train_model.py`.
Use este script apenas se precisar reconverter um modelo .h5 existente com
configurações diferentes de quantização.
"""
import os
import tensorflow as tf
from pathlib import Path
import numpy as np
from common import get_base_dir, FRAMES_PER_SEQUENCE, FEATURES_PER_FRAME

BASE_DIR = get_base_dir()
MODEL_DIR = BASE_DIR / "client" / "public" / "models"
FEATURES_DIR = BASE_DIR / "datasets" / "features"

def representative_dataset_gen():
    """
    Função Geradora (Generator) para a quantização representativa INT8.
    
    A quantização INT8 diminui drasticamente o peso e a latência de inferência
    do modelo convertendo pontos flutuantes de 32-bits (Float32) para Inteiros de 8-bits.
    No entanto, para evitar perda de precisão absurda, a rede neural precisa de uma amostra
    representativa dos dados (algumas sequências do dataset original) para mapear corretamente 
    a distribuição de ativações (min e max) em sua nova faixa comprimida de 8-bits.
    
    Yields:
        list: Uma sequência calibradora contendo tensores do tamanho da entrada do modelo.
    """
    from glob import glob
    # Pega uma amostra de 100 sequencias para calibração
    search_pattern = str(FEATURES_DIR / "**" / "*.npy")
    npy_files = [f for f in glob(search_pattern, recursive=True) if "_aug_" not in f][:100]
    
    for file_path in npy_files:
        data = np.load(file_path)
        if data.shape == (FRAMES_PER_SEQUENCE, FEATURES_PER_FRAME):
            # Formato de entrada esperado pela rede: (batch_size, frames, features)
            data = data.astype(np.float32)
            data = np.expand_dims(data, axis=0) 
            yield [data]

def main():
    """
    Ponto de entrada do script de re-exportação standalone.
    
    Carrega um modelo Keras (.h5) salvo e converte para TensorFlow Lite (.tflite).
    Aplica Dynamic Range Quantization via `tf.lite.Optimize.DEFAULT`.
    
    NOTA: A exportação principal agora é feita em `train_model.py` ao final do treinamento.
    """
    model_path = MODEL_DIR / "best_model.h5"
    if not model_path.exists():
        print(f"Modelo não encontrado: {model_path}")
        print("Dica: O modelo é salvo automaticamente por train_model.py em client/public/models/")
        return
        
    print(f"Carregando modelo {model_path}...")
    model = tf.keras.models.load_model(str(model_path))
    
    print("Iniciando conversão para TFLite com quantização...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Otimizações gerais de quantização
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float32]
    
    # Dataset representativo para calibração de pesos INT8 (desabilitado por padrão)
    # converter.representative_dataset = representative_dataset_gen
    
    try:
        tflite_model = converter.convert()
        tflite_path = MODEL_DIR / "libras_model.tflite"
        
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
            
        print(f"Modelo quantizado salvo com sucesso: {tflite_path}")
        
    except Exception as e:
        print(f"Erro durante a conversão: {e}")
        print("Tentando conversão sem strict int8 ops...")
        
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        tflite_model = converter.convert()
        tflite_path = MODEL_DIR / "libras_model.tflite"
        
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
            
        print(f"Modelo fallback salvo com sucesso: {tflite_path}")

if __name__ == "__main__":
    main()
