import os
import tensorflow as tf
from pathlib import Path
import numpy as np

MODEL_DIR = Path("../models")
PROCESSED_DIR = Path("../processed_data")

# Parâmetros
FRAMES = 30
FEATURES = 159

def representative_dataset_gen():
    """Gera um dataset representativo para calibração da quantização INT8"""
    from glob import glob
    # Pega uma amostra de 100 sequencias para calibração
    search_pattern = str(PROCESSED_DIR / "**" / "*.npy")
    npy_files = glob(search_pattern, recursive=True)[:100]
    
    for file_path in npy_files:
        data = np.load(file_path)
        if data.shape == (FRAMES, FEATURES):
            # Formato de entrada esperado pela rede: (batch_size, frames, features)
            # Para TFLite o cast pra float32 é recomendado
            data = data.astype(np.float32)
            data = np.expand_dims(data, axis=0) 
            yield [data]

def main():
    model_path = MODEL_DIR / "best_model.h5"
    if not model_path.exists():
        print(f"Modelo não encontrado: {model_path}")
        return
        
    print(f"Carregando modelo {model_path}...")
    model = tf.keras.models.load_model(str(model_path))
    
    print("Iniciando conversão para TFLite com quantização...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Otimizações gerais de quantização
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Dataset representativo para calibração de pesos INT8
    # converter.representative_dataset = representative_dataset_gen
    
    # Forçar operações para INT8 se necessário, mas para LSTM pode ser complicado.
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.inference_input_type = tf.int8
    # converter.inference_output_type = tf.int8
    
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
