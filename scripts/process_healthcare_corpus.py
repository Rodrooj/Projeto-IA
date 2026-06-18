import os
import csv
from moviepy import VideoFileClip
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from common import get_base_dir

# Diretórios — usa caminhos relativos ao projeto
BASE_DIR = get_base_dir()
DATASET_DIR = BASE_DIR / "datasets" / "Healthcare Corpus Dataset"
VIDEOS_DIR = DATASET_DIR / "videos"
ANNOTATIONS_DIR = DATASET_DIR / "annotations"
OUTPUT_DIR = DATASET_DIR / "extracted_clips"

def process_segment(row):
    """
    Função de trabalhador (Worker) para cortar clipes individuais a partir de um vídeo longo.
    
    Ele lê os timestamps de início (`start_time`) e fim (`end_time`) contidos na
    planilha CSV de anotações (oriundas do corpus clínico).
    Usando a biblioteca `moviepy`, realiza o "Trim" do vídeo longo e exporta um MP4 mais curto
    comprimido via `libx264` agrupado em pastas conforme seu significado (`target_gloss`).
    """
    try:
        segment_id = row['segment_id']
        video_file = row['video_file']
        start_time = float(row['start_time'])
        end_time = float(row['end_time'])
        
        # Obter classe
        target_gloss = row['target_gloss'].strip()
        produced_gloss = row['produced_gloss'].strip()
        
        # Priorizar target_gloss, fallback para produced_gloss
        label = target_gloss if target_gloss else produced_gloss
        
        # Se não houver classe clara, ignorar ou classificar como 'open_prompt'
        if not label:
            return f"Ignorado {segment_id}: Sem label"
            
        # Remover caracteres inválidos do nome da pasta
        label_safe = "".join([c for c in label if c.isalpha() or c.isdigit() or c in (' ', '-', '_')]).strip()
        
        video_path = VIDEOS_DIR / video_file
        if not video_path.exists():
            return f"Erro {segment_id}: Vídeo {video_file} não encontrado"
            
        class_dir = OUTPUT_DIR / label_safe
        class_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = class_dir / f"{segment_id}.mp4"
        
        # Pular se já existir
        if output_path.exists():
            return f"Já existe {segment_id}"
            
        # Recortar o vídeo (moviepy)
        # Desabilitar áudio para acelerar se não for necessário
        with VideoFileClip(str(video_path)) as video:
            # Validar limites de tempo
            if start_time < 0: start_time = 0
            if end_time > video.duration: end_time = video.duration
                
            clip = video.subclip(start_time, end_time)
            # fps=video.fps ou algo padronizado, codec="libx264"
            clip.write_videofile(str(output_path), codec="libx264", audio=False, verbose=False, logger=None)
            
        return f"Sucesso {segment_id}: {label_safe}"
    except Exception as e:
        return f"Erro crítico {row.get('segment_id', 'Unknown')}: {e}"

def main():
    """
    Ponto de entrada do script de particionamento do Healthcare Corpus.
    
    Diferente de datasets já picotados (como o V-LIBRASIL), o corpus clínico vem 
    em vídeos inteiros de consulta (longos). Este script lê o dicionário de tempos 
    (segments.csv) e orquestra workers usando `ProcessPoolExecutor` para extrair
    os mini-clipes relevantes.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = ANNOTATIONS_DIR / "segments.csv"
    
    if not csv_path.exists():
        print(f"Arquivo CSV não encontrado: {csv_path}")
        return
        
    tasks = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tasks.append(row)
            
    print(f"Total de segmentos encontrados: {len(tasks)}")
    
    # Executar em paralelo
    # Reduzindo max_workers para não sobrecarregar I/O
    with ProcessPoolExecutor(max_workers=4) as executor:
        for result in executor.map(process_segment, tasks):
            if "Erro" in result:
                print(result)

if __name__ == "__main__":
    main()
