import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import glob
import multiprocessing
from tqdm import tqdm

mp_holistic = mp.solutions.holistic

FRAMES_PER_SEQUENCE = 30
FEATURES_PER_FRAME = 159

# Global per-worker holistic instance
global_holistic = None

def worker_init():
    global global_holistic
    global_holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    )

def get_shoulder_center(pose_landmarks):
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
    features = []
    origin = get_shoulder_center(results.pose_landmarks)
    
    pose_indices = [11, 12, 13, 14, 15, 16, 23, 24, 19, 20, 21]
    if results.pose_landmarks:
        for idx in pose_indices:
            if idx < len(results.pose_landmarks.landmark):
                lm = results.pose_landmarks.landmark[idx]
                features.extend([lm.x - origin['x'], lm.y - origin['y'], lm.z - origin['z']])
            else:
                features.extend([0.0, 0.0, 0.0])
    else:
        features.extend([0.0] * (11 * 3))
        
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            features.extend([lm.x - origin['x'], lm.y - origin['y'], lm.z - origin['z']])
    else:
        features.extend([0.0] * (21 * 3))
        
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            features.extend([lm.x - origin['x'], lm.y - origin['y'], lm.z - origin['z']])
    else:
        features.extend([0.0] * (21 * 3))
        
    return features

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
        
    # Determine which frames to sample
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
    
    seq_len = len(frames_features)
    if seq_len == 0:
        return False
        
    final_sequence = frames_features.copy()
    while len(final_sequence) < FRAMES_PER_SEQUENCE:
        final_sequence.append([0.0] * FEATURES_PER_FRAME)
            
    np.save(output_path, np.array(final_sequence, dtype=np.float32))
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
    base_dir = r"c:\Users\Rodrigo\Downloads\Tradutor-Libras"
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

