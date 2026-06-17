import cv2
import numpy as np
import os
import glob
import multiprocessing
from tqdm import tqdm

FRAMES_PER_SEQUENCE = 30
FEATURES_PER_FRAME = 159

global_holistic = None

def worker_init():
    global global_holistic
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic
    global_holistic = mp_holistic.Holistic(
        static_image_mode=True,
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

def process_image(image_path, output_path):
    if os.path.exists(output_path):
        return True
        
    image = cv2.imread(image_path)
    if image is None:
        return False
        
    global global_holistic
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = global_holistic.process(image_rgb)
    
    # We only process if hands are detected, to ensure quality data
    if not results.left_hand_landmarks and not results.right_hand_landmarks:
        # If we can't see hands in a photo dataset of letters, it's probably a bad crop or empty
        # but let's just save zeros or skip it. Let's skip.
        return False
        
    features = extract_and_normalize_spatial(results)
    
    # Duplicate features 30 times
    final_sequence = [features] * FRAMES_PER_SEQUENCE
    
    np.save(output_path, np.array(final_sequence, dtype=np.float32))
    return True

def worker(item):
    image_path, output_path = item
    success = process_image(image_path, output_path)
    return success

def main():
    base_dir = r"c:\Users\Rodrigo\Downloads\Tradutor-Libras"
    dataset_dir = os.path.join(base_dir, "datasets", "LIBRAS Photo Dataset")
    output_dir = os.path.join(base_dir, "datasets", "features_photos")
    
    os.makedirs(output_dir, exist_ok=True)
    
    tasks = []
    
    for split in ['train', 'test']:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        for class_dir in os.listdir(split_dir):
            class_path = os.path.join(split_dir, class_dir)
            if not os.path.isdir(class_path):
                continue
                
            # Create corresponding output directory structure
            out_class_dir = os.path.join(output_dir, split, class_dir)
            os.makedirs(out_class_dir, exist_ok=True)
            
            # Find all images (jpg, png)
            for ext in ('*.jpg', '*.png', '*.jpeg'):
                for image_path in glob.glob(os.path.join(class_path, ext)):
                    filename = os.path.basename(image_path)
                    out_name = filename.rsplit('.', 1)[0] + '.npy'
                    output_path = os.path.join(out_class_dir, out_name)
                    tasks.append((image_path, output_path))
                    
    print(f"Total images found: {len(tasks)}")
    
    success_count = 0
    with multiprocessing.Pool(processes=os.cpu_count() or 4, initializer=worker_init) as pool:
        for success in tqdm(pool.imap_unordered(worker, tasks), total=len(tasks)):
            if success:
                success_count += 1
                
    print(f"Finished processing! Successfully extracted features for {success_count}/{len(tasks)} images.")

if __name__ == "__main__":
    main()
