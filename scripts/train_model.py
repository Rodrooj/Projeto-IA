import os
import json
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, SpatialDropout1D, GaussianNoise
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report

def main():
    base_dir = r"c:\Users\Rodrigo\Downloads\Tradutor-Libras"
    dataset_dir = os.path.join(base_dir, "datasets", "V-LIBRASIL Dataset")
    features_dir = os.path.join(base_dir, "datasets", "features")
    models_dir = os.path.join(base_dir, "client", "public", "models")
    
    os.makedirs(models_dir, exist_ok=True)
    
    print("Loading V-LIBRASIL annotations...")
    valid_rows = []
    class_counts = {}
    
    with open(os.path.join(dataset_dir, 'annotations.csv'), 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            npy_path = os.path.join(features_dir, row['video_name'].replace('.mp4', '.npy'))
            if os.path.exists(npy_path):
                valid_rows.append(row)
                cls = row['class']
                class_counts[cls] = class_counts.get(cls, 0) + 1
            
    print(f"Found {len(valid_rows)} videos with extracted features.")
    
    if len(valid_rows) == 0:
        print("No video features found. Did extract_features.py run?")
        return
    
    # Select only the top 10 most frequent classes
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    top_10_classes = [c[0] for c in sorted_classes[:10]]
    
    # Filter rows
    df_valid = [r for r in valid_rows if r['class'] in top_10_classes]
    
    print(f"Selected top 10 video classes")

    # Create Class Mapping
    all_classes = sorted(list(set(top_10_classes)))
    class_map = {name: idx for idx, name in enumerate(all_classes)}
    reverse_class_map = {idx: name for idx, name in enumerate(all_classes)}
    
    with open(os.path.join(models_dir, 'class_mapping.json'), 'w', encoding='utf-8') as f:
        json.dump(reverse_class_map, f, ensure_ascii=False, indent=2)
        
    num_classes = len(all_classes)
    print(f"Total unique classes: {num_classes}")
    
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    print("Loading Video feature arrays...")
    for row in df_valid:
        npy_path = os.path.join(features_dir, row['video_name'].replace('.mp4', '.npy'))
        features = np.load(npy_path)
        label = class_map[row['class']]
        
        if row['user_id'] == 'Articulador3':
            X_test.append(features)
            y_test.append(label)
        else:
            X_train.append(features)
            y_train.append(label)
            
            # Load augmented data
            noisy_path = npy_path.replace('.npy', '_aug_noise.npy')
            if os.path.exists(noisy_path):
                X_train.append(np.load(noisy_path))
                y_train.append(label)
                
            scaled_path = npy_path.replace('.npy', '_aug_scale.npy')
            if os.path.exists(scaled_path):
                X_train.append(np.load(scaled_path))
                y_train.append(label)
            
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    if len(X_train) == 0:
        print("No training data found!")
        return
        
    print(f"Final Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Final Test shapes: X={X_test.shape}, y={y_test.shape}")
    
    # Model Definition & Training
    strategy = tf.distribute.MirroredStrategy() if len(tf.config.list_physical_devices('GPU')) > 1 else tf.distribute.get_strategy()
    
    with strategy.scope():
        model = Sequential([
            tf.keras.layers.Input(batch_shape=(1, 30, 159)),
            Bidirectional(LSTM(64, return_sequences=False, activation='relu', unroll=True)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=30, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7)
    ]
    
    # Shuffle training data
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32).prefetch(tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32).prefetch(tf.data.AUTOTUNE)

    print("Starting training...")
    model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=150,
        callbacks=callbacks
    )
    
    print("Evaluating on test set...")
    if len(X_test) > 0:
        y_pred_probs = model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        unique_labels_test = np.unique(y_test)
        target_names = [reverse_class_map[i] for i in unique_labels_test]
        print(classification_report(y_test, y_pred, target_names=target_names, labels=unique_labels_test))
    
    # Convert to TFLite
    print("Converting model to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float32]
    tflite_model = converter.convert()
    
    tflite_path = os.path.join(models_dir, 'libras_model.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
        
    print(f"TFLite model saved to {tflite_path}!")

if __name__ == "__main__":
    main()
