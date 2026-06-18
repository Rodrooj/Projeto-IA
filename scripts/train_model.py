import os
import json
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Bidirectional,
    SpatialDropout1D, GaussianNoise, Conv1D
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from common import get_base_dir, FRAMES_PER_SEQUENCE, FEATURES_PER_FRAME

def main():
    """
    Script de Treinamento da Rede Neural de Tradução de LIBRAS.
    
    Responsabilidades deste script:
    1. Carregar as anotações do dataset e mapear as top 10 classes mais frequentes.
    2. Importar as sequências numpy (`.npy`) extraídas e aplicar as matrizes aumentadas 
       (data augmentation: noise, scale, temporal, mirror).
    3. Separar os dados em Treino e Validação com base no emissor ('Articulador3' vai para teste).
    4. Definir a arquitetura CNN-1D + BiLSTM conforme documentação técnica.
    5. Treinar usando EarlyStopping e class weights para combater desbalanceamento.
    6. Exportar automaticamente o resultado final para `.tflite` para o Frontend React consumir.
    """
    base_dir = get_base_dir()
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
    
    # Sufixos de augmentation reconhecidos pelo pipeline
    AUG_SUFFIXES = ['_aug_noise', '_aug_scale', '_aug_temporal', '_aug_mirror']
    
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
            
            # Load all augmented variants
            for suffix in AUG_SUFFIXES:
                aug_path = npy_path.replace('.npy', f'{suffix}.npy')
                if os.path.exists(aug_path):
                    X_train.append(np.load(aug_path))
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
    
    # Compute class weights para combater desbalanceamento entre classes
    unique_classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
    class_weight_dict = dict(zip(unique_classes.astype(int), class_weights))
    print(f"Class weights: {class_weight_dict}")
    
    # Model Definition & Training
    # Utiliza paralelismo em GPUs (MirroredStrategy) se disponível, caso contrário usa CPU
    strategy = tf.distribute.MirroredStrategy() if len(tf.config.list_physical_devices('GPU')) > 1 else tf.distribute.get_strategy()
    
    with strategy.scope():
        model = Sequential([
            # Input com shape dinâmico (batch_size livre) ao invés de batch_shape=(1,...)
            # que travava o batch_size em 1 e tornava o treinamento ~32× mais lento.
            tf.keras.layers.Input(shape=(FRAMES_PER_SEQUENCE, FEATURES_PER_FRAME)),
            
            # Bloco CNN-1D: captura padrões espaciais locais (posições dos dedos)
            # antes da LSTM processar a dinâmica temporal.
            Conv1D(64, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            Conv1D(64, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            SpatialDropout1D(0.2),
            
            # Bloco LSTM Bidirecional: aprende temporalidade em ambas as direções
            Bidirectional(LSTM(128, return_sequences=True)),
            Bidirectional(LSTM(64, return_sequences=False)),
            
            # Classificador
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
    model.summary()
        
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
        callbacks=callbacks,
        class_weight=class_weight_dict
    )
    
    print("Evaluating on test set...")
    if len(X_test) > 0:
        y_pred_probs = model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        unique_labels_test = np.unique(y_test)
        target_names = [reverse_class_map[i] for i in unique_labels_test]
        print(classification_report(y_test, y_pred, target_names=target_names, labels=unique_labels_test))
    
    # Convert to TFLite
    # Conversão explícita garantindo que as operações complexas como LSTMs
    # sejam mantidas como Float32 se necessário para evitar erros de compatibilidade no client.
    # ERRO COMUM: TFLite (TensorListReserve) requer batch_size estático para LSTMs.
    # Solução: Recriar o modelo com batch_shape fixo = 1 e carregar os pesos treinados.
    print("Rebuilding model with static batch size for TFLite conversion...")
    
    temp_weights_path = os.path.join(models_dir, 'temp_weights.weights.h5')
    model.save_weights(temp_weights_path)
    
    tflite_model_keras = Sequential([
        tf.keras.layers.Input(batch_shape=(1, FRAMES_PER_SEQUENCE, FEATURES_PER_FRAME)),
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        SpatialDropout1D(0.2),
        Bidirectional(LSTM(128, return_sequences=True)),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    tflite_model_keras.load_weights(temp_weights_path)
    
    print("Converting model to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(tflite_model_keras)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float32]
    tflite_model = converter.convert()
    
    tflite_path = os.path.join(models_dir, 'libras_model.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
        
    # Clean up temporary weights
    if os.path.exists(temp_weights_path):
        os.remove(temp_weights_path)
        
    print(f"TFLite model saved to {tflite_path}!")

if __name__ == "__main__":
    main()
