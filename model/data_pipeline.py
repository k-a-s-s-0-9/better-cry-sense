import os
import re
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split

# --- CONSTANTS ---
CATEGORIES = {"belly_pain": 0, "burping": 1, "discomfort": 2, "hungry": 3, "tired": 4}

def extract_uuid(filename):
    match = re.search(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}', filename)
    return match.group(0) if match else "UNKNOWN"

def _flatten_uuids(uuids, uuid_to_files):
    files, labels = [], []
    for uid in uuids:
        for filepath, label in uuid_to_files[uid]:
            files.append(filepath)
            labels.append(label)
    return files, labels

def build_dataset_manifest():
    uuid_to_files = {}  # { "UUID": [ (filepath, label_int), ... ] }
    uuid_to_label = {}  # { "UUID": label_int } - Needed for stratification
    
    # 1. CRAWL AND GROUP
    for category, label_int in CATEGORIES.items():
        mel_dir = BASE_DIR / category / "mel"
        if not mel_dir.exists(): continue
            
        for filepath in mel_dir.glob("*.npy"):
            uuid = extract_uuid(filepath.name)
            if uuid not in uuid_to_files:
                uuid_to_files[uuid] = []
                uuid_to_label[uuid] = label_int # Associate the baby with their class
            uuid_to_files[uuid].append((str(filepath), label_int))

    # 2. STRATIFIED SPLIT
    all_uuids = list(uuid_to_files.keys())
    labels_per_uuid = [uuid_to_label[uid] for uid in all_uuids]
    
    # Stratified split ensures 'belly_pain' babies are represented in all sets
    temp_uuids, test_uuids, temp_labels, test_labels_uuid = train_test_split(
        all_uuids, labels_per_uuid, test_size=0.2, stratify=labels_per_uuid, random_state=42
    )
    
    # Split remaining 80% into Train and Val
    train_uuids, val_uuids, _, _ = train_test_split(
        temp_uuids, temp_labels, test_size=0.2, stratify=temp_labels, random_state=42
    )
    
    # 3. FLATTEN
    train_files, train_labels = _flatten_uuids(train_uuids, uuid_to_files)
    val_files, val_labels = _flatten_uuids(val_uuids, uuid_to_files)
    test_files, test_labels = _flatten_uuids(test_uuids, uuid_to_files)

    print(f"📊 Pipeline Built: {len(train_files)} Train | {len(val_files)} Val | {len(test_files)} Test")
    return train_files, train_labels, val_files, val_labels, test_files, test_labels

class MelSpecGenerator(tf.keras.utils.Sequence):
    def __init__(self, filepaths, labels, batch_size=32, shuffle=True):
        self.filepaths = np.array(filepaths)
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_classes = len(CATEGORIES)
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.filepaths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        return self.__data_generation(self.filepaths[indexes], self.labels[indexes])

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.filepaths))
        if self.shuffle: np.random.shuffle(self.indexes)

    def __data_generation(self, batch_filepaths, batch_labels):
        X = []
        y = tf.keras.utils.to_categorical(batch_labels, num_classes=self.num_classes)

        for path in batch_filepaths:
            mel_spec = np.load(path)
            
            # --- CRITICAL CHANGE: MIN-MAX SCALING (0 to 1) ---
            # This prevents raw decibel values from exploding the gradients
            m_min, m_max = mel_spec.min(), mel_spec.max()
            if m_max - m_min > 1e-6: # Prevent division by zero
                mel_spec = (mel_spec - m_min) / (m_max - m_min)
            
            X.append(np.expand_dims(mel_spec, axis=-1))

        return np.array(X), y
