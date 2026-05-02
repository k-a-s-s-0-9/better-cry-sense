import os
import re
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
# This defines where the script looks for your processed .npy files
ROOT_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = ROOT_DIR / "data" / "processed" 
CATEGORIES = {"belly_pain": 0, "burping": 1, "discomfort": 2, "hungry": 3, "tired": 4}

def extract_uuid(filename):
    """Extracts the 36-character UUID from the filename."""
    match = re.search(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}', filename)
    return match.group(0) if match else "UNKNOWN"

def _flatten_uuids(uuids, uuid_to_files):
    """Helper to convert a list of UUIDs back into flat file and label lists."""
    files, labels = [], []
    for uid in uuids:
        for filepath, label in uuid_to_files[uid]:
            files.append(filepath)
            labels.append(label)
    return files, labels

def build_dataset_manifest():
    """Crawls directories and performs stratified split by UUID."""
    uuid_to_files = {}  
    uuid_to_label = {}  # Tracks the class of the baby for stratification
    
    print(f"DEBUG: Scanning directories in {BASE_DIR}...")

    # 1. CRAWL AND GROUP
    for category, label_int in CATEGORIES.items():
        mel_dir = BASE_DIR / category / "mel"
        if not mel_dir.exists():
            print(f"WARNING: Directory not found: {mel_dir}")
            continue
            
        for filepath in mel_dir.glob("*.npy"):
            uuid = extract_uuid(filepath.name)
            if uuid not in uuid_to_files:
                uuid_to_files[uuid] = []
                uuid_to_label[uuid] = label_int
            uuid_to_files[uuid].append((str(filepath), label_int))

    # 2. STRATIFIED SPLIT BY UUID
    all_uuids = list(uuid_to_files.keys())
    labels_per_uuid = [uuid_to_label[uid] for uid in all_uuids]
    
    # Split 20% for Test (Stratified)
    temp_uuids, test_uuids, temp_labels, _ = train_test_split(
        all_uuids, labels_per_uuid, test_size=0.2, stratify=labels_per_uuid, random_state=42
    )
    
    # Split remaining into Train (80%) and Val (20%) (Stratified)
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
    """Custom Keras Generator with Min-Max Scaling."""
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
        """Updates indexes after each epoch and OVER-SAMPLES minority classes."""
        # 1. Start with the base indexes
        base_indexes = np.arange(len(self.filepaths))
        
        # 2. Identify where the minority classes are
        # CATEGORIES: belly_pain=0, burping=1, discomfort=2, hungry=3, tired=4
        minority_indices = []
        for i, label in enumerate(self.labels):
            if label in [0, 1, 2, 4]:  # Everything EXCEPT Hungry
                minority_indices.append(i)
                
        # 3. Duplicate the minority indices to "balance" the epoch
        # This forces the model to see minority classes 3x more often
        oversampled_indices = minority_indices * 2 
        
        # 4. Combine and Shuffle
        self.indexes = np.concatenate([base_indexes, oversampled_indices])
        
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
    def __data_generation(self, batch_filepaths, batch_labels):
        X = []
        y = tf.keras.utils.to_categorical(batch_labels, num_classes=self.num_classes)

        for path in batch_filepaths:
            mel_spec = np.load(path)
            
            # SCALING: Normalize to 0.0 - 1.0 range
            m_min, m_max = mel_spec.min(), mel_spec.max()
            if m_max - m_min > 1e-6:
                mel_spec = (mel_spec - m_min) / (m_max - m_min)
            
            X.append(np.expand_dims(mel_spec, axis=-1))

        return np.array(X), y
