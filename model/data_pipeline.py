import os
import re
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent / "data" / "processed"
CATEGORIES = {"belly_pain": 0, "burping": 1, "discomfort": 2, "hungry": 3, "tired": 4}
BATCH_SIZE = 32

def extract_uuid(filename):
#   Extracts the 36-character UUID from the filename.
#   Matches standard UUID format: 8-4-4-4-12
#   Regex to find the UUID pattern anywhere in the string
    match = re.search(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}', filename)
    if match:
        return match.group(0)
    return "UNKNOWN"

def build_dataset_manifest():
#   Crawls the processed directories and groups files by UUID to prevent data leakage.
#   Returns: train_files, val_files, train_labels, val_labels

    uuid_to_files = {}  # { "UUID": [ (filepath, label_int), ... ] }
    
    for category, label_int in CATEGORIES.items():
        mel_dir = BASE_DIR / category / "mel"
        if not mel_dir.exists():
            continue
            
        for filepath in mel_dir.glob("*.npy"):
            uuid = extract_uuid(filepath.name)
            if uuid not in uuid_to_files:
                uuid_to_files[uuid] = []
            uuid_to_files[uuid].append((str(filepath), label_int))

    # 1. Split the UUIDs (80% Train, 20% Val)
    all_uuids = list(uuid_to_files.keys())
    train_uuids, val_uuids = train_test_split(all_uuids, test_size=0.2, random_state=42)
    
    # 2. Flatten the splits back into lists of files
    train_files, train_labels = [], []
    for uid in train_uuids:
        for filepath, label in uuid_to_files[uid]:
            train_files.append(filepath)
            train_labels.append(label)
            
    val_files, val_labels = [], []
    for uid in val_uuids:
        for filepath, label in uuid_to_files[uid]:
            val_files.append(filepath)
            val_labels.append(label)

    print(f"📊 Pipeline Built: {len(train_files)} Train segments | {len(val_files)} Val segments")
    return train_files, train_labels, val_files, val_labels

class MelSpecGenerator(tf.keras.utils.Sequence):
    """
    Custom Keras Generator to load .npy files lazily.
    """
    def __init__(self, filepaths, labels, batch_size=32, shuffle=True):
        self.filepaths = np.array(filepaths)
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_classes = len(CATEGORIES)
        self.on_epoch_end()

    def __len__(self):
        # Number of batches per epoch
        return int(np.floor(len(self.filepaths) / self.batch_size))

    def __getitem__(self, index):
        # Generate indices of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        batch_filepaths = self.filepaths[indexes]
        batch_labels = self.labels[indexes]

        # Generate data
        X, y = self.__data_generation(batch_filepaths, batch_labels)
        return X, y

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.filepaths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_filepaths, batch_labels):
        # Initialization
        X = []
        y = tf.keras.utils.to_categorical(batch_labels, num_classes=self.num_classes)

        # Generate data
        for path in batch_filepaths:
            mel_spec = np.load(path)
            
            # CNNs need a channel dimension! Shape goes from (128, T) -> (128, T, 1)
            mel_spec = np.expand_dims(mel_spec, axis=-1)
            X.append(mel_spec)

        return np.array(X), y