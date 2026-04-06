# checking if the augmented audio snippets are distinct enough from the original audio snippets
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# --- CONFIGURATION ---
CATEGORY = "hungry"  # Options: belly_pain, burping, discomfort, hungry, tired
RAW_DIR = f"raw/donateacry_corpus_cleaned_and_updated_data/{CATEGORY}"
PROCESSED_DIR = f"processed/augmented/{CATEGORY}"
THRESHOLD = 0.98  # Anything above this is "Too Similar"
# ---------------------

def get_spectrogram(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        y = np.append(y[0], y[1:] - 0.97 * y[:-1]) # Pre-emphasis
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        return librosa.power_to_db(S, ref=np.max)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def audit_category():
    if not os.path.exists(RAW_DIR) or not os.path.exists(PROCESSED_DIR):
        print("Directory error. Check your paths.")
        return

    raw_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.wav')]
    processed_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith('.wav')]

    print(f"Auditing Category: {CATEGORY}")
    print(f"Found {len(raw_files)} raw files and {len(processed_files)} processed files.\n")

    for raw_name in raw_files:
        # Strip extension to get the UUID/base name
        base_name = os.path.splitext(raw_name)[0]
        
        # Find all processed files that contain this base name
        siblings = [p for p in processed_files if base_name in p]
        
        if not siblings:
            continue

        print(f"Raw File: {raw_name}")
        raw_spec = get_spectrogram(os.path.join(RAW_DIR, raw_name))
        
        for sib_name in siblings:
            sib_spec = get_spectrogram(os.path.join(PROCESSED_DIR, sib_name))
            
            if raw_spec is not None and sib_spec is not None:
                # Calculate correlation
                corr, _ = pearsonr(raw_spec.flatten(), sib_spec.flatten())
                
                status = "FAIL" if corr > THRESHOLD else "✅ PASS"
                print(f"   -> {status} | Corr: {corr:.4f} | {sib_name}")
        print("-" * 30)

if __name__ == "__main__":
    audit_category()