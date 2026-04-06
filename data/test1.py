# checking if the augmented audio snippets are distinct enough from the original audio snippets
import os
from pathlib import Path
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# --- CONFIGURATION ---
CATEGORY = "belly_pain"  # Options: belly_pain, burping, discomfort, hungry, tired

# This finds the root directory (better-cry-sense) regardless of where you run it
BASE_DIR = Path(__file__).resolve().parent.parent

# These point exactly to the folders in your screenshot
RAW_DIR = BASE_DIR / "data" / "raw" / "donateacry_corpus_cleaned_and_updated_data" / CATEGORY
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "augmented" / CATEGORY

print(f"📂 Looking for Raw in: {RAW_DIR}")
print(f"📂 Looking for Processed in: {PROCESSED_DIR}")

THRESHOLD = 0.98  # Anything above this is "Too Similar"

# New naming convention for the output file
OUTPUT_FILE = BASE_DIR / f"results-{CATEGORY}.txt"

# ---------------------

def get_spectrogram(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        y = np.append(y[0], y[1:] - 0.97 * y[:-1]) 
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        return librosa.power_to_db(S, ref=np.max)
    except Exception as e:
        return None

def audit_category():
    if not RAW_DIR.exists() or not PROCESSED_DIR.exists():
        print(f"❌ Directory error. Check: {RAW_DIR}")
        return

    raw_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.wav')]
    processed_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith('.wav')]

    print(f"🕵️  Starting audit for {CATEGORY}...")
    print(f"📝 Results will be saved to: {OUTPUT_FILE.name}")

    # Use 'w' mode to overwrite the file every time you run it
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"Auditing Category: {CATEGORY}\n")
        f.write(f"Found {len(raw_files)} raw files and {len(processed_files)} processed files.\n")
        f.write("="*50 + "\n\n")

        for raw_name in raw_files:
            base_name = os.path.splitext(raw_name)[0]
            siblings = [p for p in processed_files if base_name in p]
            
            if not siblings:
                continue

            f.write(f"Raw File: {raw_name}\n")
            raw_spec = get_spectrogram(os.path.join(RAW_DIR, raw_name))
            
            for sib_name in siblings:
                sib_spec = get_spectrogram(os.path.join(PROCESSED_DIR, sib_name))
                
                if raw_spec is not None and sib_spec is not None:
                    corr, _ = pearsonr(raw_spec.flatten(), sib_spec.flatten())
                    status = "--FAIL--" if corr > THRESHOLD else "--PASS--"
                    f.write(f"   -> {status} | Corr: {corr:.4f} | {sib_name}\n")
            f.write("-" * 30 + "\n")

    print(f"✅ Audit Complete! Open {OUTPUT_FILE.name} to see the results.")

if __name__ == "__main__":
    audit_category()