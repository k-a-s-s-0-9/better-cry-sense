import os
import numpy as np
import librosa

# -------- Task 2.1: Pre-emphasis --------
def pre_emphasis(signal, alpha=0.97):
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])


# -------- Task 2.3: Segmentation --------
def segment_audio(signal, sr, duration=2):
    segment_length = int(duration * sr)

    for start in range(0, len(signal), segment_length):
        segment = signal[start:start + segment_length]

        if len(segment) == segment_length:
            yield segment   # memory efficient


# -------- Task 2.2: Feature Extraction --------
def extract_features(signal, sr):
    mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
    log_mel = librosa.power_to_db(mel)

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)

    # ✅ SAFE NORMALIZATION (avoid divide by zero)
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)
    mfcc = (mfcc - mfcc.min()) / (mfcc.max() - mfcc.min() + 1e-8)

    return log_mel, mfcc


# -------- Process + Save --------
def process_and_save(file_path, category):
    base_output = "data/processed"   # clean output folder

    file_name = os.path.splitext(os.path.basename(file_path))[0]

    mel_dir = os.path.join(base_output, category, "mel")
    mfcc_dir = os.path.join(base_output, category, "mfcc")

    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(mfcc_dir, exist_ok=True)

    # ✅ Fixed sample rate
    signal, sr = librosa.load(file_path, sr=22050)

    # Step 1: Pre-emphasis
    signal = pre_emphasis(signal)

    # Step 2 + 3: Segment + Extract + Save
    for i, segment in enumerate(segment_audio(signal, sr)):
        mel, mfcc = extract_features(segment, sr)

        np.save(os.path.join(mel_dir, f"{file_name}_mel_{i}.npy"), mel)
        np.save(os.path.join(mfcc_dir, f"{file_name}_mfcc_{i}.npy"), mfcc)


# -------- Run for Entire Dataset --------
if __name__ == "__main__":

    # ✅ FIXED PATH (IMPORTANT: raw string + correct folder)
    base_folder = r"C:\Users\91700\OneDrive\Desktop\better-cry-sense\data\processed\augmented"

    for category in os.listdir(base_folder):
        category_path = os.path.join(base_folder, category)

        if os.path.isdir(category_path):
            print(f"\nProcessing category: {category}")

            for file in os.listdir(category_path):
                if file.endswith(".wav"):
                    file_path = os.path.join(category_path, file)

                    print("Processing:", file_path)
                    process_and_save(file_path, category)

    print("\n✅ Phase 2 completed successfully (no memory issues)!")