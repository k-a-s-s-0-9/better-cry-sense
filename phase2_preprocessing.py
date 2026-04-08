import os
import numpy as np
import librosa

# -------- Task 2.1: Pre-emphasis --------
def pre_emphasis(signal, alpha=0.97):
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])


# -------- Task 2.3: Segmentation --------
def segment_audio(signal, sr, duration=2):
    segment_length = int(duration * sr)
    segments = []

    for start in range(0, len(signal), segment_length):
        segment = signal[start:start + segment_length]

        if len(segment) == segment_length:
            segments.append(segment)

    return segments


# -------- Task 2.2: Feature Extraction --------
def extract_features(signal, sr):
    mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
    log_mel = librosa.power_to_db(mel)

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)

    return log_mel, mfcc


# -------- Full Pipeline --------
def process_audio(file_path):
    signal, sr = librosa.load(file_path, sr=None)

    # Step 1: Pre-emphasis
    signal = pre_emphasis(signal)

    # Step 2: Segmentation
    segments = segment_audio(signal, sr)

    features = []

    for seg in segments:
        mel, mfcc = extract_features(seg, sr)
        features.append((mel, mfcc))

    return features


# -------- Save Features (Category-wise, No Overwrite) --------
def save_features(file_path, category):
    base_output = "processed"

    file_name = os.path.splitext(os.path.basename(file_path))[0]

    mel_dir = os.path.join(base_output, category, "mel")
    mfcc_dir = os.path.join(base_output, category, "mfcc")

    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(mfcc_dir, exist_ok=True)

    features = process_audio(file_path)

    for i, (mel, mfcc) in enumerate(features):
        np.save(os.path.join(mel_dir, f"{file_name}_mel_{i}.npy"), mel)
        np.save(os.path.join(mfcc_dir, f"{file_name}_mfcc_{i}.npy"), mfcc)


# -------- Run for Entire Dataset --------
if __name__ == "__main__":

    base_folder = "data/raw/donateacry_corpus_cleaned_and_updated_data"

    for category in os.listdir(base_folder):
        category_path = os.path.join(base_folder, category)

        if os.path.isdir(category_path):
            print(f"\nProcessing category: {category}")

            for file in os.listdir(category_path):
                if file.endswith(".wav"):
                    file_path = os.path.join(category_path, file)

                    print("Processing:", file_path)
                    save_features(file_path, category)

    print("\n✅ Phase 2 preprocessing completed successfully!")