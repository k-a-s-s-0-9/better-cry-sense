# Project Overview

This project is an advanced remake of the original CrySense repository. While the baseline used static Image Classification (CNNs) on spectrograms, our version treats baby cries as time-series audio signals. By combining Convolutional Neural Networks (CNN) for frequency feature extraction with Long Short-Term Memory (LSTM) networks for temporal rhythm analysis, we aim to surpass the 85.7% accuracy barrier.

## Datasets
* https://www.kaggle.com/datasets/aniruth100/baby-cry-detection?select=donateacry_corpus_cleaned_and_updated_data
* https://data.mendeley.com/datasets/hbppd883sd/1

## File naming convention
The audio files should contain baby cry samples, with the corresponding tagging information encoded in the filenames. The samples were tagged by the contributors themselves. So here's how to parse the filenames.

```
iOS:
0D1AD73E-4C5E-45F3-85C4-9A3CB71E8856-1430742197-1.0-m-04-hu.caf
app instance uuid (36 chars)-unix epoch timestamp-app version-gender-age-reason
```
So, the above translates to:
- the sample was recorded with the app instance having the unique id 0D1AD73E-4C5E-45F3-85C4-9A3CB71E8856. These ids are generated upon installation, so they identify an installed instance, not a device or a user
- the recording was made at 1430742197 (unix time epoch) , which translates to Mon, 04 May 2015 12:23:17 GMT
- version 1.0 of the mobile app was used
- the user tagged the recording to be of a boy
- the baby is 0-4 weeks old according to the user
- the suspected reason of the cry is hunger

```
Android:
0c8f14a9-6999-485b-97a2-913c1cbf099c-1431028888092-1.7-m-26-sc.3gp
The structure is the same with the exception that the unix epoch timestamp is in milliseconds
```

## Tags
### Gender
- *m* - male
- *f* - female

### Age
- *04* - 0 to 4 weeks old
- *48* - 4 to 8 weeks old
- *26* - 2 to 6 months old
- *72* - 7 month to 2 years old
- *22* - more than 2 years old

### Reason
- *hu* - hungry
- *bu* - needs burping
- *bp* - belly pain
- *dc* - discomfort
- *ti* - tired
- *lo* - lonely
- *ch* - cold/hot
- *sc* - scared
- *dk* - don't know


## Original repo
* https://github.com/SrijanShovit/CrySense

# Tech Stack
* Deep Learning: PyTorch / Keras (TensorFlow)
* Audio Processing: Librosa, Torchaudio, Audiomentations
* Data Science: NumPy, Pandas, Matplotlib, Seaborn
* Environment: Python 3.10+, Jupyter Notebooks / Python Scripts
* Deployment (Planned): Streamlit / Gradio (Web Interface)

# Project File Structure

```
    CrySense-Remake/
    ├── data/
    │   ├── raw/              # Original datasets   (Kaggle, Mendeley, etc.)
    │   ├── processed/        # Fused, cleaned, and augmented .npy or .wav files
    │   └── metadata/         # Combined .csv with labels and file paths
    ├── models/
    │   ├── architecture/     # Model definition scripts (CNN_LSTM_Model.py)
    │   └── saved_weights/    # Trained .h5 or .pth files
    ├── notebooks/            # Research, EDA, and Phase-wise experiments
    ├── src/
    │   ├── preprocessing/    # Feature extraction (MFCC, Mel-Spectrogram)
    │   ├── training/         # Training loops and validation scripts
    │   └── utils/            # Helper functions (plotting, metrics)
    ├── tests/                # Unit tests for audio loading and model shapes
    ├── requirements.txt      # Project dependencies
    └── README.md             # You are here
```

# Production Cycle (Phase-wise Split)

## Phase 1: Data Acquisition & Fusion (Done)

Goal: Create a robust, balanced "Golden Dataset."

    Task 1.1: Retrieve and download datasets (Kaggle Donate-a-Cry, Mendeley, etc.).

    Task 1.2: Data Cleaning: Remove silence, normalize volume, and standardize sample rates (e.g., 22050Hz).

    Task 1.3: Address Imbalance: Implement Oversampling (SMOTE) or data augmentation (Pitch Shifting, Time Stretching) for minority classes (e.g., "Burping").

    Task 1.4: Generate a unified master_metadata.csv mapping all files to labels.

## Phase 2: Feature Engineering & Pre-processing (Done)

Goal: Transform raw audio into high-dimensional input for the CNN.

    Task 2.1: Apply Pre-emphasis filtering to boost high-frequency cry signals.

    Task 2.2: Extract Log-Mel Spectrograms and MFCCs.

    Task 2.3: Segment long audio clips into fixed-length windows (e.g., 2-second chunks) to provide sequences for the LSTM.

## Phase 3: Model Architecture & Training (Done)

Goal: Build and optimize the CNN + LSTM hybrid.

    Task 3.1: Design the CNN Backbone (2D Conv layers) to act as a feature extractor.

    Task 3.2: Bridge CNN output to LSTM Layers to capture the "rhythm" of the cry.

    Task 3.3: Implement training callbacks (Early Stopping, Learning Rate Scheduler).

    Task 3.4: Evaluate using Confusion Matrices and F1-Scores (vital for imbalanced data).

## Phase 4: Optimization & Deployment

Goal: Finalize the "Applied" aspect of the project.

    Task 4.1: Hyperparameter Tuning: Optimize dropout rates and hidden unit sizes.

    Task 4.2: Develop a Streamlit/Gradio App for real-time file uploads and predictions.

    Task 4.3: Document results and prepare the final presentation/report.
