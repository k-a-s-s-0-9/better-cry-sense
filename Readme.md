Project Overview

This project is an advanced remake of the original CrySense repository. While the baseline used static Image Classification (CNNs) on spectrograms, our version treats baby cries as time-series audio signals. By combining Convolutional Neural Networks (CNN) for frequency feature extraction with Long Short-Term Memory (LSTM) networks for temporal rhythm analysis, we aim to surpass the 85.7% accuracy barrier.

Tech Stack
* Deep Learning: PyTorch / Keras (TensorFlow)
* Audio Processing: Librosa, Torchaudio, Audiomentations
* Data Science: NumPy, Pandas, Matplotlib, Seaborn
* Environment: Python 3.10+, Jupyter Notebooks / Python Scripts
* Deployment (Planned): Streamlit / Gradio (Web Interface)

Project File Structure
Plaintext

CrySense-Remake/
├── data/
│   ├── raw/              # Original datasets (Kaggle, Mendeley, etc.)
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

Production Cycle (Phase-wise Split)

Phase 1: Data Acquisition & Fusion (Current Phase)

Goal: Create a robust, balanced "Golden Dataset."

    Task 1.1: Retrieve and download datasets (Kaggle Donate-a-Cry, Mendeley, etc.).

    Task 1.2: Data Cleaning: Remove silence, normalize volume, and standardize sample rates (e.g., 22050Hz).

    Task 1.3: Address Imbalance: Implement Oversampling (SMOTE) or data augmentation (Pitch Shifting, Time Stretching) for minority classes (e.g., "Burping").

    Task 1.4: Generate a unified master_metadata.csv mapping all files to labels.

Phase 2: Feature Engineering & Pre-processing

Goal: Transform raw audio into high-dimensional input for the CNN.

    Task 2.1: Apply Pre-emphasis filtering to boost high-frequency cry signals.

    Task 2.2: Extract Log-Mel Spectrograms and MFCCs.

    Task 2.3: Segment long audio clips into fixed-length windows (e.g., 2-second chunks) to provide sequences for the LSTM.

Phase 3: Model Architecture & Training

Goal: Build and optimize the CNN + LSTM hybrid.

    Task 3.1: Design the CNN Backbone (2D Conv layers) to act as a feature extractor.

    Task 3.2: Bridge CNN output to LSTM Layers to capture the "rhythm" of the cry.

    Task 3.3: Implement training callbacks (Early Stopping, Learning Rate Scheduler).

    Task 3.4: Evaluate using Confusion Matrices and F1-Scores (vital for imbalanced data).

Phase 4: Optimization & Deployment

Goal: Finalize the "Applied" aspect of the project.

    Task 4.1: Hyperparameter Tuning: Optimize dropout rates and hidden unit sizes.

    Task 4.2: Develop a Streamlit/Gradio App for real-time file uploads and predictions.

    Task 4.3: Document results and prepare the final presentation/report.

Contribution Guidelines

    Branching: Create a new branch for each phase (e.g., feature/phase1-data-fusion).

    Commits: Use descriptive commit messages (e.g., feat: added pitch-shifting augmentation script).

    Validation: Ensure all new preprocessing scripts are tested against the tests/ folder.