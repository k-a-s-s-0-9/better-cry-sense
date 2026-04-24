import os
from xml.parsers.expat import model
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Import your custom modules
from data_pipeline import build_dataset_manifest, MelSpecGenerator
from model_architecture import build_crnn_model

def setup_gpu():
#   Prevents TensorFlow from hoarding all your GPU VRAM at once
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU Memory Growth Enabled: {len(gpus)} GPU(s) found.")
        except RuntimeError as e:
            print(f"⚠️ GPU Setup Error: {e}")
    else:
        print("⚠️ No GPU detected. Training will run on CPU (this will be slower).")

def run_detailed_diagnostics(model, test_generator):
    print("📊 Generating Confusion Matrix and Classification Report...")
    
    y_true = []
    y_pred = []
    
    # Iterate through the test generator
    for i in range(len(test_generator)):
        x, y = test_generator[i]
        preds = model.predict(x, verbose=0)
        
        # Convert one-hot to class indices
        y_true.extend(np.argmax(y, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
    
    # Define labels based on your dataset structure
    class_names = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
    
    # 1. Compute the Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # 2. Plotting
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted Label (Model Guessed)')
    plt.ylabel('True Label (Actual Reason)')
    plt.title('Better Cry Sense: Confusion Matrix')
    plt.show()

    # 3. Textual Report (Precision, Recall, F1 per class)
    print("\n📝 Per-Class Performance Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))


def main():
    print("🚀 Initiating Better Cry Sense Training Sequence...")
    setup_gpu()

    # 1. Fetch the Manifest (Safeguarded against Data Leakage)
    train_files, train_labels, val_files, val_labels, test_files, test_labels = build_dataset_manifest()
    if len(train_files) == 0:
        print("❌ CRITICAL ERROR: Pipeline returned empty datasets. Check your paths.")
        return

    # 2. Calculate Class Weights (Handling the 382 Hungry vs 50 Burping Imbalance)
    print("⚖️ Calculating Class Weights...")
    classes = np.unique(train_labels)

    weight_dict = {
        0: 1.5,  # belly_pain (96 samples)
        1: 3.9,  # burping (50 samples) - Capped at 3.5 to keep gradients stable
        2: 3.4,  # discomfort (54 samples)
        3: 0.6,  # hungry (382 samples) - Enough weight to remain a 'baseline'
        4: 1.5   # tired (96 samples)
    }
    
    for class_id, weight in weight_dict.items():
        print(f"   -> Class {class_id} Weight: {weight:.2f}")

    # 3. Initialize Generators
    BATCH_SIZE = 32
    print(f"📦 Spinning up Data Generators (Batch Size: {BATCH_SIZE})...")
    train_generator = MelSpecGenerator(train_files, train_labels, batch_size=BATCH_SIZE, shuffle=True)
    val_generator = MelSpecGenerator(val_files, val_labels, batch_size=BATCH_SIZE, shuffle=False)
    test_generator = MelSpecGenerator(test_files, test_labels, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Build the CRNN
    print("🏗️ Constructing CRNN Architecture...")
    # Assuming the shape you found earlier: (128, 87) + 1 for the channel
    model = build_crnn_model(input_shape=(128, 87, 1), num_classes=5)

    # 5. Define Callbacks (The Safety Nets)
    checkpoint_dir = Path("saved_models")
    checkpoint_dir.mkdir(exist_ok=True)
    
    callbacks = [
        # Stops training if Validation Loss doesn't improve for 8 epochs
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=15,
            restore_best_weights=True,
            verbose=1,
            start_from_epoch=5
        ),
        # Saves the absolute best version of the model to your hard drive
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / "best_crnn_model.keras"),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # Automatically lowers the learning rate if the model gets "stuck"
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_auc', 
            factor=0.5,
            patience=8,        
            min_lr=1e-5,      
            verbose=1
),
        tf.keras.callbacks.CSVLogger(
            filename='training_log.csv',
            separator=',',
            append=False # Set to True if you are resuming a stopped training run
        )
    ]

    # 6. Ignite the Training Loop
    EPOCHS = 50
    print(f"\n🔥 Starting Training for up to {EPOCHS} Epochs...")
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        class_weight=weight_dict,
        callbacks=callbacks
    )

    print("\n✅ Training Complete. The best model has been saved to the 'saved_models' folder.")

    print("\n🏆 Running Final Evaluation on Unseen Test Set...")

    test_results = model.evaluate(test_generator)
    print("\n--- TEST SET METRICS ---")
    print(f"Test Accuracy: {test_results[1]:.4f}")
    print(f"Test AUC:      {test_results[2]:.4f}")
    print(f"Test F1 Score: {test_results[3]:.4f}") 

if __name__ == "__main__":
    main()
