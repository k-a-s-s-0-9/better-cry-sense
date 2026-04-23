import tensorflow as tf
from keras import layers, Model

def build_crnn_model(input_shape=(128, 87, 1), num_classes=5):

#   Builds a CRNN (CNN + LSTM) for audio classification.
    # 1. THE INPUT LAYER
    inputs = layers.Input(shape=input_shape, name="mel_input")

    # 2. THE SPATIAL BRANCH (CNN)
    # Goal: Extract local frequency patterns (pitch slopes, bursts)
    
    # Block 1: Basic textures
    x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.SpatialDropout2D(0.1)(x)

    # Block 2: Patterns
    x = layers.Conv2D(24, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.SpatialDropout2D(0.1)(x)

    # Block 3: High-level features
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x) # Replace the Bridge and LSTM

    # 3. THE BRIDGE (Reshape)
    # Goal: Convert 3D CNN output into 2D Sequence for LSTM
    # We want to keep the 'Time' steps (10) and flatten the 'Frequency' (8) and 'Channels' (128)
    # 8 * 128 = 1024 features per time step.
    
    # We extract the dimensions dynamically to avoid hardcoding
    shape = x.shape
    time_steps = shape[2] 
    features = shape[1] * shape[3] 
    
    x = layers.Reshape((time_steps, features), name="reshape_to_sequence")(x)
    # Shape is now (10, 1024) - A sequence of 10 time frames, each with 1024 features.

    # 4. THE TEMPORAL BRANCH (LSTM)
    # Goal: Analyze the rhythm and duration of the cry
    
    # Bidirectional LSTM looks at the sequence forwards and backwards to understand context
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False), name="bilstm")(x)
    x = layers.Dropout(0.2, name="drop_lstm")(x)

    # Shape is now a flat vector (64)

    # 5. THE CLASSIFIER HEAD
    x = layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), name="dense_1")(x)
    x = layers.Dropout(0.5, name="drop_dense")(x)
    
    outputs = layers.Dense(num_classes, activation='softmax', name="classifier_output")(x)

    # Assemble Model
    model = Model(inputs=inputs, outputs=outputs, name="Better_Cry_Sense_CRNN")
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy', 
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='prc'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.F1Score(name='f1'),
        ]
    )
    
    return model

# Quick Test to verify the architecture mathematics
if __name__ == "__main__":
    model = build_crnn_model()
    model.summary()