import tensorflow as tf
from keras import layers, Model

def build_crnn_model(input_shape=(128, 87, 1), num_classes=5):

#   Builds a CRNN (CNN + LSTM) for audio classification.
    # 1. THE INPUT LAYER
    inputs = layers.Input(shape=input_shape, name="mel_input")

    # 2. THE SPATIAL BRANCH (CNN)
    # Goal: Extract local frequency patterns (pitch slopes, bursts)
    
    # Block 1
    x = layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', name="conv_1")(inputs)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool_1")(x) 
    # Shape is now roughly (64, 43, 32)

    # Block 2
    x = layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', name="conv_2")(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool_2")(x)
    # Shape is now roughly (32, 21, 64)

    # Block 3
    x = layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu', name="conv_3")(x)
    x = layers.BatchNormalization(name="bn_3")(x)
    # We pool heavily on the frequency axis (4) to squash it, but gently on time (2) to preserve sequence
    x = layers.MaxPooling2D(pool_size=(4, 2), name="pool_3")(x)
    # Shape is now roughly (8, 10, 128)

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
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True), name="bilstm_1")(x)
    x = layers.Dropout(0.5, name="drop_lstm_1")(x)
    
    # Second LSTM layer (return_sequences=False because we only need the final summarized context)
    x = layers.Bidirectional(layers.LSTM(32), name="bilstm_2")(x)
    x = layers.Dropout(0.5, name="drop_lstm_2")(x)
    
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
            tf.keras.metrics.MeanAbsoluteError(name='mae'),
            tf.keras.metrics.MeanSquaredError(name='mse')
        ]
    )
    
    return model

# Quick Test to verify the architecture mathematics
if __name__ == "__main__":
    model = build_crnn_model()
    model.summary()