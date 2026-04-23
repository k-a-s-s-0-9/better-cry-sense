import tensorflow as tf
from keras import layers, Model

def build_crnn_model(input_shape=(128, 87, 1), num_classes=5):
    # 1. THE INPUT LAYER
    inputs = layers.Input(shape=input_shape, name="mel_input")

    # 2. THE SPATIAL BRANCH (CNN)
    # Block 1
    x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Block 2
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 3
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    # Pool heavily on frequency to leave a clean time-sequence
    x = layers.MaxPooling2D(pool_size=(4, 2))(x) 

    # 3. THE BRIDGE (Reshape to Sequence)
    # After pooling, your shape is (8, 10, 64). We flatten height (8) into filters (64).
    shape = x.shape
    time_steps = shape[2] 
    features = shape[1] * shape[3] 
    x = layers.Reshape((time_steps, features))(x) # Shape: (10, 512)

    # 4. THE TEMPORAL BRANCH (LSTM)
    # return_sequences=False is MANDATORY here to fix the Rank Mismatch error.
    # We use 64 units—enough to learn patterns, but small enough to avoid total memorization.
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False), name="bilstm_final")(x)
    x = layers.Dropout(0.3)(x)

    # 5. CLASSIFIER HEAD
    x = layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="classifier_output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="CRNN_Cry_Sense")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), # Stable 0.0001
        loss='categorical_crossentropy',
        metrics=[
            'accuracy', 
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.F1Score(average='macro', name='f1')
        ]
    )
    return model
