import tensorflow as tf
from keras import layers, Model

def build_crnn_model(input_shape=(128, 87, 1), num_classes=5):
    # 1. THE INPUT LAYER
    inputs = layers.Input(shape=input_shape, name="mel_input")

    # 2. THE SPATIAL BRANCH (CNN)
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
    x = layers.BatchNormalization()(x)
    
    # 3. THE SQUEEZE (Replaces Bridge & LSTM)
    # This collapses all spatial/temporal features into one 'vibe' vector per class.
    x = layers.GlobalAveragePooling2D()(x) 
    
    # 4. CLASSIFIER HEAD
    x = layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="classifier_output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="Squeezed_Cry_Sense")

    # Compile with safer metric names
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001(1e-4)),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy', 
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.F1Score(average='macro', name='f1')
        ]
    )
    return model

if __name__ == "__main__":
    model = build_crnn_model()
    model.summary()
