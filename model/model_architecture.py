import tensorflow as tf
from keras import layers, Model

def build_crnn_model(input_shape=(128, 87, 1), num_classes=5):
    inputs = layers.Input(shape=input_shape, name="mel_input")

    # 1. SPATIAL BRANCH (CNN) - Keeps it simple
    x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(4, 2))(x) 

    # 2. THE BRIDGE (Reshape)
    shape = x.shape
    time_steps = shape[2] 
    features = shape[1] * shape[3] 
    x = layers.Reshape((time_steps, features))(x) 

    # 3. THE "GUTTED" LSTM
    # - Slashing units to 16: It can only "remember" a tiny summary.
    # - Single Direction: Less parameters than Bidirectional.
    # - L2 Regularization: Punishes complex internal weights.
    x = layers.LSTM(16, 
                    return_sequences=False, 
                    kernel_regularizer=tf.keras.regularizers.l2(0.01),
                    name="lstm_bottleneck")(x)
    x = layers.Dropout(0.4)(x)

    # 4. CLASSIFIER HEAD
    x = layers.Dense(16, activation='relu')(x) # Shrinking this too
    outputs = layers.Dense(num_classes, activation='softmax', name="classifier_output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="Lean_CRNN_Cry_Sense")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy', 
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.F1Score(average='macro', name='f1')
        ]
    )
    return model
