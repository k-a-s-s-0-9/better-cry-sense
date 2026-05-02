import tensorflow as tf
from keras import layers, Model

def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        # Prevent log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Calculate cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Calculate focal loss modulating factor
        loss = alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy
        return tf.math.reduce_sum(loss, axis=-1)
    return focal_loss_fixed

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

    # 3. THE Bidirectional LSTM
    # 1. Spatial Dropout: Drops entire feature maps, great for audio/Conv outputs
    x = layers.SpatialDropout2D(0.3)(x) 
    x = layers.Reshape((-1, x.shape[-1] * x.shape[-2]))(x)

    # 2. The Bi-LSTM: 32 units (doubled to 64 internally)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=False, name="bi_lstm_core"))(x)

    # 3. Heavy Dropout: The "handbrake" on overfitting
    x = layers.Dropout(0.5)(x)

    # 4. CLASSIFIER HEAD
    x = layers.Dense(16, activation='relu')(x) # Shrinking this too
    outputs = layers.Dense(num_classes, activation='softmax', name="classifier_output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="Lean_CRNN_Cry_Sense")

    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=focal_loss(gamma=2.0, alpha=0.25),
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.F1Score(average='macro', name='f1')]
    )
    return model
