import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization, GlobalAveragePooling2D, SeparableConv2D
import hyperparameters as hp

class SceneClassificationModel(tf.keras.Model):
    """Custom CNN for 15-scene classification."""

    def __init__(self):
        super().__init__()
        self.optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=hp.learning_rate, momentum=hp.momentum
        )
        
        self.architecture = [
            Conv2D(64, 5, activation="relu", padding="same"),
            Conv2D(64, 5, activation="relu", padding="same"),
            BatchNormalization(),
            MaxPool2D(pool_size=3, strides=2),
            Dropout(0.2),

            SeparableConv2D(128, 5, activation="relu", padding="same"),
            SeparableConv2D(128, 5, activation="relu", padding="same"),
            BatchNormalization(),
            MaxPool2D(pool_size=3, strides=2),
            Dropout(0.2),

            SeparableConv2D(256, 5, activation="relu", padding="same"),
            SeparableConv2D(256, 5, activation="relu", padding="same"),
            BatchNormalization(),
            MaxPool2D(pool_size=3, strides=2),
            Dropout(0.2),

            SeparableConv2D(512, 5, activation="relu", padding="same"),
            SeparableConv2D(512, 5, activation="relu", padding="same"),
            BatchNormalization(),
            MaxPool2D(pool_size=3, strides=2),
            Dropout(0.2),

            SeparableConv2D(512, 7, activation="relu", padding="same"), 
            BatchNormalization(),
            MaxPool2D(pool_size=3, strides=2),
            Dropout(0.2),

            GlobalAveragePooling2D(),
            Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),  
            Dropout(0.2),
            Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            Dropout(0.2),
            Dense(15, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(0.01))
        ]

    def call(self, img):
        """Pass input image through the network."""
        for layer in self.architecture:
            img = layer(img)
        return img

    @staticmethod
    def loss_fn(labels, predictions):
        """Compute sparse categorical cross-entropy loss."""
        return tf.keras.losses.sparse_categorical_crossentropy(labels, predictions, from_logits=False)
