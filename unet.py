import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def get_model(input_shape=(128,128,3)):
    def downsample_block(x, n_filters):
        x = layers.Conv2D(n_filters, 3, padding = "same", activation = "elu", kernel_initializer = "he_normal")(x)
        f = layers.Conv2D(n_filters, 3, padding = "same", activation = "elu", kernel_initializer = "he_normal")(x)
        p = layers.MaxPool2D(2)(f)
        p = layers.Dropout(0.3)(p)
        return f, p
    def upsample_block(x, conv_features, n_filters):
        x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.concatenate([x, conv_features])
        x = layers.Conv2D(n_filters, 3, padding = "same", activation = "elu", kernel_initializer = "he_normal")(x)
        x = layers.Conv2D(n_filters, 3, padding = "same", activation = "elu", kernel_initializer = "he_normal")(x)
        return x
    inputs = layers.Input(shape=input_shape)
    f1, p1 = downsample_block(inputs, 16)
    f2, p2 = downsample_block(p1, 32)
    x = layers.Conv2D(64, 3, padding = "same", activation = "elu", kernel_initializer = "he_normal")(p2)
    x = layers.Dropout(0.3)(x)
    u2 = upsample_block(x, f2, 32)
    u1 = upsample_block(u2, f1, 16)
    outputs = layers.Conv2D(1, 3, padding = "same", activation = "sigmoid", kernel_initializer = "he_normal")(u1)
    return tf.keras.Model(inputs, outputs, name="U-Net")
