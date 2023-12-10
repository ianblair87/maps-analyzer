import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def get_model():
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
    inputs = layers.Input(shape=(128,128,3))
    f1, p1 = downsample_block(inputs, 16)
    f2, p2 = downsample_block(p1, 32)
    x = layers.Conv2D(64, 3, padding = "same", activation = "elu", kernel_initializer = "he_normal")(p2)
    x = layers.Dropout(0.3)(x)
    u2 = upsample_block(x, f2, 32)
    u1 = upsample_block(u2, f1, 16)
    outputs = layers.Conv2D(1, 3, padding = "same", activation = "sigmoid", kernel_initializer = "he_normal")(u1)
    return tf.keras.Model(inputs, outputs, name="U-Net")


def get_runnable_layer(session_id):
    img = cv2.imread(f'sessions/{session_id}/image.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    model = get_model()
    model.load_weights('checkpoints/my_checkpoint_2')
    res = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float64)
    for i in range(0, img.shape[0], 128):
        for j in range(0, img.shape[1], 128):
            if i + 128 > img.shape[0]:
                i = img.shape[0] - 128
            if j + 128 > img.shape[1]:
                j = img.shape[1] - 128
            res[i:i+128,j:j+128] = model.predict(np.float64(img[i:i+128,j:j+128]).reshape(-1, 128, 128, 3) / 255)[0]
    res = cv2.blur(res, (5,5))
    _, res = cv2.threshold(res, 0.4, 1.0, cv2.THRESH_BINARY)

    return res
