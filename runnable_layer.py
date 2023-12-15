import cv2
import numpy as np
from unet import get_model

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
