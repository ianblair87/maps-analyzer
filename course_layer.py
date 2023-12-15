import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.nn as nn
from unet import get_model
import skimage

def course_color_mask(img):
    course = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
#     course = cv2.blur(course, (3, 3))
    lower_pink = np.array([120, 50, 20])
    upper_pink = np.array([180, 256, 256])
    lower_red = np.array([0, 150, 0])
    upper_red = np.array([10, 256, 256])
    mask_pink = cv2.inRange(course, lower_pink, upper_pink)
    mask_red = cv2.inRange(course, lower_red, upper_red)
    mask_course = cv2.bitwise_or(mask_pink, mask_red)
    course = cv2.bitwise_and(course,course, mask=mask_course)
    course = cv2.cvtColor(course, cv2.COLOR_HSV2RGB)
    course = cv2.cvtColor(course, cv2.COLOR_RGB2GRAY)
    _, course = cv2.threshold(course, 50, 255, cv2.THRESH_BINARY)
    return course
    

def detect_course_impl(img):
    course = course_color_mask(img)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    # course = cv2.erode(course, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    course = cv2.dilate(course, kernel)
    course = cv2.medianBlur(course, 5)
    
    return course

def detect_course_model(img):
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()

            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=5, padding=2),
                nn.Sigmoid()
            )

            def weights_init(m):
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.xavier_uniform_(m.weight)
                    torch.nn.init.zeros_(m.bias)

            self.features.apply(weights_init)
            torch.nn.utils.clip_grad_norm_(self.features.parameters(), max_norm=1)

        def forward(self, x):
            return self.features(x)


    # Evaluate the model on test data
    model = SimpleCNN()
    model.load_state_dict(torch.load('573-orienteering/layer_separation_model.pth'))
    model.eval()
    with torch.no_grad():
        testing_tensor = torch.tensor(img).permute(2, 0, 1).to(torch.float32)
        testing_tensor = testing_tensor.unsqueeze(0)
        outputs = model(testing_tensor)
        predictions = (outputs > 0.5).float()

        prediction = np.array(predictions[0].squeeze())
        return prediction


def detect_course_unet(img):
    img = course_color_mask(img)
    while img.shape[0] > 1500 or img.shape[1] > 1500:
        img = skimage.measure.block_reduce(img, (2,2), np.max)
    
    model = get_model((128,128,1))
    model.load_weights('checkpoints/unet_course_4')
    pred = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float64)

    for i in range(0, img.shape[0], 128):
        for j in range(0, img.shape[1], 128):
            if i + 128 > img.shape[0]:
                i = img.shape[0] - 128
            if j + 128 > img.shape[1]:
                j = img.shape[1] - 128
            pred[i:i+128,j:j+128] = model.predict(img[i:i+128,j:j+128].reshape(-1, 128, 128, 1) / 255)[0]
    _, res = cv2.threshold(pred, 0.1, 1.0, cv2.THRESH_BINARY)
    res = np.uint8(res * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    res = cv2.dilate(res, kernel)
    res = cv2.medianBlur(res, 5)
    
    return res

def detect_course(session_id):
    img = cv2.imread(f'sessions/{session_id}/image_orig.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return detect_course_unet(img)

