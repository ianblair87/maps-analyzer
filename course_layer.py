import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.nn as nn

def detect_course_impl(img):
    course = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
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


def detect_course(session_id):
    img = cv2.imread(f'sessions/{session_id}/image.jpg')
    return detect_course_impl(img)

