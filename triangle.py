import matplotlib.pyplot as plt
import numpy as np
import cv2


def detect_start(course, erode, dilate, visualize=False):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate,dilate))
    course = cv2.dilate(course, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erode,erode))
    course = cv2.erode(course, kernel)
    contours,hierarchy = cv2.findContours(course, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    triangles = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.1*cv2.arcLength(cnt, True), True)
        if len(approx) == 3:
            l = np.linalg.norm(approx[0] - approx[1])
            all_sizes_equal = True
            for i in range(3):
                if abs(np.linalg.norm(approx[i] - approx[(i + 1) % 3]) - l) / l > 0.1:
                    all_sizes_equal = False
            if not all_sizes_equal:
                continue
            if visualize:
                res = np.zeros_like(course)
                res = cv2.drawContours(res, [cnt], -1, 255, 1)
                plt.imshow(res)
                plt.show()
            
            triangles.append({
                'a': approx[0].tolist(),
                'b': approx[1].tolist(),
                'c': approx[2].tolist()
            })
    return triangles

def get_triangle_impl(session_id, erode, dilate):
    
    # todo: replace with finding a mask with NN
    
    course = np.load(open(f'sessions/{session_id}/course_layer.npy', 'rb'))

    return detect_start(course, erode, dilate)
