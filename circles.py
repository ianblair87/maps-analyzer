import matplotlib.pyplot as plt
import numpy as np
import cv2
import tqdm

def circle_mask(c, shape):
    mask = np.zeros((shape[0],shape[1],1), np.uint8)
    cv2.circle(mask,(c[1],c[0]),c[2],1,10)
    return mask

def check(c, img, visualize=False):
    mask = circle_mask(c, img.shape)    
    res = cv2.bitwise_xor(img, mask)
    a = int(np.sqrt(2)/2 * c[2]) - 1
    res_square = res[c[0]-a:c[0] + a, c[1] - a:c[1]+a]
    if visualize:
        plt.imshow(img[c[0] - c[2]: c[0] + c[2], c[1] - c[2]:c[1] + c[2]])
        plt.show()
    return res_square.sum() < 0.02 * 255 * (4 * a * a)

def check_circle_cv_kernel(c, img, visualize=False, print_score=False):
    kernel = np.float32(cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2 * c[2] + 1, 2 * c[2] + 1)))
    kernel[3:-3, 3:-3] -= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2 * c[2] - 5, 2 * c[2] - 5))
    kernel[6:-6, 6:-6] -= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2 * c[2] - 11, 2 * c[2] - 11))
    res_square = img[c[0]-c[2]:c[0] + c[2] + 1, c[1] - c[2]:c[1]+c[2] + 1] / 255
    if visualize:
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)
        ax1.imshow(res_square)
        ax2.imshow((kernel + 1) / 2)
        plt.show()

    score = np.multiply(kernel, res_square).sum()
    threshold = 0.3 * np.absolute(kernel).sum()
    if print_score:
        print(score, threshold)
    return score > threshold

def hough_circle_fixed_radius(img, r, visualize=False):
    img8 = np.uint8(img * 255)
    res = np.zeros_like(img)
    n, m = img.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
    course = cv2.dilate(img8, kernel)
    plt.imshow(course)
    plt.show()
    
    for i in tqdm.tqdm(range(0, n)):
        for j in range(0, m):
            if course[i,j] == 255:
                res += circle_mask((i,j,r), course.shape).reshape((n, m))
    threshold = 0.8 * res.max()
    res[res < threshold] = 0
#     print(res.max())
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,8))
    course = cv2.erode(course, kernel)
    circles = []
    for i in tqdm.tqdm(range(0, n)):
        for j in range(0, m):
            if res[i,j] >= threshold and check_circle_cv_kernel((i, j, r+3), course,visualize=False, print_score=False):
                circles.append(np.array([i, j]))
    circles_filtered = []
    for circle in circles:
        duplicate = False
        for filtered_circle in circles_filtered:
            if np.linalg.norm(circle - filtered_circle) < r * 2:
                duplicate = True
                break
        if not duplicate:
            circles_filtered.append(circle)
    if visualize:
        n_res = np.zeros(res.shape)
        for circle in circles_filtered:
            n_res += circle_mask((circle[0],circle[1],r), course.shape).reshape((n, m))
        plt.imshow(n_res, cmap='gray')
        plt.show()
    return circles_filtered


def get_circles_impl(session_id):
    r = 12
    course = np.load(open(f'{session_id}/course_layer.npy', 'rb'))

    circles = hough_circle_fixed_radius(course, r)
    res = {'circles': []}
    for elem in circles:
        res['circles'].append({
            'x': elem[0],
            'y': elem[1],
            'r': r,
        })
    return res
