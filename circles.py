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

def check_circle_cv_kernel(img, kernel, match_threshold, visualize=False, print_score=False):
    res_square = np.array(img / 255, dtype=np.float64)
    if visualize:
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)
        ax1.imshow(res_square)
        ax2.imshow((kernel + 1) / 2)
        plt.show()

    score = np.multiply(kernel, res_square).sum()
    if print_score:
        print(score, match_threshold)
    return score > match_threshold

def hough_circle_fixed_radius(img, r, candidate_threshold, match_threshold, visualize=False):
    res = np.zeros_like(img, dtype=np.int32)
    n, m = img.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
    course = cv2.dilate(img, kernel)
    plt.imshow(course)
    plt.show()
    
    for i in tqdm.tqdm(range(0, n)):
        for j in range(0, m):
            if course[i,j] == 255:
                res += circle_mask((i,j,r), course.shape).reshape((n, m))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,8))
    course = cv2.erode(course, kernel)
    kernel = np.float32(cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2 * r + 1, 2 * r + 1)))
    kernel[(r//4):-(r//4), (r//4):-(r//4)] -= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1-r//4-r//4,2*r+1-r//4 - r//4))
    kernel[(r//2):-(r//2), (r//2):-(r//2)] -= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(r+2, r+2))
    
    circles = []
    for i in tqdm.tqdm(range(r, n-r)):
        for j in range(r, m-r):
            duplicate = False
            for filtered_circle in circles:
                if np.linalg.norm(np.array([i, j]) - filtered_circle) < r * 2:
                    duplicate = True
                    break
            if duplicate:
                continue

            if res[i,j] >= candidate_threshold and check_circle_cv_kernel(course[i-r:i+r+1, j-r:j+r+1], kernel, match_threshold, visualize=visualize, print_score=visualize):
                circles.append(np.array([i, j]))
    if visualize:
        n_res = np.zeros(res.shape)
        for circle in circles:
            n_res += circle_mask((circle[0],circle[1],r), course.shape).reshape((n, m))
        plt.imshow(n_res, cmap='gray')
        plt.show()
    return circles


def get_circles_impl(session_id, r, candidate_threshold, match_threshold):
    course = np.load(open(f'sessions/{session_id}/course_layer.npy', 'rb'))

    circles = hough_circle_fixed_radius(course, r, candidate_threshold, match_threshold)
    res = {'circles': []}
    for elem in circles:
        res['circles'].append({
            'x': int(elem[0]),
            'y': int(elem[1]),
            'r': r
        })
    return res
