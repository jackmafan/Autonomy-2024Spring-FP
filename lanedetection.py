
import numpy as np
import cv2
from util import *
def color_based_segmentation(image):
    #blur 
    img_b_blur_50 = cv2.boxFilter(image, -1, (1,1))
    # rgb to gray
    gray = cv2.cvtColor(img_b_blur_50, cv2.COLOR_BGR2GRAY)
    #apply histogram equalization
    equalize_img = cv2.equalizeHist(gray)
    #use gaussian or box filter
    img_b_blur_50 = cv2.GaussianBlur(equalize_img, (3, 3), 50)
    # binary thresholding
    _, binary = cv2.threshold(img_b_blur_50, 250, 255, cv2.THRESH_BINARY)
    cv2.imwrite('test.jpg', binary)
    return binary


def detect_lane_lines(binary_image):
    # hough transform
    # canny
    edges = cv2.Canny(binary_image,50, 150, apertureSize=3)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=30, maxLineGap=10)
    
    return lines


def least_square_fit(points):
    # curve 
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    A = np.vstack([y**2, y, np.ones(len(y))]).T
    a, b, c = np.linalg.lstsq(A, x, rcond=None)[0]
    return a, b, c

def draw_lane_lines(image, lines):
    if lines is not None:
        points = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 10)
            for x1, y1, x2, y2 in line:
                points.append((x1, y1))
                points.append((x2, y2))
        if points:
            
            a, b, c = least_square_fit(points)
            # draw
            y_vals = np.arange(image.shape[0])
            x_vals = (a * (y_vals ** 2) + b * y_vals + c).astype(int)
            for (x, y) in zip(x_vals, y_vals):
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    cv2.circle(image, (x, y), radius=5, color=(255, 0, 0), thickness=-2)  
    return image

def main(image_path):
    image = read_image(image_path) 
    binary_image = color_based_segmentation(image)   
    lines = detect_lane_lines(binary_image)  
    result_image = draw_lane_lines(image, lines)   
    cv2.imwrite('test2.jpg',result_image)

    

if __name__ == "__main__":
    main("road10.png")

