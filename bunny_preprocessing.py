import cv2
import numpy as np

def bunny_preprocessing(name):
    img = cv2.imread(name)
    (h, w, c) = img.shape

    for i in range(h):
        for j in range(w):
            if np.array_equal(img[i][j], [255, 255, 255]):
                img[i][j] = np.array([0, 0, 0])

    cv2.imwrite("pre_" + name, img)
