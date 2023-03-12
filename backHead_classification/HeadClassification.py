from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import glob
import re


def main():
    face, backHead = data_load() # face_label = 0 / backHead_label = 1
    print("face dataset = %d, back head dataset = %d" % (len(face), len(backHead)))

    trainI_idx, testI_idx, trainL, testL = train_test_split(
        face + backHead, [0] * len(face) + [1] * len(backHead), test_size=0.25, random_state=41)
    trainI = [img[0].flatten() for img in trainI_idx]
    testI_flatten = [img[0].flatten() for img in testI_idx]
    testI = [img[0] for img in testI_idx]
    trainIdx = [img[1] for img in trainI_idx]
    testIdx = [img[1] for img in testI_idx]

    # Training & Result
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(trainI, trainL)
    print("KNN Accuracy = ", knn.score(testI_flatten, testL) * 100) # 79.06976744186046

    # Find misclassified images
    predictL = knn.predict(testI_flatten)
    count = 0
    for i in range(len(testL)):
        if predictL[i] != testL[i]:
            save_file = "misclassified_KNN/"
            cv2.imwrite(save_file + testIdx[i], testI[i])
            count += 1

    print("%d images out of %d images are misclassified" % (count, len(testL)))
    return knn

def data_load():
    faceImages = [] # label = 0
    backHeadImages = [] # label = 1
    for img in glob.glob("backHead_classification/face/*.png"):
        temp = cv2.imread(img)
        # temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        faceImages.append((resize_image(np.asarray(temp)), re.sub(r'[^0-9]', '', img) + ".png"))
    for img in glob.glob("backHead_classification/backHead/*.png"):
        temp = cv2.imread(img)
        # temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        backHeadImages.append((resize_image(np.asarray(temp)), re.sub(r'[^0-9]', '', img) + ".png"))
    return faceImages, backHeadImages

def resize_image(image, size=(100, 100)):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


if __name__ == '__main__':
    main()