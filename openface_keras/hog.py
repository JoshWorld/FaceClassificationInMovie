import cv2
import numpy as np

hog = cv2.HOGDescriptor()
im = cv2.imread('C:\\Users\\ADMIN\\PycharmProjects\\FaceClassificationInMovie\\openface_keras\\data\\hwang_jungmin\\hwang_jungmin_0001.jpg')
h = hog.compute(im)
print(np.array(h).shape)
