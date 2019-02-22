import cv2
import numpy as np

hog = cv2.HOGDescriptor()
im = cv2.imread('C:\\Users\\ADMIN\\PycharmProjects\\FaceClassificationInMovie\\blackpink_error.jpg')

h = hog.compute(im)

n = np.resize(h,(3,im.shape[0], im.shape[1]))
print(n)


print()