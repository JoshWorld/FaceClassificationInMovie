import cv2
import numpy as np


img = cv2.imread('fff.PNG')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
t = np.max(cv2.convertScaleAbs(cv2.Laplacian(gray,3)))
print(t)