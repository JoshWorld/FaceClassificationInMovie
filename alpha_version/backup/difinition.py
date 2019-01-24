import cv2
import numpy as np
import os


frame_list = os.listdir('frame')

c = 0
c2 = 0
for frame in frame_list:
    img = cv2.imread('frame' + os.sep + frame)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t = np.max(cv2.convertScaleAbs(cv2.Laplacian(gray,3)))
    if t >= 100:
        cv2.imwrite('not_blur_frame/frame{}.jpg'.format(c),img)
        c += 1
    else:
        cv2.imwrite('blur_frame/frame{}.jpg'.format(c), img)
        c2 += 1

