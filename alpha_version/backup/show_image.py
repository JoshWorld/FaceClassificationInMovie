import os
import cv2

frame_list = os.listdir('blur_frame')

c = 0
c2 = 0
for frame in frame_list:

    img = cv2.imread('not_blur_frame' + os.sep + frame)
    cv2.imshow(frame,img)

    cv2.waitKey(1000)
    cv2.destroyAllWindows()