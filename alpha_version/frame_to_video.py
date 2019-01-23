import cv2
import os

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

frame_list = os.listdir('frame')


for frame in frame_list:
    img = cv2.imread('frame' + os.sep + frame)
    out.write(img)

out.release()
cv2.destroyAllWindows()