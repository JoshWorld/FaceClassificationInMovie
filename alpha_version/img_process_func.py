import cv2
import numpy as np


def video_to_frame(video_path):
  vidcap = cv2.VideoCapture(video_path)

  success,image = vidcap.read()
  count = 0

  while success:
    success,image = vidcap.read()
    cv2.imwrite("frame/frame%d.jpg" % count, image)     # save frame as JPEG file
    count += 1


def feature_matching_orb(img1, img2):

  res = None
  orb = cv2.ORB_create()
  kp1, des1 = orb.detectAndCompute(img1, None)
  kp2, des2 = orb.detectAndCompute(img2, None)

  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
  matches = bf.match(des1, des2)
  matches = sorted(matches, key=lambda x: x.distance)
  for item in matches:
    print(item)



  res = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], res, flags=2)
  res = cv2.resize(res, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
  cv2.imshow('t',res)
  cv2.waitKey(0)


vidcap = cv2.VideoCapture('C:\\Users\ADMIN\PycharmProjects\FaceClassificationInMovie\\test_video\\sinsegae.mp4')

ret, pre_frame = vidcap.read()
while ret:
  ret, frame = vidcap.read()

  frame_copy1 = np.array(frame.copy())

  frame = (img - np.mean(img)) / np.std(img)



  imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  _, thresh = cv2.threshold(imgray, 127, 255, 0)
  contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  cv2.drawContours(pre_frame, contours, -1, (0, 255, 0), 1)

  cv2.imshow('t',pre_frame)
  cv2.waitKey(0)


  #feature_matching_orb(frame, pre_frame)

  pre_frame = frame




feature_matching_orb('C:\\Users\ADMIN\\PycharmProjects\\FaceClassificationInMovie\\alpha_version\\frame\\frame0.jpg','C:\\Users\ADMIN\\PycharmProjects\\FaceClassificationInMovie\\alpha_version\\frame\\frame2.jpg')

#video_to_frame('C:\\Users\\ADMIN\\PycharmProjects\\FaceClassificationInMovie\\test_video\\sinsegae.mp4')