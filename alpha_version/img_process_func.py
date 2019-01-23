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



video_to_frame('C:\\Users\\ADMIN\\PycharmProjects\\FaceClassificationInMovie\\test_data\\sample2.mp4')