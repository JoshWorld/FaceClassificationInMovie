import cv2


def video_to_frame(video_path):
  vidcap = cv2.VideoCapture(video_path)

  success,image = vidcap.read()
  count = 0

  while success:
    success,image = vidcap.read()
    cv2.imwrite("/frame/frame%d.jpg" % count, image)     # save frame as JPEG file
    count += 1

