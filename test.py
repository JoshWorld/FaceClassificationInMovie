import cv2

vidcap = cv2.VideoCapture("C:\\Users\ADMIN\PycharmProjects\FaceDR\\test_data\sinsegae.mp4")

success,image = vidcap.read()
count = 0

while success:
  success,image = vidcap.read()
  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
  count += 1