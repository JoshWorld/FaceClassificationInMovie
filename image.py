import cv2

img = cv2.imread('C:\\Users\ADMIN\\PycharmProjects\\FaceClassificationInMovie\\sinsegae2_face\\frame33.jpg')
original_img = img.copy()
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (80, 80), interpolation=cv2.INTER_CUBIC)
cv2.imshow('t',img)
cv2.waitKey(0)