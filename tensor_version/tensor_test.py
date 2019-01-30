


import cv2
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt

w=10
h=10
fig=plt.figure(figsize=(10, 10))
columns = 5
rows = 6
for i in range(1, columns*rows +1):
    img = np.random.randint(10, size=(h,w))
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
plt.show()