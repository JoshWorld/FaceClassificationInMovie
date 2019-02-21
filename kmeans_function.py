import numpy as np
import cv2
from matplotlib import pyplot as plt

def show(Z):

    Z = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 1.0)
    ret,label,center=cv2.kmeans(Z,4,None,criteria,5,cv2.KMEANS_RANDOM_CENTERS)

    A = Z[label.ravel()==0]
    B = Z[label.ravel()==1]

    C = Z[label.ravel() == 2]
    D = Z[label.ravel() == 3]


    plt.scatter(A[:,0],A[:,1], c ='blue')
    plt.scatter(B[:,0],B[:,1],c = 'red')
    plt.scatter(C[:, 0], C[:, 1], c='green')
    plt.scatter(D[:, 0], D[:, 1], c='yellow')

    plt.scatter(center[:,0],center[:,1],s = 40,c = 'y', marker = 's')
    plt.xlabel('Height'),plt.ylabel('Weight')
    plt.show()
