import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys

img = cv.imread(sys.argv[1])

#img = cv.imread("2D.jpg")
img = img[:,:,1]

eq = cv.equalizeHist(img)

clahe = cv.createCLAHE(clipLimit=5.0, tileGridSize=(10,10))
img_clahe_5_10 = clahe.apply(img)

clahe = cv.createCLAHE(clipLimit=15.0, tileGridSize=(10,10))
img_clahe_15_10 = clahe.apply(img)

clahe = cv.createCLAHE(clipLimit=50.0, tileGridSize=(10,10))
img_clahe_50_10 = clahe.apply(img)

clahe = cv.createCLAHE(clipLimit=5.0, tileGridSize=(20,20))
img_clahe_5_20 = clahe.apply(img)

clahe = cv.createCLAHE(clipLimit=15.0, tileGridSize=(20,20))
img_clahe_15_20 = clahe.apply(img)

clahe = cv.createCLAHE(clipLimit=50.0, tileGridSize=(20,20))
img_clahe_50_20 = clahe.apply(img)

images = [img_clahe_5_10, img_clahe_5_20, img_clahe_15_10, img_clahe_15_20, img_clahe_50_10, img_clahe_50_20, eq]
titles = ['img_clahe_5_10', 'img_clahe_5_20', 'img_clahe_15_10', 'img_clahe_15_20', 'img_clahe_50_10', 'img_clahe_50_20', 'eq']

for i in range(7):
	plt.subplot(4,2,i+1),plt.imshow(images[i],'gray')
	plt.title(titles[i])
	plt.xticks([]),plt.yticks([])
plt.show()