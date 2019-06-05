import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys

img = cv.imread(sys.argv[1])

#img = cv.imread("2D.jpg")
img = img[:,:,1]

img_eq = cv.equalizeHist(img)

#clahe = cv.createCLAHE(clipLimit=5.0, tileGridSize=(10, 10))
clahe = cv.createCLAHE(clipLimit=15.0, tileGridSize=(20, 20))
img_clahe = clahe.apply(img)

p = 5   #taille filtre gaussien

#the = eq_otsu
#the2 = eq_gauss_otsu
#tha = clahe_otsu
#tha2 = clahe_gauss_otsu

# Otsu's thresholding  IMG EQ
rete,eq_otsu = cv.threshold(img_eq,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# Otsu's thresholding after Gaussian filtering
blur = cv.GaussianBlur(img_eq,(p,p),0)
rete2,eq_gauss_otsu = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

# Otsu's thresholding  IMG CLAHE
reta,clahe_otsu = cv.threshold(img_clahe,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# Otsu's thresholding after Gaussian filtering
blur = cv.GaussianBlur(img_clahe,(p,p),0)
reta2,clahe_gauss_otsu = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)


images = [img_eq, eq_otsu, eq_gauss_otsu, img_clahe, clahe_otsu, clahe_gauss_otsu]
titles = ['img_eq', 'eq_otsu', 'eq_gauss_otsu', 'img_clahe', 'clahe_otsu', 'clahe_gauss_otsu']	

for i in range(6):
	plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
	plt.title(titles[i])
	plt.xticks([]),plt.yticks([])
plt.show()