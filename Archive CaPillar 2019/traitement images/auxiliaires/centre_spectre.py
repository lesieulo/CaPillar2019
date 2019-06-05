import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys

img = cv.imread(sys.argv[1])

#img = cv.imread("2D.jpg")
img = img[:,:,1]

clahe = cv.createCLAHE(clipLimit=50.0, tileGridSize=(20,20))
img_clahe = clahe.apply(img)

blur = cv.GaussianBlur(img_clahe,(15,15),0)
reta2,img_otsu = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)


f = np.fft.fft2(img_otsu)
fshift = np.fft.fftshift(f)
mag = 20*np.log(np.abs(fshift))

p = 11  #taille filtre gaussien dans l'img binarisee
#centre = blur = cv.GaussianBlur(mag[300:500,600:900],(p,p),0)
centre = blur = cv.GaussianBlur(mag[350:450,656:844],(p,p),0)

ret2,centre_bin = cv.threshold(centre,270,255,cv.THRESH_BINARY)
ret2,centre_bin2 = cv.threshold(centre,280,255,cv.THRESH_BINARY)
ret2,centre_bin3 = cv.threshold(centre,290,255,cv.THRESH_BINARY)
ret2,centre_bin4 = cv.threshold(centre,300,255,cv.THRESH_BINARY)



plt.subplot(2,3,1),plt.imshow(img,"gray")
plt.title("après filtre vert")
plt.subplot(2,3,2),plt.imshow(centre,cmap=plt.cm.hot)
plt.title("centre spectre après filtre gaussien")
plt.colorbar()
plt.subplot(2,3,3),plt.imshow(centre_bin,"gray")
plt.title("centre>270")
plt.subplot(2,3,4),plt.imshow(centre_bin2,"gray")
plt.title("centre>280")
plt.subplot(2,3,5),plt.imshow(centre_bin3,"gray")
plt.title("centre>290")
plt.subplot(2,3,6),plt.imshow(centre_bin4,"gray")
plt.title("centre>300")
plt.show()