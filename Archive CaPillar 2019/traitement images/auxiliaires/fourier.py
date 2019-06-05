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
magnitude_spectrum = 20*np.log(np.abs(fshift))

ret2,fourier_otsu260 = cv.threshold(magnitude_spectrum,260,255,cv.THRESH_BINARY)
ret2,fourier_otsu270 = cv.threshold(magnitude_spectrum,270,255,cv.THRESH_BINARY)
ret2,fourier_otsu280 = cv.threshold(magnitude_spectrum,280,255,cv.THRESH_BINARY)
ret2,fourier_otsu290 = cv.threshold(magnitude_spectrum,290,255,cv.THRESH_BINARY)

#plt.subplot(3,2,1),plt.imshow(f,'gray')
#plt.title("fft")
#plt.subplot(3,2,2),plt.imshow(fshift,'gray')
#plt.title("fshift")
plt.subplot(3,2,3),plt.imshow(fourier_otsu260,'gray')
plt.title("260")
plt.subplot(3,2,4),plt.imshow(fourier_otsu270,'gray')
plt.title("270")
plt.subplot(3,2,5),plt.imshow(fourier_otsu280,'gray')
plt.title("280")
plt.subplot(3,2,6),plt.imshow(fourier_otsu290,'gray')
plt.title("290")
plt.show()