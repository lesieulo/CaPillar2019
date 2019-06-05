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

hough_otsu = cv.HoughLines(img_otsu,30,np.pi/180,100)    #applique hough à otsu
						#img,rho,theta,threshold
						



# images = [img_otsu, img_clahe]
# titles = ["1","2"]	

# for i in range(2):
	# plt.subplot(2,1,i+1),plt.imshow(images[i],'gray')
	# plt.title(titles[i])
	# plt.xticks([]),plt.yticks([])
# plt.show()

print(hough_otsu.shape)

rho = hough_otsu[:,:,0]
theta = hough_otsu[:,:,1]
#plt.plot(theta,rho,'ro')     #graphe de hough
#plt.show()

x = np.linspace(0,180,180)
y = np.zeros(180)
for a in hough_otsu[:,:,1]:
	degre = int(a*180/np.pi)
	y[degre]+=1

plt.plot(x, y, 'ro')      #nombre d'occurences de chaque angle en degre
plt.show()


#essayer d'appliquer canny (contours) avant hough
#ou alors simplement échantilloner l'img

