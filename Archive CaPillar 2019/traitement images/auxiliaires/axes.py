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
# pour conserver le rapport de l'img de base

#print(centre.shape)

seuil = 280  # seuil pour le centre du spectre
ret2,centre_bin = cv.threshold(centre,seuil,255,cv.THRESH_BINARY)

aux = np.uint8(centre_bin)  #a voir si on peut pas faire ca avant
_, contours, _ = cv.findContours(aux, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
#print(type(contours))
#print(contours)
#print(len(contours))



# Construct a buffer used by the pca analysis
print("nombre de formes: ", len(contours))
if len(contours) == 3:   #si 3 formes, on prend la derniere (la plus grosse)
	pts = contours[2]
else:
	pts = contours[0]
sz = len(pts)
data_pts = np.empty((sz, 2), dtype=np.float64)
for i in range(data_pts.shape[0]):
	data_pts[i,0] = pts[i,0,0]
	data_pts[i,1] = pts[i,0,1]
# Perform PCA analysis
mean = np.empty((0))
mean, eigenvectors = cv.PCACompute(data_pts,mean)
x1, y1, x2, y2 = eigenvectors.flatten()
print("vecteur de la mire:", x1, y1)
x, y = 100,50
x_img, y_img = 400, 300

plt.subplot(2,1,1),plt.imshow(img,"gray")
plt.plot([x_img+0,x_img+500*x1],[y_img+0,y_img+500*y1], "b")
plt.plot([x_img+0,x_img+500*x2],[y_img+0,y_img+500*y2], "r")
plt.title("après filtre vert")
plt.subplot(2,1,2),plt.imshow(centre_bin,"gray")
plt.plot([x+0,x+60*x1],[y+0,y+60*y1], "b")
plt.plot([x+0,x+30*x2],[y+0,y+30*y2], "r")
plt.title("centre>280")  #seuil

plt.show()
