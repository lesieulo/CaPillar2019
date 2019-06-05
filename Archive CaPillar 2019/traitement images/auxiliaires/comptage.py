import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys
import scipy.ndimage

img = cv.imread(sys.argv[1])
img = img[:,:,1]

x0, y0 = 420, 390 # These are in _pixel_ coordinates!!
x1, y1 = 1038, 178

num = 200   #nombre points sur la droite
			#attention a bien definir pour pas avoir trop de bruit
			
nb_liss = 5  #nombre de fois qu'on lisse

x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)

# Extract the values along the line, using cubic interpolation
zi = scipy.ndimage.map_coordinates(img, np.vstack((y,x)))

# Extract the values along the line, nearest neighbour sampling
zn = img[y.astype(np.int), x.astype(np.int)]


def deriver(values):
	dx = 1
	derivees = [(values[i+1] - values[i])/dx for i in range(len(values)-1)]
	return derivees

def lisser(f):
	n = len(f)
	g = n*[0]
	for i in range(1,len(f)-1):
		g[i] = (f[i-1] + 2*f[i] + f[i+1])/4
	g[0] = (3*f[0] + f[1])/4
	g[n-1] = (f[n-2] + 3*f[n-1])/4
	return g

def lisser_n(f,n):
	lisse = f
	for i in range(n):
		lisse = lisser(lisse)
	return lisse


def cht(f):
	compteur = 0
	signe = 1
	graphe = len(f)*[0]
	for i in range(len(f)):
		if f[i] == 1 and signe == -1:
			compteur += 1
			signe = 1
			graphe[i] = 1
		if f[i] == -1 and signe == 1:
			signe = -1
	return compteur, graphe

compteur, graphe = cht(np.sign(deriver(lisser_n(zi,nb_liss))))
print("nombre de minima:", compteur)

plt.figure()

plt.subplot(121),plt.imshow(img,"gray")
plt.plot([x0,x1],[y0,y1], "r--")

plt.subplot(322), plt.plot(zi, "r"), plt.title("profil d'intensité")
plt.subplot(324), plt.plot(lisser_n(zi,nb_liss), "b"), plt.title("lissage du profil")
plt.subplot(326), plt.plot(graphe, "r"), plt.title("minima")

plt.show()