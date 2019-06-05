import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys
import scipy.ndimage


#dans toutes les fonctions, le parametre show permet
#l'affichage du resultat. par defaut show=0, pas d'affichage.


#---------------------------
# OPENCV: FILTRES ET FOURIER
#---------------------------

def gris(img, show=0):          #img en niveaux de gris
	img = img[:,:,1]
	
	if show:
		plt.imshow(img,'gray')
		plt.title("img en niveaux de gris")
		plt.show()
	return img

	
def eq(img, show=0):        #img apres egalisation d'histo
	img = cv.equalizeHist(img)
	
	if show:
		plt.imshow(img,'gray')
		plt.title("img apres equalizeHist")
		plt.show()
	return img


def eq_clahe(img, a=50.0, b=20, show=0):   #img apres egalisation d'histo CLAHE
	###a: (?)
	###b: taille des sous-images methode CLAHE (?)

	clahe = cv.createCLAHE(clipLimit=a, tileGridSize=(b,b))
	img = clahe.apply(img)
	
	if show:
		plt.imshow(img,'gray')
		plt.title("img apres egalisation CLAHE, a="+str(a)+", b="+str(b))
		plt.show()
	return img


def otsu(img, dim=15, show=0):       #img lissee et binarisee avec OTSU
	###dim: dimension du filtre gaussien

	blur = cv.GaussianBlur(img,(dim,dim),0).astype('uint8')  #astype rajouté le 6/4/19
	reta2,img_otsu = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
	
	if show:
		plt.imshow(img_otsu,'gray')
		plt.title("img binaire apres Otsu, dim="+str(dim))
		plt.show()
	return img_otsu

	
def fourier(img, show=0):        #TF en amplitude de img
	f = np.fft.fft2(img)
	fshift = np.fft.fftshift(f)
	
	#ma methode
	mag = 20*np.log(np.abs(fshift))
	
	#methode helene thomas
	mag2 = np.log2(1+np.abs(fshift))

	if show:
		plt.subplot(221)
		plt.imshow(mag,'gray')
		plt.title("spectre en amplitude ma methode")
		plt.subplot(223)
		plt.imshow(img,'gray')
		plt.title("img")
		plt.subplot(122)
		plt.imshow(mag2, origin=('upper'), extent=(-0.5,0.5,0.5,-0.5), cmap = 'gray')
		plt.title("spectre en amplitude normalise")
		plt.show()
	return mag


#---------------
# AXE DE L'IMAGE
#---------------

def spectre_centre_bin(mag, p=11, seuil=280, show=0):     #centre, lisse et binarise le spectre
	###p: taille du filtre gaussien
	###seuil: pour la binarisation

	centre = blur = cv.GaussianBlur(mag[350:450,656:844],(p,p),0)
	ret2,centre_bin = cv.threshold(centre,seuil,255,cv.THRESH_BINARY)
	
	if show:
		plt.imshow(centre_bin,'gray')
		plt.title("spectre centre, lisse, binarise, p="+str(p)+", seuil="+str(seuil))
		plt.show()
	return centre_bin


def axe(centre_bin, show=0):        #renvoie le vecteur (X,Y) de l'axe du spectre binarise
	aux = np.uint8(centre_bin)
	_, contours, _ = cv.findContours(aux, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

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
	
	# Display
	if show:
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
	return x1, y1

	
#--------------------------------------
# COMPTAGE DES CAPILLAIRES SUR UNE MIRE
#--------------------------------------
	
def lisser(f, show=0):    #lisse une fonction sous forme de liste
	n = len(f)
	g = n*[0]
	for i in range(1,len(f)-1):
		g[i] = (f[i-1] + 2*f[i] + f[i+1])/4
	g[0] = (3*f[0] + f[1])/4
	g[n-1] = (f[n-2] + 3*f[n-1])/4
	
	if show:
		plt.figure()
		plt.subplot(121),plt.plot(f,"gray")
		plt.title("fonction")
		plt.subplot(122),plt.plot(g,"gray")
		plt.title("fonction lissee")
		plt.show()
	return g


def lisser_n(f, n, show=0):      #lisse n fois
	lisse = f
	for i in range(n):
		lisse = lisser(lisse)
		
	if show:
		plt.figure()
		plt.subplot(121),plt.plot(f,"gray")
		plt.title("fonction")
		plt.subplot(122),plt.plot(lisse,"gray")
		plt.title("fonction lissee "+str(n)+" fois")
		plt.show()
	return lisse


def deriver(values, show=0):    #derive une fonction sous forme de liste
	dx = 1
	derivee = [(values[i+1] - values[i])/dx for i in range(len(values)-1)]
	
	if show:
		plt.figure()
		plt.subplot(121),plt.plot(f,"gray")
		plt.title("fonction")
		plt.subplot(122),plt.plot(derivee,"gray")
		plt.title("fonction derivee")
		plt.show()
	return derivee


def cht(f, show=0):        #compte le nombre de changements de signe de f
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
	if show:
		plt.figure()
		plt.subplot(121),plt.plot(f,"gray")
		plt.title("fonction")
		plt.subplot(122),plt.plot(graphe,"gray")
		plt.title("nombre de changements de signe: "+str(compteur))
		plt.show()		
	
	return compteur, graphe

	
def comptage(img, x0, y0, x1, y1, num=150, nb_liss=5, show=0):   #compte les capillaires suivant une mire
	###num: nombre de points sur la mire
	###nb_liss: nombre lissages successifs
	
	x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)

	# Extract the values along the line, using cubic interpolation
	zi = scipy.ndimage.map_coordinates(img, np.vstack((y,x)))

	# Extract the values along the line, nearest neighbour sampling
	zn = img[y.astype(np.int), x.astype(np.int)]
	
	zi_lisse = lisser_n(zi,nb_liss)
	zi_derivee = deriver(zi_lisse)
	
	compteur, graphe = cht(np.sign(zi_derivee))
	
	if show:
		plt.figure()
		
		plt.subplot(121),plt.imshow(img,"gray")
		plt.plot([x0,x1],[y0,y1], "r--")

		plt.subplot(322), plt.plot(zi, "r"), plt.title("profil d'intensité")
		plt.subplot(324), plt.plot(zi_lisse, "b"), plt.title("lissage du profil")
		plt.subplot(326), plt.plot(graphe, "r"), plt.title("minima")

		plt.show()
	
	
	
	### a MAJ selon la camera, le grossissement choisi
	#longueur_cote_fenetre =
	#nbr_px_sur_un_cote =   (1280*1024)
	#grossissement = 95
	# px_en_mm = longueur_cote_fenetre / nbr_px_sur_un_cote /grossissement
	px_en_mm = 0.26
	
	longueur_px = np.sqrt((x1-x0)**2+(y1-y0)**2)
	longueur = longueur_px * px_en_mm
	densite = compteur / longueur
	
	print(" nombre de capillaires:", compteur)
	print(" longueur de la mire:", round(longueur,2), "mm")
	print(" densite sur la mire:", round(densite,3), "/mm")
	
	return compteur

	
#-------------------------------------
# TRAITEMENT DES CAPILLAIRES PONCTUELS
#-------------------------------------

def dessiner_rond(img, rayon, show=0):
	xmax, ymax = img.shape[1], img.shape[0] #x abscisses, y ordonnees
	x0, y0 = xmax//2, ymax//2
	
	masque = np.zeros((ymax, xmax))
	
	if show:
		plt.imshow(img,'gray')
		plt.plot([x0, x0+10],[y0, y0+10], "r")
		
		cercle =  plt.Circle((x0, y0), 150, color='r')
		fig = plt.gcf()
		ax = fig.gca()
		ax.add_artist(cercle)
		
		plt.title("blabla")
		plt.show()
	
	return xmax, ymax


def filtre_disc(img, r=250, show=0):
	n, p = img.shape
	i, j = n//2, p//2+150
	y,x = np.ogrid[-i:n-i, -j:p-j]
	masque = x*x + y*y <= r*r
	
	new = np.zeros((n, p))
	new[masque] = img[masque]
	
	
	if show:
		plt.imshow(new,'gray')
		plt.title("sélection cercle des capillaires détectés")
		plt.show()

	return new


def fermeture(img, t, show=0):

	kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(t,t))
	closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
	
	if show:
		plt.imshow(closing,'gray')
		plt.title("closing")
		plt.show()
		
	return closing

def ouverture(img, t, show=0):

	kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(t,t))
	opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
	
	if show:
		plt.imshow(opening,'gray')
		plt.title("opening")
		plt.show()
		
	return opening

def p(list1, ttl1, list2, ttl2):
	for i in range(len(list1)):
		plt.subplot(2,4,i+1)
		plt.imshow(list1[i],'gray')
		plt.title(ttl1[i])
	for i in range(len(list2)):
		plt.subplot(2,4,i+5)
		plt.imshow(list2[i],'gray')
		plt.title(ttl2[i])
	plt.show()
	return

def transfo(img):
	min = img.min()
	max = img.max()
	a = 255 / (max-min)
	b = 255*(min/(min-max))
	return (a * img + b)

def traitement_capi_ponctuels(img, tclahe=60, t1=19, t2=23, show=0, show_zoom=0):
	img = gris(img)
	img_clahe = eq_clahe(img, b=tclahe)
	img_otsu = otsu(img_clahe)
	
	imgt1 = fermeture(img_otsu, t1)
	imgt2 = fermeture(img_otsu, t2)
	imgdif = abs(255-(imgt2 - imgt1))
	imgreste = fermeture(imgdif, t1)
	imgfin = 255 * (255 - imgt2 - imgreste)
	img[img[:,:]==128] = 0
	
	
	if show:
		liste1 = [img, img_clahe, img_otsu, imgt1]
		liste2 = [imgt2, imgdif, imgreste, imgfin]
		titles1 = ["A:img", "B:clahe(A)", "C:otsu(B)", "D:fermeture(C,t1)"]
		titles2 = ["E:fermeture(C,t2)", "F:diff(D,E)", "G:fermeture(F)", "H:E+G"]
		p(liste1, titles1, liste2, titles2)
		
	if show_zoom:
		plt.subplot(121)
		plt.imshow(img,'gray')
		plt.plot([600, 600, 800, 800,600],[400,800,800,400,400], "b")
		plt.title("img originale")
		plt.subplot(143)
		plt.imshow(img[400:800,600:800],'gray')
		plt.title("zoom")
		plt.subplot(144)
		plt.imshow(imgfin[400:800,600:800],'gray')
		plt.title("algo de détection")
		plt.show()
	
	return imgfin

	
	
	
def compte_blobs(img, zoom=1, show=0):
	
	img_traitee = traitement_capi_ponctuels(img)
	
	if zoom:
		img_traitee = img_traitee[400:800,600:800]
	
	t = 10
	kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(t,t))
	img_dil = cv.dilate(img_traitee,kernel,iterations = 1)
	img_dil = 255 - img_dil
	frame, contours, hierarchy = cv.findContours(img_dil, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	n = len(contours)

	areas = []

	for i in range(0, n):
		areas.append(cv.contourArea(contours[i]))
	mass_centres_x = []
	mass_centres_y = []

	for i in range(0, n):
		M = cv.moments(contours[i], 0)
		mass_centres_x.append(int(M['m10']/M['m00']))
		mass_centres_y.append(int(M['m01']/M['m00']))

	print('Nombre de capillaires ponctuels: ', str(n))

	if show:
		img =  cv.cvtColor(img, cv.COLOR_BGR2RGB)
		if zoom:
			plt.imshow(img[400:800,600:800], 'gray')
		else:
			plt.imshow(img, 'gray')
		plt.title('Capillaires détectés:'+str(n))
		theta = np.linspace(0, 2*np.pi, 40)
		for k in range(len(contours)):
			x = mass_centres_x[k] + 1*np.cos(theta)
			y = mass_centres_y[k] + 1*np.sin(theta)
			plt.plot(x,y)
		plt.show()
	
	return n
	
def traitement_capi_ponctuels2(img, tclahe=60, t1=19, t2=23, show=0, show_zoom=0):
	img = gris(img)
	img_clahe = eq_clahe(img, b=tclahe)
	img_otsu = otsu(img_clahe)
	
	imgt1 = fermeture(img_otsu, t1)
	imgt2 = fermeture(img_otsu, t2)
	imgdif = abs(255-(imgt2 - imgt1))
	imgreste = fermeture(imgdif, t1)
	imgfin = 255 * (255 - imgt2 - imgreste)
	img[img[:,:]==128] = 0
	
	
	if show:
		liste1 = [img, img_clahe, img_otsu, imgt1]
		liste2 = [imgt2, imgdif, imgreste, imgfin]
		titles1 = ["A:img", "B:clahe(A)", "C:otsu(B)", "D:fermeture(C,t1)"]
		titles2 = ["E:fermeture(C,t2)", "F:diff(D,E)", "G:fermeture(F)", "H:E+G"]
		p(liste1, titles1, liste2, titles2)
		
	if show_zoom:
		plt.subplot(121)
		plt.imshow(img,'gray')
		plt.plot([600, 600, 800, 800,600],[400,800,800,400,400], "b")
		plt.title("img originale")
		plt.subplot(143)
		plt.imshow(img[400:800,600:800],'gray')
		plt.title("zoom")
		plt.subplot(144)
		plt.imshow(imgfin[400:800,600:800],'gray')
		
		img = cv.imread(sys.argv[1])
		img_traitee = traitement_capi_ponctuels(img)
		img_traitee = img_traitee[400:800,600:800]
		t = 10
		kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(t,t))
		img_dil = cv.dilate(img_traitee,kernel,iterations = 1)
		img_dil = 255 - img_dil
		frame, contours, hierarchy = cv.findContours(img_dil, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
		n = len(contours)
		areas = []
		for i in range(0, n):
			areas.append(cv.contourArea(contours[i]))
		mass_centres_x = []
		mass_centres_y = []
		for i in range(0, n):
			M = cv.moments(contours[i], 0)
			mass_centres_x.append(int(M['m10']/M['m00']))
			mass_centres_y.append(int(M['m01']/M['m00']))
		theta = np.linspace(0, 2*np.pi, 40)
		for k in range(len(contours)):
			x = mass_centres_x[k] + 3*np.cos(theta)
			y = mass_centres_y[k] + 3*np.sin(theta)
			plt.plot(x,y)
		
		plt.title("algo de détection")
		plt.show()
	
	return imgfin
	
	
	
	

if __name__ == "__main__":

	'''
	img = cv.imread(sys.argv[1])
	f1 = filtre_disc(gris(img))
	f2 = filtre_disc(traitement_capi_ponctuels(img))
	
	plt.subplot(121)
	plt.imshow(f1,'gray')
	plt.title("cercle image originale")
	
	plt.subplot(122)
	plt.imshow(f2,'gray')
	plt.title("cercle capillaires détectés")
	plt.show()
	'''
	
	
	
	
	img = cv.imread(sys.argv[1])
	#compte_blobs(img, zoom=0, show=1)
	traitement_capi_ponctuels2(img, show_zoom=1)
	
	
	'''
	x0, y0 = 420, 390 # These are in _pixel_ coordinates!!
	x1, y1 = 1038, 178
	comptage(img,x0, y0, x1, y1, nb_liss=4, num=200)
	'''
	
	
	
