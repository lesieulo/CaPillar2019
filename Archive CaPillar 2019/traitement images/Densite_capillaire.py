# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 08:19:23 2018

@author: Etienne Monnier
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2

def aff(img,name):
    res = img 
    cv2.imshow(name, res)
    cv2.waitKey(0)
	
def histo(img):
    plt.figure()
    plt.hist(img.flatten(), bins='auto')  
    plt.title("Histogramme")
    plt.show()

def seuillage(image):
    ''' On crée un seuil pour essayer de mieux faire ressortir les capillaires  sur 
        une image dont on a gardé une seule composante de couleur
    
    entrée: image (matrice n*m)
    sortie: resultat (matrice n*m)
    
    '''
    resultat = image.copy()
    resultat[resultat>140] = 255
    resultat[resultat<=140] = 130
    return resultat

def vecteurImageSelonDroite(imageFiltree,a,b,x):
    ''' On récupère un vecteur de pixels selon une droite  
    
    entrées: imageFiltree (Matrice n*m), a (float), b (float), x (int array)
    sortie: vecteur (int array)
    
    '''
    y = np.round(x*a + b)
    vecteur = np.zeros(y.shape)
    for k  in range(len(x)):
        vecteur[k] = imageFiltree[int(y[k]),x[k]]
    return vecteur

def suppressionBruitVecteur(vecteur):
    ''' On retire les points ponctuels qui fausseraien le calcul de densité 
        Pour ce faire on parcourt le vecteur et les groupes de taille inférieur 
        à 5 pixels sont remplacer 
    
    entrée: vecteur (int array)
    sortie: vect (int array)
    
    '''
    vect = np.copy(vecteur)
    n = len(vect)
    k = 0
    bruit = 0
    while k<n-1:
        i = k+1
        if vect[k] == 255:
            while vect[i] != 255:
                bruit+=1
                if bruit==5 or i == n-1:
                    break
                i+=1
            if bruit<5:
                vect[k+1:i] = 255
        else:
            while vect[i] == 255:
                bruit+=1
                if bruit==5 or i == n-1:
                    break
                i+=1
            if bruit<5:
                vect[k+1:i] = 130
        k = i
        bruit = 0
    return vect                              

def densiteVecteurSelonDroite(vect,grossissement,A,B): # 1pixel est de longueur variant entre 0.21 et 0.31 mm
    ''' On essaye de calculer la densité sur le vecteur couleur ayant subit un seuillage
        Ainsi qu'une suppression de bruit et une fonction laissant que deux valeurs 
        On prendra comme longueur absolue d'un pixel 0.26mm 
    
    entrées: vect (int array), grossissement (int), A (int tuple), B (int tuple)
    sortie: densite (float)
    
    '''
    compteur = 0
    switch = 1
    if vect[0] != 255:
        compteur = 1
        switch = 0
    for k in range(1,len(vect)-2):
        if vect[k] == 255 and switch != 1:
            switch = 1
        if vect[k] == 130 and switch != 0:
            switch = 0
            compteur += 1
    longueur = np.sqrt((B[0]-A[0])**2+(B[1]-A[1])**2)*0.26
    return compteur*grossissement/longueur

def calculMoyenneDensiteSelonDroites(img,grossissement,A,B,c,x):
    ''' Fonction reprenant les fonctions précédentes pour retourner la densité
        capillaire moyenne selon plusieurs droites parallèles en capillaire/mm 
    
    entrées: img (matrice n*m*3 pour RVB), grossissement (int), A (int tuple), B (int tuple)
                n (int), x (int array)
    sortie: densiteMoyenne (float)
        
    '''
    a = (B[1]-A[1])/(B[0]-A[0])
    b = B[1]-a*B[0]
    L=20*np.arange(0,c)
    s = 0
    imageApresSeuillage = seuillage(img[:,:,1])
    for i in L:
        vect2 = suppressionBruitVecteur(vecteurImageSelonDroite(imageApresSeuillage,a,b-i,x))
        densite = densiteVecteurSelonDroite(vect2,grossissement,A,B)
        s+=densite
        print(densite)
    return s/c

def affichage(img,A,B,c):
    ''' Fonction permettant d'afficher l'image des capillaires selon la composante verte,
        l'image après seuillage ainsi que les différentes droites grâce auxquelles on calcule
        la densité moyenne.
        
    entrées: img (matrice n*m*3 pour RVB), A (int tuple), B (int tuple), c (int)
    sortie: None
    
    '''
    vert = img[:,:,1] #Etonnamment le filtre vert fait bien ressortir les capillaires
    a = (B[1]-A[1])/(B[0]-A[0])
    b = B[1]-a*B[0] 
    
    plt.figure()
    plt.imshow(seuillage(vert),cmap = 'gray') #Image après avoir effectué un seuillage
    plt.colorbar()
    plt.plot(x,a*x+b) #première droite
    for i in 20*np.arange(1,c):
        plt.plot(x,a*x+b-i,'red')  #Droites sur lesquelles on calcule les capillaires
    plt.figure()
    plt.imshow(vert,cmap = 'gray') #Image après filtre vert
    plt.colorbar()
    plt.plot(x,a*x+b)
    for i in 20*np.arange(1,c):
        plt.plot(x,a*x+b-i,'red')
    plt.show()
    
    

if __name__ == '__main__':
    
    # print("Entrez les coordonnées du premier point séparées par une virgule:") # (200,640)
    # A = tuple(int(x.strip()) for x in input().split(','))
    # print("Entrez les coordonnées du deuxième point séparées par une virgule:") # (1270,230)
    # B = tuple(int(x.strip()) for x in input().split(','))
    # print("Combien de droites pour calculer la moyenne ? (>0)")
    # c = int(input())
    
    # img = mpimg.imread("4D.png")
    # x = np.arange(200,1300) #On réduit la zone où calculer les capillaires (On évite les bords)
    # densite = calculMoyenneDensiteSelonDroites(img,100,A,B,c,x)
    # print("La densité est de",densite," capillaire(s) par mm")
    # affichage(img,A,B,c)
	
	#img = mpimg.imread("4D.png")
	
	
	PATH = "4D.jpg"
	img = cv2.imread(PATH)
	#img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	#aff(img_couleur,"image")
	#seuille=seuillage(img[:,:,1])
	#mpimg.imsave("temp.png", seuille)

	# bgr2hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	# bgr2lab=cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	# cv2.imwrite('img.png',img)
	# cv2.imwrite('bgr2hsv.png',bgr2hsv)
	# cv2.imwrite('bgr2lab.png',bgr2lab)
	# histo(img)
	
	# bgr = cv2.imread(PATH)	
	# lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
	# lab_planes = cv2.split(lab)
	# clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(gridsize,gridsize))
	# lab_planes[0] = clahe.apply(lab_planes[0])
	# lab = cv2.merge(lab_planes)
	# bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
	# cv2.imwrite('internet.png',bgr)
	
	im2d=cv2.imread('2D.jpg')
	im2g=cv2.imread('2G.jpg')
	im3d=cv2.imread('3D.jpg')
	im3g=cv2.imread('3G.jpg')
	im4d=cv2.imread('4D.jpg')
	im4g=cv2.imread('4G.jpg')
	im5d=cv2.imread('5D.jpg')
	im5g=cv2.imread('5G.jpg')
	images = [im2d,im2g,im3d,im3g,im4d,im4g,im5d,im5g]
	
	#histo(im2d[:,:,0])
	#histo(im2d[:,:,1])
	#histo(im2d[:,:,2])
	
	img=im2d
	imgNEB=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	cv2.imwrite('0.png',img[:,:,0])
	cv2.imwrite('1.png',img[:,:,1])
	cv2.imwrite('2.png',img[:,:,2])
	cv2.imwrite('noirblanc.png',imgNEB)
	cv2.imshow('NEB',imgNEB)
	
	



''' Améliorations:
    
-> Programme effectué pour l'image 4D, "seuillage" devra s'adapter selon l'éclairage de la photo
et ne pourra donc se faire sur une constante (paramètre seuil a rajouter dans la fonction)

-> L'équation de la droite pourra être obtenue en demandant à l'utilisateur de placer la première
droite sur le bout de l'ongle

-> Une fois la droite positionnnée on pourra calculer une densité pour chaque droite parallèle à
celle positionnée pour ensuite effectuer une moyenne sur les densités trouvées en introduisant différentes
pondérations

-> Temps de calcul beaucoup trop élévé à cause de nombreuses boucles "for, essayer de jouer sur le fait qu'on
manipule des tableaux numpy pour réduire la complexité temporelle

'''
