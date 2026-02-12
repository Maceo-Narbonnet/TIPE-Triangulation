# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 14:53:05 2023

@author: timothee
"""

#### Algo Tracking ####

## Bibliothèques ##
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import PIL
import cv2 
'''
PARAMETRES:
A Determiner : calibrage photo ( format , nb pixel, taille image)
'''

## --------------------------Fonctions-----------------------------##

def detection ( image1 , image2 ):
    tableau = np.zeros (( len ( image1 ), len ( image1 [0]))) # initialisation du tableau
    for i in range ( len ( image1 )): # parcours des lignes
        for j in range ( len ( image1 [0])): # parcours des colonnes
            delta = 0 # initialisation de la valeur de delta
            for k in range(len( image1 [0][0])): # parcours des couleurs
                print(delta)
                delta += image1[i][j][k]- image2[i][j][k] # Calcul du delta du pixel (i ,j)
                if delta > 230:
                    delta =255
                else:
                    delta = 0
            tableau[i][j]= delta
    return ( tableau )

def changementfond(couleur, image):
    image_copy = np.copy(image)
    l, c = image_copy.shape[0], image_copy.shape[1]   # shape: [[2, 5, 48], [1,5,8]]  --> (2,3)
    r, g, b = couleur
    for i in range(l):
        for k in range(c):
            if 240 < image_copy[i][k][0] < 255 and 240 < image_copy[i][k][1] < 255 and 0 < image_copy[i][k][2] < 60:
                image_copy[i][k][0] = r
                image_copy[i][k][1] = g
                image_copy[i][k][2] = b
    return image_copy
                
                
def Video2Image(video):
    # Initialiser le compteur d'images
    compteur = 0
    # Boucler tant que la vidéo n'est pas terminée
    while video.isOpened():
        # Lire une image de la vidéo
        success, image = video.read()
        print(success)
        # Si la lecture a réussi
        if success:
            print(compteur)
            # Enregistrer l'image dans le dossier images avec le nom image_count.jpg
            cv2.imwrite('image_{}.jpg'.format(compteur), image)
            # Augmenter le compteur d'images
            compteur += 1
        # Sinon, arrêter la boucle
        else:
            break



def nettoyer_image(image):
    '''
    fonction qui prend en parametre l'image avec les deux objets en mouvement
    le BUT --> enlever tout les sois disant objet en mouvement qui ne sont pas une boule
    pour ça on enleve les objet repérés qui n'ont pas une certaine taille
    '''
    image_modif = np.copy(image)
    largeur, hauteur = len(image)-1, len(image[0])-1
    for y in range(len(image)-20):
        for x in range(len(image[0])-20):
            print(y,x)
            compteur = 0
            for i in range(-20, 21):    # commence a -20 et finis 20
                for j in range(-20, 21):
                    # Vérifier que les coordonnées sont dans les limites de l'image
                    if 0 <= x + i < largeur and 0 <= y + j < hauteur:
                        if image[y+j][x+i] == 0:
                            compteur += 1
            if compteur != 1600:
                for i in range(-20, 21):
                    for j in range(-20, 21):
                        image[y+j][x+i] == 255
    return image_modif
## -------------------------Programme Principal--------------------------##


## transformation image --> tableaux ##
image1 = Image.open("D:/PrépaTSI/2eme annee/TIPE/Identification balle/DATA/DATA_balletennis_mathis/Image_Mathis_1.jpg") #
im1 = np.array(image1)                                                                                                  #                   
                                                                                                                        #  Attention bien mettre des / dans ce sens
image2 = Image.open("D:/PrépaTSI/2eme annee/TIPE/Identification balle/DATA/DATA_balletennis_mathis/Image_Mathis_2.jpg") #
im2 = np.array(image2)                                                                                                  #


#cv2.imwrite('resultat.jpg', detection(im1, im2))
img = Image.open("D:/PrépaTSI/2eme annee/TIPE/Identification balle/resultat.jpg")
img_resized = img.resize((img.width // 2, img.height // 2))
img_resized.save("image_compressed.jpg", quality=1)
image_result = Image.open("D:/PrépaTSI/2eme annee/TIPE/Identification balle/image_compressed.jpg") 
im_result = np.array(image_result) 
cv2.imwrite('resultat_nettoyer.jpg', nettoyer_image(im_result))