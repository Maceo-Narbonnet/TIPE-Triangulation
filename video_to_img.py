# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 19:17:28 2023

@author: maceo
"""
### Video-->Images ###

'''
le but est de transformer une vidéo en en image 
pour ensuite pouvoir utiliser chaque image
ce code est capapble de capturer les images d'une video de n'importe quelle fréquence
donc si la video est en 60 images/seconde alors le code capture 60images/seconde
'''

import cv2  # OpenCV


# Ouvrir la vidéo
video = cv2.VideoCapture("ball_tracking_example.mp4")

'''
import cv2

# Read image 
img = cv2.imread('D:/my-image.png')
 
# Show image
cv2.imshow('Example - Show image in window',img)
 
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # destroys the window showing image
'''

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