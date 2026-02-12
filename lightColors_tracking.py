# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 16:18:55 2024

@author: maceo
"""

# *******************
# Librairies
# *******************
import time
from math import cos,sin,pi
import matplotlib.pyplot as plt
import cv2
# *******************
# Variables
# *******************
list_time=[]
data=[]
data1=[]
data2=[]
filtered_data=[]

#################### Fonctions ##################################

def suivi(video, HSV_bas, HSV_haut, pos_init_x = 0, pos_init_y = 0):
    # on capture la video
    cap = cv2.VideoCapture(video)
    # acquérir hauteur et largeur de webcam
    hauteur=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    largeur=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # def zone de suivi
    ret=True
    roi=pos_init_x,pos_init_y,200,200
    x, y, w, h = roi
    # Définition de la fenêtre de suivi
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    # def liste coordonnées
    liste_x = []
    liste_y = []
    compteur_fps = 0
    ## programme principale ##
    
    while True:
        ret, frame = cap.read()
        #frame=cv2.flip(frame,1)
        #frame= cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if not ret:
            break
        
        # Calcul de l'histogramme du flux video
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Création d'un masque pour filtrer les pixels dans la plage de couleurs définie
        color_mask = cv2.inRange(hsv_frame, HSV_bas, HSV_haut)
        #cv2.imshow('color_mask', color_mask)
        
        # Calcul de la nouvelle position de la ROI
        ret, roi = cv2.meanShift(color_mask, roi, term_crit)
        x, y, w, h = roi
        
        # Dessin du rectangle autour de la ROI
        img = cv2.rectangle(frame, (x, y), (x+w, y+h), 255, 2)
        cv2.imshow("suivi", img)
        
        # Sortie de la boucle si la touche 'q' est pressée
        if cv2.waitKey(1) == ord('q'):
            break
        liste_x.append(x+(w//2))
        liste_y.append(y+(h//2))
        compteur_fps += 1
        cv2.waitKey(50)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calcule la durée en secondes
    duree = frame_count / fps
    
    # Libération des ressources
    cap.release()
    cv2.destroyAllWindows()
    return liste_x,liste_y, compteur_fps/duree, duree, compteur_fps
    



#----------------------------------------------------------------
################# Programme Principale ##########################
#----------------------------------------------------------------
# Def plage HSV
HSV_bas = (0, 163, 163)    #|  code HSV pour balle de ping pong !!
HSV_haut = (255,255,255)   #| 
x,y, fps_tracking, duree, compteur_fps = suivi('C:/Users/maceo/Documents/Coffre/CPGE/TIPE/Identification balle/DATA/V2mathis.mp4', HSV_bas, HSV_haut, 1200, 500)
#print('les coordonnées de cam Mathis sont ',x,y)
print('il y a ', compteur_fps,' points capturés')
print('FPS = ', fps_tracking,'/ sur une duree de ', duree)
print('maththhhh')
print('x = ', x)
print('y = ', y)
x,y, fps_tracking, duree, compteur_fps = suivi('C:/Users/maceo/Documents/Coffre/CPGE/TIPE/Identification balle/DATA/V2.mp4', HSV_bas, HSV_haut, 300, 600)
#print('les coordonnées de cam Mass sont ', x, y)
print('il y a ', compteur_fps,' points capturés')
print('FPS = ', fps_tracking,'/ sur une duree de ', duree)
print('masssssssssss')
print('x = ', x)
print('y = ', y)