# -*- coding: utf-8 -*-

"""
Tracking d'une balle grâce à Open CV

liens:
https://www.youtube.com/watch?v=RaCwLrKuS1w

"""
#-------------------------------# Bibliothèques #-----------------------------#

import cv2 as cv
import numpy as np



#---------------------------------# Fonctions #-------------------------------#




#-------------------------------# Main Programme #----------------------------#

### Lire la Video ###

videoCapture = cv.VideoCapture('C:/Users/maceo/Documents/Coffre/CPGE/TIPE/Identification balle/DATA/trajectoire_carre_Hough_v2.mp4')
cercle_precedent = None
dist = lambda x1, y1, x2, y2: (x1-x2)**2 + (y1-y2)**2  # lambda utiliser pour declarer des petites fonctions anonymes
x = []
y = []
coor = []
while True:
    ret, frame = videoCapture.read()    # on lit la video
    
    if not ret:                         # ret indique si la video est lisible
        break
    
    frame_gris = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)       # on grise la video
    frame_flou    = cv.GaussianBlur(frame_gris, (17, 17), 0)   # on floutte la video
    cercles = cv.HoughCircles(frame_flou, cv.HOUGH_GRADIENT, 1.2, 100, # transformée de Hough pour detetecter droites, cercles, ... avec la methode, le DP ?, la distance min
                              param1=100, param2=30, minRadius=20, maxRadius=80)   # la sensibilité, la précision, le cercle le + petit/grand qu'on peut detecter
    
    if cercles is not None:
        cercles = np.uint16(np.around(cercles))
        choix = None
        for i in cercles[0, :]:
            if choix is None: choix = i
            if cercle_precedent is not None:
                if dist(choix[0], choix[1], cercle_precedent[0], 
                        cercle_precedent[1]) <= dist(i[0], i[1], cercle_precedent[0], cercle_precedent[1]):
                    choix = i
        cv.circle(frame, (choix[0], choix[1]), 1, (0,100,100), 3)              #| detail fonction : https://www.geeksforgeeks.org/python-opencv-cv2-circle-method/
        cv.circle(frame, (choix[0], choix[1]), choix[2], (255,0,255), 3)       #|  
        cercle_precedent = choix

    cv.imshow("cercles", frame)
    
    coor.append((choix[0], choix[1]))
    x.append(choix[0])
    y.append(choix[1])
    
    if cv.waitKey(1) & 0xFF == ord('q'):    # commandes pour stopper la lecture (ici: espace)
        break
print(x, '\n', y)
videoCapture.release()
cv.destroyWindow('video')
videoCapture = cv.VideoCapture(0)
while True:
    ret, frame = videoCapture.read()    # on lit la video
    if not ret:                         # ret indique si la video est lisible
        break
    cv.imshow("video", frame)          
    if cv.waitKey(1) & 0xFF == ord('q'):    # commandes pour stopper la lecture (ici: espace)
        break
videoCapture.release()
cv.destroyWindow('video')
