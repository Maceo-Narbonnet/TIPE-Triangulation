import numpy as np
import cv2 as cv
import glob
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#------------------------------------------------------------------------------------------------------
############################################ FONCTIONS ###############################################
#------------------------------------------------------------------------------------------------------

def acquerir_pts(chessboardSize, taille_carré_echequier, chemin_images1, chemin_images2):
    '''
    on lui rentre des photos de damier sous différents angles,
    l'algo obtient les différents points utiles pour la calibration

    Parameters
    ----------
    chessboardSize : tuple
        --> représente le nb de carré en largeur/longueur
    taille_carré_echequier : entier
        --> longueur en mm d'une arrete des carrés de l'échéquier
    chemin_images1 : string
        --> chemin du dossiers avec images échéquier prise par Cam1
    chemin_images2 : string
        --> chemin du dossiers avec images échéquier prise par Cam2

    Returns
    -------
    imgpoints1 : liste
        --> représente les points d'image des coins du damier pour Cam1
    imgpoints5 : liste
        --> représente les points d'image des coins du damier pour Cam5
    objpoints : liste
        --> coord 3D des coins du damier. Se base sur la taille réelle des carré du damier
    img1 : matrice d'image
        --> image capturée de Cam1 sous forme de matrice d'image
    img5 : matrice d'image
        --> image capturée de Cam5 sous forme de matrice d'image
    gray1 : matrice d'image
        --> représente img1 en nuance de gris
    gray5 : matrice d'image
        --> représente img5 en nuance de gris

    '''
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)
    objp = objp * taille_carré_echequier
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints1 = [] # 2d points in image plane.
    imgpoints5 = [] # 2d points in image plane.
    # chercher les images
    images1 = glob.glob(chemin_images1)   ## copie tout les chemins des fichiers qui suivent le modele entre ()
    images5 = glob.glob(chemin_images2)
    for images1, images5 in zip(images1,images5):
        img1 = cv.imread(images1)
        #img1 = cv.resize(img1, (int(img1.shape[1]/2), int(img1.shape[0]/2)))
        gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        img5 = cv.imread(images5)
        #img5 = cv.resize(img5, (int(img5.shape[1]/2), int(img5.shape[0]/2)))
        gray5 = cv.cvtColor(img5, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret1, corners1 = cv.findChessboardCornersSB(gray1, chessboardSize, None)
        ret5, corners5 = cv.findChessboardCornersSB(gray5, chessboardSize, None)
        # If found, add object points, image points (after refining them)
        if ret1 and ret5 == True:
            objpoints.append(objp)
            corners1 = cv.cornerSubPix(gray1, corners1, (11,11), (-1,-1), criteria)
            imgpoints1.append(corners1)
            
            corners5 = cv.cornerSubPix(gray5, corners5, (11,11), (-1,-1), criteria)
            imgpoints5.append(corners5)
            
            # Draw and display the corners
            cv.drawChessboardCorners(img1, chessboardSize, corners1, ret1)
            cv.imwrite('img1.png', img1)
            cv.drawChessboardCorners(img5, chessboardSize, corners5, ret5)
            cv.imwrite('img5.png', img5)
            cv.waitKey(3000)


    cv.destroyAllWindows()
    return imgpoints1, imgpoints5, objpoints, img1, img5, gray1, gray5


def calibration(objpoints, imgpoints1, imgpoints5, frameSize, gray1, gray5,img1, img5):
    '''
    on lui rentre les points 2d des coins et 3D pour faire une calibration stéréo des caméras.
    on obtient plusieurs parametres notamment intrinsèques et surtout extrinsèques relatifs aux positions des 2 caméras

    Parameters
    ----------
    objpoints : liste
        --> coord 3D des coins du damier. Se base sur la taille réelle des carré du damier
    imgpoints1 : liste
        --> représente les points d'image des coins du damier pour Cam1
    imgpoints5 : liste
        --> représente les points d'image des coins du damier pour Cam5
    frameSize : tuple
        représente la résolution de l'image. Pour les cam de Mass et Mathis on a (4032, 3024)
    gray1 : matrice d'image
        --> représente img1 en nuance de gris
    gray5 : matrice d'image
        --> représente img5 en nuance de gris
    img1 : matrice d'image
        --> image capturée de Cam1 sous forme de matrice d'image
    img5 : matrice d'image
        --> image capturée de Cam5 sous forme de matrice d'image

    Returns
    -------
    projMatrix1 : matrice
        --> représente la matrice de projection de la CAm1 pour passer d'un repère 2D à 3D    
    projMatrix5 : matrice
        --> représente la matrice de projection de la CAm5 pour passer d'un repère 2D à 3D
    '''
    ret, cameraMatrix1, dist1, rvecs1, tvecs1 = cv.calibrateCamera(objpoints, imgpoints1, frameSize, None, None)
    height1, width1, channels1 = img1.shape
    newCameraMatrix1, roi1 = cv.getOptimalNewCameraMatrix(cameraMatrix1, dist1, (width1, height1), 1, (width1, height1))
    
    ret, cameraMatrix5, dist5, rvecs5, tvecs5 = cv.calibrateCamera(objpoints, imgpoints5, frameSize, None, None)
    height5, width5, channels5 = img5.shape
    newCameraMatrix5, roi5 = cv.getOptimalNewCameraMatrix(cameraMatrix5, dist5, (width5, height5), 1, (width5, height5))
    
    ######################### Stereo Vision Calibration ######################################
    flags = 0
    flags = cv.CALIB_FIX_INTRINSIC
    # Here we fix the intrinsic camara matrixes SO that only Rot, Trns, Emat and Fmat are calculated.
    # Hence intrinsic parameters are the same
    criteria_stereo= (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl m.
    retStereo, newCameraMatrix1, dist1, newCameraMatrix5, dist5, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objpoints,
                        imgpoints1, imgpoints5, newCameraMatrix1, dist1, newCameraMatrix5, dist5, gray1.shape[::-1], criteria_stereo, flags)
    
    ######################## Stereo Rectification ############################################
    rectifyScale = 1
    rect1, rect5, projMatrix1, projMatrix5, Q, roi1, roi5 = cv.stereoRectify(newCameraMatrix1, dist1, newCameraMatrix5, dist5, gray1.shape[::-1], rot, trans, rectifyScale, (0,0))
    stereoMap1 = cv.initUndistortRectifyMap(newCameraMatrix1, dist1, rect1, projMatrix1, gray1.shape[::-1], cv.CV_16SC2)
    stereoMap5 = cv.initUndistortRectifyMap(newCameraMatrix5, dist5, rect5, projMatrix5, gray5.shape[::-1], cv.CV_16SC2)
    
    cv_file = cv.FileStorage('stereoMap.xml', cv.FILE_STORAGE_WRITE)
    cv_file.write('stereoMapL_x',stereoMap1[0])
    cv_file.write('stereoMapL_y',stereoMap5[1])
    cv_file.write('stereoMapR_x',stereoMap1[0])
    cv_file.write('stereoMapR_y',stereoMap5[1])
    cv_file.release()
    
    ############## UNDISTORTION #####################################################
    '''
    img = cv.imread('C:/Users/maceo/Documents/Coffre/CPGE/TIPE/Calibration/DATA/CamL-1913001/Capture_rognee (13).png')
    h,  w = img.shape[:2]
    newCameraMatrix1, roi = cv.getOptimalNewCameraMatrix(cameraMatrix1, dist1, (w,h), 1, (w,h))
    # Undistort
    dst = cv.undistort(img, cameraMatrix1, dist1, None, newCameraMatrix1)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imwrite('caliResult1.png', dst)
    # Undistort with Remapping
    mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix1, dist1, None, newCameraMatrix1, (w,h), 5)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imwrite('caliResult2.png', dst)  
    # Reprojection Error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints, _ = cv.projectPoints(objpoints[i], rvecs1[i], tvecs1[i], cameraMatrix1, dist1)
        error = cv.norm(imgpoints1[i], imgpoints, cv.NORM_L2)/len(imgpoints1)
        mean_error += error
    total_error = mean_error/len(objpoints)
    '''
    print('coeff de distorsion de la caméra 1 est de ', dist1)
    print('\ncoeff de distorsion de la camera 5 est de ', dist5)
    print('\nNew_Matrice de Camera1 :', newCameraMatrix1)
    print('\nNew_Matrice de Camera5 :', newCameraMatrix5)
    print('\nMatrice de Camera1 :', cameraMatrix1)
    print('\nMatrice de Camera5 :', cameraMatrix5)
    print('\nmatrice de projection 1 :', projMatrix1)
    print('\nmatrice de projection 5 :', projMatrix5)
    print('\nmatrice de rotation: ', rot)
    print('\nmatrice de translation : ', trans)
    print('\nmatrice essentielle :', essentialMatrix)
    #print('\nerreur totale : ', total_error)
    
    return projMatrix1, projMatrix5, rot, trans, newCameraMatrix1, newCameraMatrix5, dist1, dist5


def verifier_taille_listes(liste1, liste2, liste3, liste4):
    # Vérifier si les listes ont la même taille
    tailles = [len(liste1), len(liste2), len(liste3), len(liste4)]
    taille_min = min(tailles)
    taille_max = max(tailles)

    if taille_min != taille_max:
        # Supprimer les éléments en trop dans les listes plus grandes
        liste1 = liste1[:taille_min]
        liste2 = liste2[:taille_min]
        liste3 = liste3[:taille_min]
        liste4 = liste4[:taille_min]

    return liste1, liste2, liste3, liste4

def listes_to_coord(liste_x_mass, liste_y_mass, liste_x_math, liste_y_math):
    coord_mass = []
    coord_math = []
    for i in range(len(liste_x_mass)):
        coord_mass.append([0]*2)
        coord_mass[i][0] = liste_x_mass[i]
        coord_mass[i][1] = liste_y_mass[i]
        coord_math.append([0]*2)
        coord_math[i][0] = liste_x_math[i]
        coord_math[i][1] = liste_y_math[i]
    return coord_mass, coord_math
        
    
#------------------------------------------------------------------------------------------------------
########################### Programme Principale ######################################################
#------------------------------------------------------------------------------------------------------

imgpoints1, imgpoints5, objpoints, img1, img5, gray1, gray5 = acquerir_pts((8,13), 7, 
                                                             'C:/Users/maceo/Documents/Coffre/CPGE/TIPE/Calibration/DATA/Cam_MaceoJuin/*.jpg', 
                                                             'C:/Users/maceo/Documents/Coffre/CPGE/TIPE/Calibration/DATA/Cam_MathisJuin/*.jpg')
projMatrice1, projMatrice5, rot , trans, newCameraMatrix1, newCameraMatrix5, dist1, dist5 = calibration(objpoints, imgpoints1, imgpoints5, (4032,2268), gray1, gray5, img1, img5)   

print('projMatrice1', projMatrice1)
print('projMatrice5', projMatrice5)
################################
## Passage de pts 2D en pt 3D ##
###############################

# verifier que les listes sont de la bonne taille !
liste_x_mass, liste_y_mass, liste_x_math, liste_y_math = verifier_taille_listes([1700, 1700, 1700, 1700, 1700, 1700, 1700, 1700, 1700, 1700, 1700, 1700, 1700, 1700, 1700, 1700, 1700, 1700, 1700, 1700, 1700, 1700, 1700, 1700, 1700, 1700, 1700, 1700, 1700, 1820, 1729, 1485, 1272, 1111, 1008, 891, 814, 746, 664, 655],
                                                                                [900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 822, 762, 763, 771, 778, 767, 750, 734, 706, 688, 677],
                                                                                [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 154, 370, 564, 714, 842, 951, 1039, 1116, 1178], 
                                                                                [600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 569, 629, 673, 694, 702, 703, 694, 686, 672, 657, 642])

# passer des listes de x et y aux coordonnées !
coord_mass, coord_math = listes_to_coord(liste_x_mass, liste_y_mass, liste_x_math, liste_y_math)
coord_mass = np.array(coord_mass)
coord_math = np.array(coord_math)
coord_mathf=coord_math*1.0
coord_massf=coord_mass*1.0

## Normalisé les coordonnées

#coord_massf = cv.undistort(coord_massf, newCameraMatrix1, dist1)
#coord_mathf = cv.undistort(coord_mathf, newCameraMatrix5, dist5)

#####################
### Triangulation ###
#####################
pts_espace4 = cv.triangulatePoints(projMatrice1, projMatrice5, coord_massf.T, coord_mathf.T)
pts_espace =pts_espace4
print(pts_espace)
'''
pts_espace = pts_espace/pts_espace[3]
#pts_espace = cv.convertPointsFromHomogeneous(pts_espace4.T)
#pts_espace = pts_espace.tolist()




## passer dans le repère monde ##
liste_coord_espace = pts_espace
new_liste = [0]*len(liste_coord_espace)
for i in range(len(pts_espace)):
    liste_coord_espace[i] = pts_espace[i][0]

print('coord = ',liste_coord_espace)

liste_coordW = [0]*len(liste_coord_espace)
for i in range(len(liste_coord_espace)):
    liste_coordW[i] = np.dot(rot, np.array(liste_coord_espace[i])+ trans).tolist()[0]
    #liste_coordW = liste_coordW.tolist()
  

x = []
y = []
z = []
for i in range(len(pts_espace)):
    x.append(pts_espace[i][0])#[0])
    y.append(pts_espace[i][1])#[1])
    z.append(pts_espace[i][2])#[2])
    
print('x = ')
print(x)
print('y= ')
print(y)
print('z = ')
print(z)
'''  



# Supposons que 'pts_espace' est votre array numpy de points 3D
# pts_espace = np.array([[X1, X2, ..., XN], [Y1, Y2, ..., YN], [Z1, Z2, ..., ZN]])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

pts_espace = np.squeeze(pts_espace)
print(pts_espace)

#### Triche ####
# enlever les 2 derniers points qui étaient incohérent
'''
pts_espace[0] = np.array(pts_espace[0][:-2])
pts_espace[1] = np.array(pts_espace[1][:-2])
pts_espace[2] = np.array(pts_espace[2][:-2])
print('nouveau pts_espace sclicé ==> ', pts_espace)
'''


ax.scatter([0.82571429, 0.82533838 ,0.82606743,0.82660599 ,0.82686425, 0.82696286, 0.82697522, 0.82686436, 0.8262], 
 [-0.20429425 ,-0.20678857 ,-0.20195597,-0.19839938, -0.19669821, -0.19604953 ,-0.19596843, -0.19669824, -0.1987] ,
 [  -0.5257938 , -0.52540879 ,-0.52614231,-0.52664942 ,-0.52688222 ,-0.52696934,-0.52698019, -0.52688229, -0.5280 ])
'''
# Tracer le nuage de points
ax.scatter(pts_espace[0], pts_espace[1] , -pts_espace[2])
'''
ax.plot([0.8275, 0.8275], [-0.19596843,-0.19596843],[-0.515, -0.53], color = 'red' )

print('x = ', pts_espace[0])
print('y = ', pts_espace[1])
print('z = ', pts_espace[2])
# Nommer les axes
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Afficher le graphique
plt.show()