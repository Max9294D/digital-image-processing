###############################################################################
import numpy as np  # module pour la manipulation de matrice

import pylab as plt  # module pour affichage des données
# from matplotlib import pyplot as plt    # Module image propre à python
# from scipy.ndimage import label, generate_binary_structure

import cv2  # module pour la manimpulation d'image via OpenCV

###############################################################################
# Lecture d'une image & information sur l'image
# fname  = "voilier_oies_blanches.jpg"
# fname  = "img_ds.jpg"
# fname  = "Maya-and-Mantra.jpg"
fname = "Data/baboon.jpg"
# fname  = "house.jpg"
img_C = cv2.imread(fname)  # Lecture image en couleurs BGR

b, g, r = cv2.split(img_C)  # Recuperation des plan de couleurs BGR ou b=img_C[:,:,0] ...
img_C2 = cv2.merge([r, g, b])  # Reconstruction image format RGB       ou np.concatenate((b,g,r),axis=1)
imgRGB = cv2.cvtColor(img_C, cv2.COLOR_BGR2RGB)  # Passage BGR --> RGB via cv2.cvtColor (non utilisé par la suite)

# 2.1.1 Affichage image via opencv --> affichage format BGR
cv2.namedWindow("mon image BGR", cv2.WINDOW_NORMAL)  # Pour dimensionner la fenetre d'affichage
cv2.imshow("mon image BGR", img_C)
cv2.namedWindow("mon image RGB", cv2.WINDOW_NORMAL)
cv2.imshow("mon image RGB", img_C2)

# Affichage image via plt  --> affichage format RGB
plt.figure()
plt.subplot(121)
plt.imshow(img_C)
plt.xticks([]), plt.yticks([])
plt.title(" Mon image BGR")
plt.subplot(122)
plt.imshow(img_C2)
plt.xticks([]), plt.yticks([])
plt.title(" Mon image RGB")
plt.show()

# 2.1.2 Affichage des trois canaux séparéments
plt.figure()
plt.subplot(131)
plt.imshow(b, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.title(" Mon image B")
plt.subplot(132)
plt.imshow(g, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.title(" Mon image G")
plt.subplot(133)
plt.imshow(r, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.title(" Mon image R")
plt.show()

rMean = np.mean(r)
gMean = np.mean(g)
bMean = np.mean(b)

a = np.array((rMean,gMean,bMean))  #couleur moyenne de l'image

def distEuclid(a,z):
    return np.linalg.norm(z-a)


px = np.shape(img_C2)[0]
py = np.shape(img_C2)[1]
D = np.zeros((px,py))
D0 = 84 #seuil

for i in range(px):
    for j in range(py):
        if distEuclid(img_C2[i][j],a) < D0:
            D[i][j] = 1
        else:
            D[i][j] = 0

plt.figure()
plt.imshow(D, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.title(" Mon image Binarisée - Euclidienne")
plt.show()

sigmaR = np.std(r) #ecart type couleur rouge
sigmaG = np.std(g) #ecart type couleur vert
sigmaB = np.std(b) #ecart type couleur bleu

alpha = 1.25

Rmin = rMean - alpha*sigmaR
Gmin = gMean - alpha*sigmaG
Bmin = bMean - alpha*sigmaB

Rmax = rMean + alpha*sigmaR
Gmax = gMean + alpha*sigmaG
Bmax = bMean + alpha*sigmaB

D2 = np.zeros((px,py))

for i in range(px):
    for j in range(py):
        if (Rmin<=img_C2[i][j][0] and Rmax>=img_C2[i][j][0] and Gmin<=img_C2[i][j][1] and Gmax>=img_C2[i][j][1] and Bmin<=img_C2[i][j][2] and Bmax>=img_C2[i][j][2]):
            D2[i][j] = 1
        else:
            D2[i][j] = 0

plt.figure()
plt.imshow(D2, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.title(" Mon image Binarisée - Mahalanobis")
plt.show()


fname1 = "Data/crayons.jpg"

img_C = cv2.imread(fname1)  # Lecture image en couleurs BGR
cv2.imshow("Mon image", img_C)

b, g, r = cv2.split(img_C)  # Recuperation des plan de couleurs BGR ou b=img_C[:,:,0] ...
imgRGB = cv2.cvtColor(img_C, cv2.COLOR_BGR2RGB)  # Passage BGR --> RGB via cv2.cvtColor (non utilisé par la suite)



Rmin = 0
Rmax = 255
Gmin = 0
Gmax = 255
Bmin = 0
Bmax = 255


def binariseRGB(pos):


    # Estimation des seuils par barre de défilement
    Rmin = cv2.getTrackbarPos("Rmin", "mon Mask")
    Rmax = cv2.getTrackbarPos("Rmax", "mon Mask")

    Gmin = cv2.getTrackbarPos("Gmin", "mon Mask")
    Gmax = cv2.getTrackbarPos("Gmax", "mon Mask")

    Bmin = cv2.getTrackbarPos("Bmin", "mon Mask")
    Bmax = cv2.getTrackbarPos("Bmax", "mon Mask")

    # Définition des seuils
    lower = np.array([Rmin, Gmin, Bmin])
    upper = np.array([Rmax, Gmax, Bmax])

    print("Seuil Bas RGB:", lower)
    print("Seuil Haut RGB:", upper)

    img_Binaire = cv2.inRange(imgRGB, lower, upper)  # seuillage dans les trois plans
    img_Calque = cv2.bitwise_and(img_C, img_C, mask=img_Binaire)  # Pour faire super jolie ...

    # Affichage
    cv2.namedWindow("mon Mask", cv2.WINDOW_NORMAL)  # Pour dimensionner la fenetre d'affichage
    cv2.imshow("mon Mask", img_Binaire)

    cv2.namedWindow("mon Resultat", cv2.WINDOW_NORMAL)  # Pour dimensionner la fenetre d'affichage
    cv2.imshow("mon Resultat", img_Calque)


# Test
binariseRGB(0)

# Creation des barres de défilement
cv2.createTrackbar("Rmin", "mon Mask", Rmin, 255, binariseRGB)
cv2.createTrackbar("Rmax", "mon Mask", Rmax, 255, binariseRGB)

cv2.createTrackbar("Gmin", "mon Mask", Gmin, 255, binariseRGB)
cv2.createTrackbar("Gmax", "mon Mask", Gmax, 255, binariseRGB)

cv2.createTrackbar("Bmin", "mon Mask", Bmin, 255, binariseRGB)
cv2.createTrackbar("Bmax", "mon Mask", Bmax, 255, binariseRGB)

##### FIN
cv2.waitKey(0)
cv2.destroyAllWindows()
