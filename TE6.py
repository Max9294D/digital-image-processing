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
fname = "Data/riviere.JPG"
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

# 2.1.3 en amenant la souris sur les couleurs verte, rouge, jaune, bleue, noire, blanche
# de l'image house par exemple, on vérifie que les coordonnées RGB de ces couleurs
# sont cohérentes.

###############################################################################
# 2.2.1 -> 2.2.4 Conversion HSV
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html
# https://www.learnopencv.com/color-spaces-in-opencv-cpp-python/
imgHSV = cv2.cvtColor(img_C, cv2.COLOR_BGR2HSV)  # Passage GBR --> HSV
h, s, v = cv2.split(imgHSV)  # Extraction des trois plans ou  h=imgHSV[:,:,0] ...

# 2.2.3 Affichage des trois canaux séparéments
plt.figure()
plt.subplot(131)
plt.imshow(h, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.title(" Mon image H Teinte")
plt.subplot(132)
plt.imshow(s, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.title(" Mon image S Saturation")
plt.subplot(133)
plt.imshow(v, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.title(" Mon image V Valeur")
plt.show()
# Remarque : les images de Teinte, Saturation et Valeur sont des scalaires [0 255] sauf
# H varies de 0 to 179 (théoriquement, la teinte est un angle qui varie
# entre 0 et 360°, qui est divisé par 2 sous OpenCV pour pouvoir être codé sur 8 bits...)

# 2.2.5 En amenant la souris sur les couleurs verte, rouge, jaune, bleue, noire, blanche
# des plans images H, S et V,on vérifie que les valeurs HSV de ces couleurs
# sont cohérentes. Par exemple pour le blanc, S est proche de 0 et V proche de 255
# pour le jaune, H est proche de 30, ...
cv2.waitKey(0)  # Attente appuie d'une touche
cv2.destroyAllWindows()
plt.close('all')

###############################################################################
# Affichage image initial
cv2.namedWindow("mon image BGR", cv2.WINDOW_NORMAL)  # Pour dimensionner la fenetre d'affichage
cv2.imshow("mon image BGR", img_C)

imgHSV = cv2.cvtColor(img_C, cv2.COLOR_BGR2HSV)  # Passage BGR --> HSV

# 2.3 Conversion HSV et modification de H
# Se souvenir que la teinte est un angle entre 0 et 360° convertie sous OpenCV en une
# valeur entre 0 et 179 "qui boucle" - cf poly de cours
I_new_HSV1 = imgHSV.copy()  # Stockage image HSV
h, s, v = cv2.split(imgHSV)  # Extraction des trois plans ou  h=imgHSV[:,:,0] ...
H_new = h + 60  # h+60 une rotation des couleurs : rouge --> vert --> bleu --> rouge
H_new[H_new > 180] = H_new[H_new > 180] - 180  # Test Sioux de la mort ;-) !
# H_new = (h+60)%180
I_new_HSV1[:, :, 0] = H_new
I_new = cv2.cvtColor(I_new_HSV1, cv2.COLOR_HSV2BGR)  # conversion HSV --> BGR
# I_new = cv2.cvtColor(I_new_HSV, cv2.COLOR_HSV2RGB) # conversion HSV --> RGB puis plt...

# Affichage
cv2.namedWindow("Ma nouvelle image H", cv2.WINDOW_NORMAL)  # Pour dimensionner la fenetre d'affichage
cv2.imshow("Ma nouvelle image H", I_new)

# 2.3 Conversion HSV et modification de S
I_new_HSV2 = imgHSV.copy()  # Stockage image HSV
h, s, v = cv2.split(imgHSV)  # Extraction des trois plans ou  h=imgHSV[:,:,0] ...
S_new = np.minimum(2.0 * s, 255)  # 2*s : Les couleurs sont plus marquées
# S_new      = s/3.0                          # s/3 : Aspect délavé et surtout perte de couleur
I_new_HSV2[:, :, 1] = S_new
I_new = cv2.cvtColor(I_new_HSV2, cv2.COLOR_HSV2BGR)  # conversion HSV --> BGR

# Affichage
cv2.namedWindow("Ma nouvelle image S", cv2.WINDOW_NORMAL)  # Pour dimensionner la fenetre d'affichage
cv2.imshow("Ma nouvelle image S", I_new)

# 2.3 Conversion HSV et modification de V
I_new_HSV3 = imgHSV.copy()  # Stockage image HSV
h, s, v = cv2.split(imgHSV)  # Extraction des trois plans ou  h=imgHSV[:,:,0] ...
V_new = np.minimum(1.5 * v, 255)  # 1.5*v : l'image est plus lumineuse
I_new_HSV3[:, :, 2] = V_new
I_new = cv2.cvtColor(I_new_HSV3, cv2.COLOR_HSV2BGR)  # conversion HSV --> BGR

# Affichage
cv2.namedWindow("Ma nouvelle image V", cv2.WINDOW_NORMAL)  # Pour dimensionner la fenetre d'affichage
cv2.imshow("Ma nouvelle image V", I_new)

##### FIN
cv2.waitKey(0)
cv2.destroyAllWindows()


###############################################################################
# Lecture d'une image
fname = "Data/crayons.jpg"
img_C = cv2.imread(fname)  # Lecture image en couleurs BGR

# Affichage
cv2.namedWindow("mon image BGR", cv2.WINDOW_NORMAL)  # Pour dimensionner la fenetre d'affichage
cv2.imshow("mon image BGR", img_C)

# Espace HSV
imgHSV = cv2.cvtColor(img_C, cv2.COLOR_BGR2HSV)  # Passage BGR --> HSV
H, S, V = cv2.split(imgHSV)  # Extraction des trois plans ou  h=imgHSV[:,:,0] ...

# Init des seuils
h_min, h_max = 40, 80  # intervalle Teinte (H) --> (40,80) pour le vert ; (110,130) pour le bleu; (140,180) pour le rouge/magenta
s_min, s_max = 0, 255  # intervalle Saturation (S)
v_min, v_max = 0, 255  # intervalle Valeurs (V)


# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html
def binariseHSV():
    # Recherche de seuil ... bof bof c'est compliqué de maniere automatique ...
    # max_S = np.max(S)           # Max de S
    # s_min = (int) (0.1*(max_S)) # Seuil = 10% de max_S

    # max_V = np.max(V)           # Max de V
    # v_min = (int) (0.1*(max_V)) # Seuil = 10% de max_V

    # Definition des seuil 'a la main'
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    print("Seuil Bas HSV:", lower)
    print("Seuil Haut HSV:", upper)

    img_Binaire = cv2.inRange(imgHSV, lower, upper)  # Seuillage dans les trois plans
    img_Calque = cv2.bitwise_and(img_C, img_C, mask=img_Binaire)  # Pour faire super jolie ...

    # Affichage
    cv2.namedWindow("mon Mask", cv2.WINDOW_NORMAL)  # Pour dimensionner la fenetre d'affichage
    cv2.imshow("mon Mask", img_Binaire)

    cv2.namedWindow("mon Resultat", cv2.WINDOW_NORMAL)  # Pour dimensionner la fenetre d'affichage
    cv2.imshow("mon Resultat", img_Calque)


# Et hop petit test de la méthode
binariseHSV()
# Possible de réaliser des trackbars ... moi j'ai pas fait ici ... c'est pas bien ...

##### FIN
cv2.waitKey(0)
cv2.destroyAllWindows()

###############################################################################
img_C = cv2.imread(fname)
# Affichage
cv2.namedWindow("mon image BGR", cv2.WINDOW_NORMAL)  # Pour dimensionner la fenetre d'affichage
cv2.imshow("mon image BGR", img_C)

# Init des seuils
b_min, b_max = 0, 90  # intervalle B
g_min, g_max = 0, 90  # intervalle G
r_min, r_max = 100, 255  # intervalle R


def binariseBGR():
    # definition des seuil 'a la main'
    lower = np.array([b_min, g_min, r_min])
    upper = np.array([b_max, g_max, r_max])
    print("Seuil Bas BGR:", lower)
    print("Seuil Haut BGR:", upper)

    img_Binaire = cv2.inRange(img_C, lower, upper)  # seuillage dans les trois plans
    img_Calque = cv2.bitwise_and(img_C, img_C, mask=img_Binaire)  # Pour faire super jolie ...

    # Affichage
    cv2.namedWindow("mon Mask", cv2.WINDOW_NORMAL)  # Pour dimensionner la fenetre d'affichage
    cv2.imshow("mon Mask", img_Binaire)

    cv2.namedWindow("mon Resultat", cv2.WINDOW_NORMAL)  # Pour dimensionner la fenetre d'affichage
    cv2.imshow("mon Resultat", img_Calque)


binariseBGR()
# Possible de réaliser des trackbars ... moi j'ai pas fait ici ... c'est vraiment pas bien

##### FIN
cv2.waitKey(0)
cv2.destroyAllWindows()


###############################################################################
# Lecture d'une image & information sur l'image
fname = "Data/crayons.jpg"

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html
img_C = cv2.imread(fname)  # Lecture image en couleurs BGR

# Affichage
cv2.namedWindow("mon image BGR", cv2.WINDOW_NORMAL)  # Pour dimensionner la fenetre d'affichage
cv2.imshow("mon image BGR", img_C)

# Espace HSV
imgHSV = cv2.cvtColor(img_C, cv2.COLOR_BGR2HSV)  # Passage BGR --> HSV
H, S, V = cv2.split(imgHSV)  # Extraction des trois plans ou  h=imgHSV[:,:,0] ...

# Init des seuils
h_min, h_max = 40, 80  # intervalle Teinte (H) --> (40,80) pour le vert (110,130) pour le bleu (140,180)
s_min, s_max = 0, 255  # intervalle Saturation (S)
v_min, v_max = 0, 255  # intervalle Valeurs (V)


def binariseHSV(pos):
    # Recherche de seuil ... bof bof ... c'est compliqué de maniere automatique ...
    max_S = np.max(S)  # Max de S
    s_min = (int)(0.1 * (max_S))  # Seuil = 10% de max_S

    max_V = np.max(V)  # Max de V
    v_min = (int)(0.1 * (max_V))  # Seuil = 10% de max_V

    # Estimation des seuils par barre de défilement
    h_min = cv2.getTrackbarPos("Hmin", "mon Mask")
    h_max = cv2.getTrackbarPos("Hmax", "mon Mask")

    # Définition des seuils
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    print("Seuil Bas HSV:", lower)
    print("Seuil Haut HSV:", upper)

    img_Binaire = cv2.inRange(imgHSV, lower, upper)  # seuillage dans les trois plans
    img_Calque = cv2.bitwise_and(img_C, img_C, mask=img_Binaire)  # Pour faire super jolie ...

    # Affichage
    cv2.namedWindow("mon Mask", cv2.WINDOW_NORMAL)  # Pour dimensionner la fenetre d'affichage
    cv2.imshow("mon Mask", img_Binaire)

    cv2.namedWindow("mon Resultat", cv2.WINDOW_NORMAL)  # Pour dimensionner la fenetre d'affichage
    cv2.imshow("mon Resultat", img_Calque)


# Test
binariseHSV(0)

# Creation des barres de défilement
cv2.createTrackbar("Hmin", "mon Mask", h_min, 180, binariseHSV)
cv2.createTrackbar("Hmax", "mon Mask", h_max, 180, binariseHSV)

##### FIN
cv2.waitKey(0)
cv2.destroyAllWindows()


###############################################################################
# Init des seuils
h_min, h_max = 40, 80  # intervalle Teinte (H) --> (40,80) pour le vert (110,130) pour le bleu (140,180)
s_min, s_max = 0, 255  # intervalle Saturation (S)
v_min, v_max = 0, 255  # intervalle Valeurs (V)


def binariseHSV(pos):
    # Recherche de seuil ... bof bof ... c'est compliqué de maniere automatique ...
    max_S = np.max(S)  # Max de S
    s_min = (int)(0.2 * (max_S))  # Seuil = 10% de max_S

    max_V = np.max(V)  # Max de V
    v_min = (int)(0.2 * (max_V))  # Seuil = 10% de max_V

    # Estimation des seuils par barre de défilement
    h_min = cv2.getTrackbarPos("Hmin", "mon Mask")
    h_max = cv2.getTrackbarPos("Hmax", "mon Mask")

    # Définition des seuils
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    print("Seuil Bas HSV:", lower)
    print("Seuil Haut HSV:", upper)

    img_Binaire = cv2.inRange(imgHSV, lower, upper)  # seuillage dans les trois plans
    img_Calque = cv2.bitwise_and(img_gaus, img_gaus, mask=img_Binaire)  # Pour faire super jolie ...

    img_binaire = 255 - img_Binaire;

    # Post-traitement sur l'image binaire
    ksize = 15  # Taille element structurant
    noyaux = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))  # Element structurant carré
    img_morph = cv2.morphologyEx(img_binaire, cv2.MORPH_OPEN, noyaux)  # ouverture : erosion -> dilatation

    # Affichage
    cv2.imshow("mon Mask", img_Binaire)
    cv2.imshow("mon Resultat", img_Calque)
    cv2.imshow("Frame Binaire", img_morph)


if __name__ == "__main__":
    print('Test image OpenCV')

    # Pour la capture de video (ici en usb)
    cap = cv2.VideoCapture(0)  # numero de la camera (0 ou 1 ... à tester )

    # Init (un peu pas bien réalisé)
    ret, frame = cap.read()
    img_gaus = frame
    imgHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Passage BGR --> HSV
    H, S, V = cv2.split(imgHSV)

    # Parametre de la camera (de 0 à 18)
    print("Width:  ", cap.get(3))
    print("Height: ", cap.get(4))
    print("Frame/sec: ", cap.get(5))

    # Creation des figures pour affichage des frames
    cv2.namedWindow("Frame Couleur")
    cv2.namedWindow("Frame Binaire")
    cv2.namedWindow("mon Mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("mon Resultat", cv2.WINDOW_NORMAL)

    binariseHSV(0)
    cv2.createTrackbar("Hmin", "mon Mask", h_min, 180, binariseHSV)
    cv2.createTrackbar("Hmax", "mon Mask", h_max, 180, binariseHSV)

    # Ouverture du flux vidéo
    while (True):
        # Capture image par image
        ret, frame = cap.read()  # retourne un bool (True,false)

        cv2.imshow("Frame Couleur", frame)  # Affichage couleur

        # Filtrage Gaussien
        taille = 15
        img_gaus = cv2.GaussianBlur(frame, (taille, taille), 0)

        # Segmentation image couleurs
        imgHSV = cv2.cvtColor(img_gaus, cv2.COLOR_BGR2HSV)  # Passage BGR --> HSV
        H, S, V = cv2.split(imgHSV)
        binariseHSV(0)

        # Si touche alors on fait des choses !
        if cv2.waitKey(1) == ord('q'):  # old : cv2.waitKey(1)  & 0xFF
            # arret de l'acquisition du flux
            break

    # Fermeture des graphiques et de la capture video
    cap.release()
    cv2.destroyAllWindows()
