import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('Data/bateau2.JPG',0)

kernel_H = np.array([(-1,-2,-1),(0,0,0),(1,2,1)])
kernel_V =np.array([(-1,0,1),(-2,0,2),(-1,0,1)])

sobelx = cv2.filter2D(img,cv2.CV_64F,kernel_H)
# sobelx = cv2.sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.filter2D(img,cv2.CV_64F,kernel_V)
# sobelx = cv2.sobel(img,cv2.CV_64F,0,1,ksize=5)

norme_sobel = np.sqrt(np.square(sobelx)+np.square(sobely))
norme_sobel_uint8 = np.uint8(norme_sobel)

seuil = np.round(0.33*np.max(np.abs(norme_sobel)))
# seuil = 100
ret,img_binaire = cv2.threshold(norme_sobel_uint8,seuil,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

img_canny = cv2.Canny(img,100,150)

plt.figure()

plt.subplot(2,3,1)
plt.imshow(img,cmap = 'gray')
plt.title('Original')
plt.xticks([]), plt.yticks([])

plt.subplot(2,3,2)
plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel Horizontale')
plt.xticks([]), plt.yticks([])

plt.subplot(2,3,3)
plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Verticale')
plt.xticks([]), plt.yticks([])

plt.subplot(2,3,4)
plt.imshow(norme_sobel,cmap = 'gray')
plt.title('Module de Sobel')
plt.xticks([]), plt.yticks([])

plt.subplot(2,3,5)
plt.imshow(img_binaire,cmap = 'gray')
plt.title('Carte des contours binaires')
plt.xticks([]), plt.yticks([])

plt.subplot(2,3,6)
plt.imshow(~img_canny,cmap = 'gray')
plt.title('Canny')
plt.xticks([]), plt.yticks([])

plt.show()


###############################################################################
# 1. Lecture d'une image & information sur l'image
#fname  = "voilier_oies_blanches.jpg"
fname  = "Data/img_ds.jpg"
#fname  = "le-robot-nao.jpg"
#fname  = "bateau2.jpg"
img    = cv2.imread(fname, 0) # Lecture image en niveau de gris (conversion si couleurs)
img_C  = cv2.imread(fname)    # Lecture image en couleurs BGR

# Affichage image
plt.figure()
plt.subplot(221)
plt.imshow(img, cmap = 'gray')
plt.xticks([]), plt.yticks([])
plt.title(" Mon image")

# Filtrage : Réduction du bruit (filtre gaussien)
taille    = 15
img_gaus  = cv2.GaussianBlur(img, (taille,taille), 0)

plt.subplot(222)
plt.imshow(img_gaus, cmap = 'gray')
plt.xticks([]), plt.yticks([])
plt.title(" Filtrage Gauss")

# Detection des contours (canny)
img_canny = 255-cv2.Canny(img_gaus, cv2.CV_64F, 90, 120);

plt.subplot(223)
plt.imshow(img_canny, cmap = 'gray')
plt.xticks([]), plt.yticks([])
plt.title(" Contours ")

# Post Traitement
ksize     = 5 # Taille element structurant
noyaux    = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize)) # Element structurant carré
img_morph = cv2.morphologyEx(img_canny, cv2.MORPH_OPEN, noyaux)             # ouverture : erosion -> dilatation

plt.subplot(224)
plt.imshow(img_morph, cmap = 'gray')
plt.xticks([]), plt.yticks([])
plt.title(" Contours ")
plt.show()

# Etiquettage
num_features, comp_conn1 = cv2.connectedComponents(255-img_morph) #par défaut : 8-connexité
print("Nbre formes: ", num_features-1)

plt.figure()
plt.subplot(221)
plt.imshow(comp_conn1, cmap = 'jet')
plt.xticks([]), plt.yticks([])
plt.title(" Composantes ")

###############################################################################
# Une autre approche pour extraire les 3 formes
# Binarisation et extraction
taille    = 15
img_gaus  = cv2.GaussianBlur(img, (taille,taille), 0)

ret_otsu, img_otsu  = cv2.threshold(img_gaus, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV) # Binarisation image init et inversion

ksize     = 9 # Taille element structurant
noyaux    = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize)) # Element structurant carré
img_morph = cv2.morphologyEx(img_otsu, cv2.MORPH_OPEN, noyaux)

num_features, comp_conn1 = cv2.connectedComponents(img_morph) #par défaut : 8-connexité
print("Nbre formes: ", num_features-1)

plt.subplot(222)
plt.imshow(img_otsu, cmap = 'gray')
plt.xticks([]), plt.yticks([])
plt.title(" Filtrage & Binaire ")

plt.subplot(223)
plt.imshow(img_morph, cmap = 'gray')
plt.xticks([]), plt.yticks([])
plt.title(" Post-traitement ")

plt.subplot(224)
plt.imshow(comp_conn1, cmap = 'jet')
plt.xticks([]), plt.yticks([])
plt.title(" Composantes ")