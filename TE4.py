import numpy as np
import matplotlib.pyplot as plt
import cv2

# img = cv2.imread('Data/voilier_oies_blanches.jpg') #en couleurs
img = cv2.imread('Data/img_ds.jpg') #en couleurs

# img = cv2.imread('Data/code_postal.png')
# img = cv2.imread('Data/le-robot-nao.jpg')
img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY ) #transforme une image couleur en image niveax de gris
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convertit une image bgr en rgb pour qu'elle soit bien affichéeé dans matplotlib

ret,thresh1 = cv2.threshold(img_gris,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))


img_erodee = cv2.erode(thresh1, kernel)
img_dilate = cv2.dilate(thresh1, kernel)


plt.figure(1)
plt.subplot(131)
plt.imshow(thresh1, cmap='gray')
plt.title("image de départ")
plt.subplot(132)
plt.imshow(img_erodee, cmap='gray')
plt.title("image érodée")
plt.subplot(133)
plt.imshow(img_dilate, cmap='gray')
plt.title("image dilatée")
plt.xlabel("")
plt.ylabel("")
plt.show()

# oouverture
img_erodee2 = cv2.erode(thresh1, kernel)
img_dilate2 = cv2.dilate(img_erodee2, kernel)

opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)


#fermeture
img_dilate3 = cv2.dilate(thresh1, kernel)
img_erodee3 = cv2.erode(img_dilate3, kernel)

closing = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)

plt.figure(2)
plt.subplot(131)
plt.imshow(thresh1, cmap='gray')
plt.title("image de départ")
plt.subplot(132)
plt.imshow(img_dilate2, cmap='gray')
# plt.imshow(opening, cmap='gray')
plt.title("Ouverture")
plt.subplot(133)
plt.imshow(img_erodee3, cmap='gray')
# plt.imshow(closing, cmap='gray')
plt.title("Fermeture")
plt.xlabel("")
plt.ylabel("")
plt.show()

# opening closing
img_erodee4 = cv2.erode(thresh1, kernel)
img_dilate4 = cv2.dilate(img_erodee4, kernel)
img_dilate5 = cv2.dilate(img_dilate4, kernel)
img_erodee5 = cv2.erode(img_dilate5, kernel)

opening_closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)


plt.figure(3)
plt.subplot(121)
plt.imshow(thresh1, cmap='gray')
plt.title("image de départ")
plt.subplot(122)
plt.imshow(opening_closing, cmap='gray')
plt.title("Opening - closing")
plt.xlabel("")
plt.ylabel("")
plt.show()

nb_ccomp, comp_conn2 = cv2.connectedComponents(thresh1,connectivity=8) #par défaut 8-connexité
print("nombre d'oies trouvées :",nb_ccomp-1)


plt.figure(4)
plt.subplot(121)
plt.imshow(thresh1, cmap='gray')
plt.title("image de départ")
plt.subplot(122)
plt.imshow(comp_conn2, cmap='gray')
plt.title("Etiquetage")
plt.xlabel("")
plt.ylabel("")
plt.show()

contours, _ = cv2.findContours(thresh1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
img_contours = img.copy()
#extérieur des contours
cv2.drawContours(img_contours,contours,-1,(0,255,255),2)
#rajout de l'affichage de l'intérieur des composants connexes en cyan
# cv2.drawContours(img_contours,contours,-1,(0,255,255),-1)


plt.figure(5)
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title("image de départ")
plt.subplot(122)
plt.imshow(img_contours, cmap='gray')
plt.title("Contours")
plt.xlabel("")
plt.ylabel("")
plt.show()

cap = cv2.VideoCapture()
cv2.namedWindow("Original frame", None)
cv2.namedWindow("Bin frame", None)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

while(cap.isOpened()):
    # Capture frame-by-frame
    ret1, frame = cap.read()
    if ret1:
        # Display the resulting frame
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # ouverture : erosion + dilatation
        img_open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        _, comp_conn2 = cv2.connectedComponents(img_open)  # par défaut : 8-connexité
        contours, _ = cv2.findContours(comp_conn2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        img_contours = frame.copy()
        # exterieur des contours
        cv2.drawContours(img_contours, contours, -1, (0, 255, 255), 2)
        # rajout de l'affichage de l'intérieur des composantes connexes en cyan
        # cv2.drawContours(img_contours, contours, -1, (255, 255, 0), -1)
        cv2.imshow('Original frame', frame)
        cv2.imshow('Bin frame', img_contours)
        key = cv2.waitKey(1)
        if key == ord('q'):
             break
    else:
        break

cap.release()
cv2.destroyAllWindows()
