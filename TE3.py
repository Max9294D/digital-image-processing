import numpy as np
import cv2
from matplotlib import pyplot as plt


def cb_seuillageP1(x):
    ret, binary = cv2.threshold(imgNG, cv2.getTrackbarPos("Seuil T", "Binary"), 255, cv2.THRESH_BINARY_INV);
    # les oies étant plus foncées que le fond, utiliser CV_THRESH_BINARY_INV
    # permet de les avoir en blanc dans l'image binaire
    # Cette solution est préférable si on souhaite identifier les composantes
    # connexes associées aux oiseaux
    cv2.imshow("Binary", binary);


# %% Partie 2 - Binarisation
# Question 1
seuil = 100;
img = cv2.imread('Data/voilier_oies_blanches.jpg');
imgNG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
# pour info la valeur de cv2.COLOR_BGR2GRAY=6!
hist = cv2.calcHist([imgNG], [0], None, [256], [0, 255])  # voir aide en ligne
ret, binary = cv2.threshold(imgNG, seuil, 255, cv2.THRESH_BINARY_INV);
# explication du parametre threshold_type : https://docs.opencv.org/3.2.0/d7/
# d4d/tutorial_py_thresholding.html
res = np.hstack((imgNG, binary))
cv2.namedWindow("Image Originale + Image binarisee", 1)
cv2.imshow("Image Originale + Image binarisee", res);

# Visualisation de l'histogramme : cf TE1
plt.figure()
plt.plot(hist);
plt.xlabel('niveau de gris i ');
plt.ylabel('h(i)');
plt.title('histogramme de l'' image');
plt.show()
cv2.waitKey(0)

plt.close('all')
cv2.destroyAllWindows()
# Segmenter cette image pourrait avoir comme but premier d'isoler les oiseaux
# pour : les dénombrer,  en mesurer des caractéristiques (suivi de la faune
# sauvage par exemple)?
# Dans la mesure où les oiseaux sont plutôt noirs sur un fond plutôt blancs,
# une approche par seuillage de l'histogramme est approprié (cf cours)


# Question 2
seuil = 128;
img = cv2.imread('Data/voilier_oies_blanches.jpg');
cv2.imshow('Image initiale', img);
imgNG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
cv2.namedWindow("Binary", 1)
cv2.createTrackbar("Seuil T", "Binary", seuil, 255, cb_seuillageP1);
cv2.waitKey(0)
cv2.destroyAllWindows()

# Question 3b) et 3c)
seuil = 128;
img = cv2.imread('Data/img_ds.jpg');
cv2.imshow('Image initiale', img);
imgNG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
imgNG = cv2.GaussianBlur(imgNG, (7, 7), 0)  # un exemple de pretraitement dont le
# but est de réduire le bruit.
cv2.namedWindow("Binary", 0)
cv2.createTrackbar("Seuil T", "Binary", seuil, 255, cb_seuillageP1);
cv2.waitKey(0)
cv2.destroyAllWindows()
# Commentaire : cette image est bruitée - sans préfiltrage, il subsiste des pixels erronés
# après la phase de segmentation, tant au niveau des objets que sur le fond

# Question 4a)
img = cv2.imread('Data/voilier_oies_blanches.jpg');
cv2.imshow('Image initiale', img);
imgNG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
_, binary = cv2.threshold(imgNG, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.namedWindow("BinaryOtsu", 1)
cv2.imshow("BinaryOtsu", binary);
cv2.waitKey(0)
cv2.destroyAllWindows()

# %% Partie 3 : acquisition, visualisation et enregistrement d'un flux vidéo
# Visualisation d'un flux vidéo
cap = cv2.VideoCapture(0)
# 0 : on récupère le flux de la 1ère caméra accessible

# Pour la question suivante : affichage de quelques paramètres du flux vidéo
print('largeur de l''image : ', cap.get(3))
print('hauteur de l''image : ', cap.get(4))
print('nombre d''images par seconde : ', cap.get(5))
print('FOURCC : ', cap.get(6))
# valeurs définies dans C:\APPLIS\opencv\build\include\opencv2\videoio\videoio_c.h!

while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        # Display the resulting frame
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1);
        if key == ord('q'):
            break
    else:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

# Enregistrement d'un flux vidéo
cap = cv2.VideoCapture(0)
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID');  # see https://www.fourcc.org/ for details
# FOURCC is short for "four character code" - an identifier for a video codec, compression format,
# color or pixel format used in media files.
# fourcc = cv2.VideoWriter_fourcc(*'MJPG');#fonctionne aussi
# fourcc = cv2.VideoWriter_fourcc(*'X264');#fonctionne aussi
out = cv2.VideoWriter('output_TE3.avi', fourcc, 20.0, (640, 480))

while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        out.write(frame)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF;
        if key == ord('q'):
            break
    else:
        break

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()

# Visualisation d'un flux vidéo
cap = cv2.VideoCapture(0)
i = 1;
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        # Display the resulting frame
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF;
        if key == ord('q'):
            break
        elif key == ord('s'):
            nom = 'fram' + str(i) + '.png'
            cv2.imwrite(nom, frame)
            i = i + 1;
    else:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
# plt.close('all')

# %% Partie 4
# Question 4b)
cap = cv2.VideoCapture(0)
# 0 : on récupère le flux de la 1ère caméra accessible
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID');  # see https://www.fourcc.org/ for details
# FOURCC is short for "four character code" - an identifier for a video codec, compression format,
# color or pixel format used in media files.
# fourcc = cv2.VideoWriter_fourcc(*'MJPG');#fonctionne aussi
# fourcc = cv2.VideoWriter_fourcc(*'X264');#fonctionne aussi
# out1 =cv2.VideoWriter('output_TE3_P4_flux_orig.avi',fourcc,20.0,(640,480))
# out2 =cv2.VideoWriter('output_TE3_P4_flux_segm.avi',fourcc,20.0,(640,480))

while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        # Display the resulting frame
        imgNG = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);
        _, binary = cv2.threshold(imgNG, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # seuil totatelement arbitraire
        res = np.hstack((imgNG, binary))
        cv2.namedWindow("flux original + flux binarise", 1)
        cv2.imshow("flux original + flux binarise", res);
        #        out1.write(imgNG)#enregistrement ne fonctionne pas !
        #        out2.write(binary)
        key = cv2.waitKey(1);
        if key == ord('q'):
            break
    else:
        break

# When everything done, release the capture
cap.release()
# out1.release()
# out2.release()
cv2.destroyAllWindows()
plt.close('all')



