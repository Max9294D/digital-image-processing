import numpy as np
import cv2

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # pour parties 2 et 3


def equalize_and_display_histogram(img):
    """
    Arguments: img array
    Return value: None
    """

    # si l'image est en niveau de gris: ndim=2, en couleur: ndim=3

    if img.ndim == 2:
        # plt.hist(img.ravel(),256,[0,256]); plt.show() #fast calculation and visualisation with Matplotlib
        # numpy.ravel() : fonction qui permet ici de transformer une matrice en vecteur ligne
        # ou calcul de l'histogramme avec OpenCV + visu avec Matplotlib
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])  # voir aide en ligne pour fixer les paramètres
        plt.figure()
        plt.subplot(121), plt.plot(hist);
        plt.xlabel('niveau de gris i ');
        plt.ylabel('h(i)');
        plt.title('histogramme de l'' image');
        plt.show()
    #        #ou Histogram Calculation with numpy
    #        hist,bins = np.histogram(img.ravel(),256,[0,256])
    #        #ou plus rapide : hist = np.bincount(img.ravel(),minlength=256)
    else:
        # si l'image est en couleur : pas demandé dans le TE2
        color = ('B', 'G', 'R')
        for i, col in enumerate(color):
            histr = cv2.calcHist([img], [i], None, [256], [0, 256])
            l = "canal" + " " + color[i]
            plt.plot(histr, color=col, label=l)
            plt.xlim([0, 256])
        plt.title('histogramme de l''image couleur');
        plt.legend()
        plt.show()
    #
    #    #visualisation with OpenCV : voir code python hist.py sous ...\opencv\sources\samples\python2
    #
    # egalisation d'histogramme avec openCV
    if img.ndim == 2:
        equ = cv2.equalizeHist(img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equ = cv2.equalizeHist(img)
    res = np.hstack((img, equ))  # stacking images side-by-side
    #    cv2.namedWindow('display_win', cv2.WINDOW_NORMAL)
    cv2.imshow('display_win', res)
    print("Minimum pixel values : ", np.amin(equ, axis=(0, 1)))
    print("Maximum pixel values : ", np.amax(equ, axis=(0, 1)))
    print("Average pixel values : ", np.mean(equ, axis=(0, 1)))
    print("Standard deviations on pixel value : ", np.std(equ, axis=(0, 1)))
    cv2.waitKey(0)

    # visualisation des deux histogrammes avant et après
    if img.ndim == 2:
        hist_equ = cv2.calcHist([equ], [0], None, [256], [0, 256])  # voir aide en ligne pour fixer les paramètres
        plt.subplot(122), plt.plot(hist_equ);
        plt.xlabel('niveau de gris i ');
        plt.ylabel('h(i)');
        plt.title('histogramme de l'' image égalisée');
        plt.show()
    else:
        # si l'image est en couleur : pas demandé dans le TE1
        color = ('B', 'G', 'R')
        for i, col in enumerate(color):
            histr_equ = cv2.calcHist([equ], [i], None, [256], [0, 256])
            l = "canal" + " " + color[i]
            plt.plot(histr_equ, color=col, label=l)
            plt.xlim([0, 256])
        plt.title('histogramme de l''image couleur égalisée');
        plt.legend()
        plt.show()

    #    algo CLAHE : préférable s'il y a des zones très claires et d'autres très peu contrastées dans la même image
    #     create a CLAHE object (Arguments are optional).
    #   L'amélioration sur les images données n'est pas criante ...

    if img.ndim == 2:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl1 = clahe.apply(img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl1 = clahe.apply(img)
    res = np.hstack((img, cl1))
    cv2.imshow('Image equ CLAHE', res)
    print("Minimum pixel values : ", np.amin(cl1, axis=(0, 1)))
    print("Maximum pixel values : ", np.amax(cl1, axis=(0, 1)))
    print("Average pixel values : ", np.mean(cl1, axis=(0, 1)))
    print("Standard deviations on pixel value : ", np.std(cl1, axis=(0, 1)))
    cv2.waitKey(0)


# %% Partie 1

# fname = "XXX/paysage_sombre.png"
# fname = "XXX/voilier_oies_blanches.jpg"
fname = "Data/pout.jpg"

# img=cv2.imread(fname,0)#0 signifie qu'on lit l'image et qu'on la convertit en niveaux de gris
# autre solution :
imgC = cv2.imread(fname);
img = cv2.cvtColor(imgC, cv2.COLOR_BGR2GRAY);
cv2.imshow("Image originale", img)
# plt.figure(),plt.imshow(img,cmap='gray'),plt.title(display_win_name)
# plt.xticks([]), plt.yticks([])
# plt.show()
# En condensé ! Valable qq soit la nature de l'image (NG ou Couleur)
print("Minimum pixel values : ", np.amin(img, axis=(0, 1)))
print("Maximum pixel values : ", np.amax(img, axis=(0, 1)))
print("Average pixel values : ", np.mean(img, axis=(0, 1)))
print("Standard deviations on pixel value : ", np.std(img, axis=(0, 1)))

equalize_and_display_histogram(img)
cv2.waitKey(0)

cv2.destroyAllWindows();
plt.close('all')

# %% Partie 2 : amélioration d'images : réduction de bruit
# https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html

img = cv2.imread('XXX/Image_epave.jpg', 0);  # indiquer le chemin complet
cv2.imshow('Image initiale', img);
# cv2.waitKey(0)

# filtrage moyenneur
img_filtree1 = cv2.blur(img, (5, 5));
# Pour traiter les bords, il y a plusieurs possibilités : paramètre bordertype
# qui peut prendre plusieurs valeurs (BORDER_REPLICATE, BORDER_REFLECT,
# BORDER_REFLECT_101, BORDER_WRAP, BORDER_CONSTANT -
# voir https://docs.opencv.org/3.0-last-rst/modules/imgproc/doc/filtering.html
# haut de la page)
# cv2.imshow('Image filtree',img_filtree1)
# cv2.waitKey(0)

# autre type d'affichage : affichage côte à côte
plt.subplot(211), plt.imshow(img, cmap='gray'), plt.title('Image originale')
plt.subplot(212), plt.imshow(img_filtree1, cmap='gray'), plt.title('Im. Filtree Moyenneur')
plt.show()
# remarque : les images sont plus petites

# autre type d'affichage :
res = np.vstack((img, img_filtree1))
cv2.imshow('resultat combine : Originale (haut) - Filtree (Moyenneur)(bas)', res)
cv2.waitKey(0)

# Commentaire filtrage moyenneur : plus la taille du filtre est grande,
# plus l'image filtree devient floue -
# Explication en considérant le domaine spatial : en moyennant sur une plus
# grande zone, on lisse davantage les variations de niveaux de gris, y compris
# au niveau des contours, qui paraissent donc moins nets.
# Explication dans le domaine fréquentiel (cf Q3) : les fréquences de coupure
# du filtre sont inversement proportionnelles à la taille du filtre, donc
# plus la taille augmente, plus les fréq de coupure diminuent et plus le filtre
# devient sélectif, c'est-à-dire qu'il laisse passer de moins en moins de
# "hautes fréquences" qui représentent généralement le bruit et les contours

# pointer également les imperfections sur les bords

# filtrage gaussien
img_filtree2 = cv2.GaussianBlur(img, (5, 5), 0);
# cv2.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]]) -> dst
# ksize : positive et impaire ou nulle (sera calculée à partir de sigmaX et sigmaY)
# sigmaX et sigmaY : valeur non nulle pour sigmaX, si sigmaY=0 alors sigmaY=sigmaX,
# si les deux sont nulles, alors calcul des valeurs à partir de ksize
# lien entre les deux : sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
# tester avec différents traitements des bords

# cv2.imshow('Image filtree',img_filtree2)
# cv2.waitKey(0)

res = np.vstack((img, img_filtree2))
cv2.imshow('resultat combine2  : Originale (haut) - Filtree (Gaussien)(bas)', res)
cv2.waitKey(0)

# Commentaire : même principe : plus la taille du filtre augmente, plus l'image
# filtrée devient floue. La taille du filtre est ajustée en fonction de sigma,
# donc plus sigma augmente, plus la taille augmente (il faut se rappeler que
# 99,7% des valeurs de la gaussienne sont entre m-3*sigma,m+3*sigma, resp. 95,5%
# entre m-2*sigma,m+2*sigma) et comme pour le moyenneur, l'effet est une image
# plus floue. Moins floue cependant car le filtre pondère les voisins selon
# une gaussienne donc plus de poids est donné au pixel central
# Explication dans le domaine fréquentiel : la fréquence de coupure radiale
# est inversement proportionnelle à l'écart-type - cf cours

# filtrage médian
img_filtree3 = cv2.medianBlur(img, 5);
# ksize doit être impaire et >1
# cv2.imshow('Image filtree',img_filtree3)
# cv2.waitKey(0)

res = np.vstack((img, img_filtree3))
cv2.imshow('resultat combine3 : Originale (haut) - Filtree (Médian)(bas)', res)
cv2.waitKey(0)

# comparaison filtre moyenneur et filtre médian
img_filtree1 = cv2.blur(img, (7, 7));
img_filtree3 = cv2.medianBlur(img, 7);
res = np.vstack((img_filtree1, img_filtree3))
cv2.imshow('resultats moyenneur-median', res)
cv2.waitKey(0)

cv2.destroyAllWindows()
plt.close('all')

# remarque : on constate que le filtre median permet une meilleure visualisation
# au sens où les contours sont mieux préservés
# Vous devez savoir théoriquement comment on applique un filtre médian
# et un filtre linéaire (moyenneur, gaussien, ...) à une image - S'en assurer !

# %% Partie 3 Transformée de fourier et filtrage

h = 1 / 9 * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32);
TFh = np.fft.fft2(h, (256, 256));
magnitude_spectrum = np.abs(np.fft.fftshift(TFh))
plt.figure()
plt.subplot(121)
plt.imshow(h, cmap='gray');
plt.colorbar()
plt.title('h'), plt.xlabel('x'), plt.ylabel('y');
plt.subplot(122);
plt.imshow(np.log2(1 + magnitude_spectrum), extent=(-0.5, 0.5, 0.5, -0.5), cmap='jet')
plt.title('Spectre d\'amplitude shifté'), plt.colorbar()
plt.xlabel('u'), plt.ylabel('v');
plt.show()

# Pour une meilleure interprétation du spectre : une autre visualisation, en 3D
fig = plt.figure()
ax = Axes3D(fig)
# Make data.
X = np.arange(-0.5, 0.5, 1 / 256)
Y = np.arange(-0.5, 0.5, 1 / 256)
X, Y = np.meshgrid(X, Y)

# Plot the surface.
surf = ax.plot_surface(X, Y, np.log2(1 + magnitude_spectrum), linewidth=0, antialiased=False, cmap='jet')

# Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

# commentaire : le spectre d'amplitude correspond à un gabarit de filtre
# passe-bas car on peut considérer que la réponse fréquentielle s'annule pour des
# fréquences f> fc où fc désigne la fréquence radiale de coupure
# Plus précisément la réponse fréquentielle du filtre moyenneur est un sinus
# cardinal 2D (cf planche de cours). Vous avez vu en 1D que la TF d'un signal
# porte est un sinus cardinal, c'est la même chose ici mais en 2D!

