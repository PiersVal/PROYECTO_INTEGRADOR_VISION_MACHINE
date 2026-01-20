import os
import cv2
import numpy as np
from sklearn.cluster import KMeans

# ---------------- CONFIGURACIÓN ----------------
MAX_IMGS = 100
K = 2  # fondo / mano

DATASETS = [
    {"input": "paper", "output": "paper_binarizada", "prefix": "paper"},
    {"input": "scissors", "output": "scissors_binarizada", "prefix": "scissors"},
    {"input": "rock", "output": "rock_binarizada", "prefix": "rock"}
]

# ====================================================
#                 FUNCIÓN OTSU (MANUAL)
# ====================================================
def OTSU(img):
    nk = np.zeros(256)

    for i in range(img.shape[0]):
    
        for j in range(img.shape[1]):
            nk[int(img[i, j])] += 1

    maxima_varianza = 0.0
    mejor_umbral = 0
    N = img.size

    for umbral in range(256):
        f1 = nk[:umbral].sum()
        f2 = nk[umbral:].sum()

        if f1 == 0 or f2 == 0:
            continue

        w1 = f1 / N
        w2 = f2 / N

        u1 = np.sum([i * nk[i] for i in range(0, umbral)]) / f1
        u2 = np.sum([i * nk[i] for i in range(umbral, 256)]) / f2

        ut = u1 * w1 + u2 * w2
        varianza = w1 * (u1 - ut)**2 + w2 * (u2 - ut)**2

        if varianza > maxima_varianza:
            maxima_varianza = varianza
            mejor_umbral = umbral

    return mejor_umbral

# ====================================================
#           BINARIZACIÓN GRIS → (0 y 1)
# ====================================================
def miGray2binaria(img, theta):
    filas, columnas = img.shape
    X = np.zeros((filas, columnas), dtype=np.uint8)

    for i in range(filas):
        for j in range(columnas):
            X[i, j] = 1 if img[i, j] > theta else 0

    return X

# ====================================================
#         K-MEANS + OTSU INTEGRADO
# ====================================================
def segmentar_kmeans_otsu(img, k=2):

    h, w = img.shape[:2]

    # --- K-MEANS EN HSV ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ruido = cv2.GaussianBlur(hsv, (7, 7), 0)
    
    X = ruido.reshape((-1, 3)).astype(np.float32)

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)

    labels_img = labels.reshape((h, w))

    # Cluster más grande = mano
    areas = [(labels_img == i).sum() for i in range(k)]
    hand_cluster = np.argmax(areas)

    mask_kmeans = (labels_img == hand_cluster).astype(np.uint8) * 255

    # Limpieza inicial
    kernel = np.ones((7, 7), np.uint8)
    mask_kmeans = cv2.morphologyEx(mask_kmeans, cv2.MORPH_CLOSE, kernel)

    # --- APLICAR MÁSCARA A LA IMAGEN ---
    img_masked = cv2.bitwise_and(img, img, mask=mask_kmeans)

    # --- OTSU MANUAL ---
    gray = cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)
    

    umbral = OTSU(gray)
    binaria_01 = miGray2binaria(gray, umbral)
    binaria_255 = (binaria_01 * 255).astype(np.uint8)

    # Limpieza final
    binaria_255 = cv2.morphologyEx(binaria_255, cv2.MORPH_OPEN, kernel)
    binaria_255 = cv2.bitwise_not(binaria_255)

    return binaria_255


