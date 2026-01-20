
import os
import random
import cv2
import kagglehub
import shutil
from src.features.procesamiento_rps  import segmentar_kmeans_otsu

def generar_datos_rps():
    SEED = 42
    random.seed(SEED)

    NUM_MUESTRAS = 100
    RUTA_SALIDA = "dataset/RPS_Procesado"
    EXT_VALIDAS = (".png", ".jpg", ".jpeg")

    TRADUCCION = {
        'rock': 'piedra',
        'paper': 'papel',
        'scissors': 'tijeras'
    }

    path_origen = kagglehub.dataset_download("drgfreeman/rockpaperscissors")

    for root, dirs, _ in os.walk(path_origen):
        clases = [d for d in dirs if d.lower() in TRADUCCION]
        if clases:
            ruta_base = root
            break

    if os.path.exists(RUTA_SALIDA):
        shutil.rmtree(RUTA_SALIDA)

    for clase in clases:
        nombre_es = TRADUCCION[clase.lower()]
        entrada = os.path.join(ruta_base, clase)
        salida = os.path.join(RUTA_SALIDA, nombre_es)
        os.makedirs(salida, exist_ok=True)

        imgs = [f for f in os.listdir(entrada) if f.lower().endswith(EXT_VALIDAS)]
        imgs = random.sample(imgs, min(NUM_MUESTRAS, len(imgs)))

        for img_name in imgs:
            img = cv2.imread(os.path.join(entrada, img_name))
            if img is None:
                continue

            mascara = segmentar_kmeans_otsu(img)
            cv2.imwrite(os.path.join(salida, img_name), mascara)
