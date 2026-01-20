import os
import random
import cv2
import numpy as np
import kagglehub
from ultralytics import YOLO
from tqdm import tqdm
from src.features.procesamiento_animales import pipeline_clasico

def generar_datos_animales():

    SEED = 42
    random.seed(SEED)

    NUM_MUESTRAS = 100
    RUTA_SALIDA = "dataset/animals_preprocesado"
    EXT_VALIDAS = (".png", ".jpg", ".jpeg", ".bmp")

    CLASES = {
        "gatto": "cat",
        "cavallo": "horse",
        "elefante": "elephant"
    }

    print("Descargando Animals-10...")
    ruta_base = kagglehub.dataset_download("alessiocorrado99/animals10")
    ruta_raw = os.path.join(ruta_base, "raw-img")

    model = YOLO("src/models/yolov8n-seg.pt")

    if os.path.exists(RUTA_SALIDA):
        import shutil
        shutil.rmtree(RUTA_SALIDA)

    for clase_it, clase_en in CLASES.items():

        entrada = os.path.join(ruta_raw, clase_it)
        salida = os.path.join(RUTA_SALIDA, clase_it)
        os.makedirs(salida, exist_ok=True)

        imgs = [f for f in os.listdir(entrada) if f.lower().endswith(EXT_VALIDAS)]
        imgs = random.sample(imgs, min(NUM_MUESTRAS, len(imgs)))

        print(f"\nProcesando {clase_it.upper()}")

        count = 0
        for img_name in imgs:

            img_path = os.path.join(entrada, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w = img.shape[:2]

            # -------- YOLO SEGMENTACIÓN --------
            res = model(img_path)[0]
            if res.masks is None:
                continue

            masks = res.masks.data.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            names = res.names

            class_id = [k for k, v in names.items() if v == clase_en][0]
            idxs = np.where(cls == class_id)[0]
            if len(idxs) == 0:
                continue

            best = idxs[np.argmax([masks[i].sum() for i in idxs])]
            mask01 = (masks[best] >= 0.7).astype(np.uint8)
            mask01 = cv2.resize(mask01, (w, h), cv2.INTER_NEAREST)

            kernel = np.ones((5, 5), np.uint8)
            mask01 = cv2.morphologyEx(mask01, cv2.MORPH_CLOSE, kernel)
            mask01 = cv2.morphologyEx(mask01, cv2.MORPH_OPEN, kernel)

            img_fg = cv2.bitwise_and(img, img, mask=mask01)

            # -------- PIPELINE FINAL --------
            img_final = pipeline_clasico(img_fg)

            out_name = f"{clase_it}_{count:03d}.png"
            cv2.imwrite(os.path.join(salida, out_name), img_final)

            count += 1

        print(f"{count} imágenes generadas")

    print("\nDataset de animales generado correctamente")