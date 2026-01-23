import os
import cv2
import numpy as np
from ultralytics import YOLO
from genera_dataset.dataset_animales import descargar_dataset

# ----------------------------------------------------
# Descargar dataset
# ----------------------------------------------------
descargar_dataset()

# ----------------------------------------------------
# Configuración general
# ----------------------------------------------------
MODEL_NAME = "yolov8n-seg.pt"
MAX_IMGS = 100
BASE_DATASET = "dataset/animales"
TARGET_SIZE = (256, 256)

DATASETS = {
    "gatto": {
        "class_name": "cat",
        "prefix": "gato"
    },
    "cavallo": {
        "class_name": "horse",
        "prefix": "caballo"
    },
    "elefante": {
        "class_name": "elephant",
        "prefix": "elefante"
    }
}

# ----------------------------------------------------
# Modelo YOLO
# ----------------------------------------------------
model = YOLO(MODEL_NAME)

# ----------------------------------------------------
# PREPROCESAMIENTO CLÁSICO
# ----------------------------------------------------
def rescalar(img):
    return cv2.resize(img, TARGET_SIZE)

def escala_grises(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def stretching(img):
    img = img / 255.0
    a, b = img.min(), img.max()
    if b - a == 0:
        return (img * 255).astype(np.uint8)
    return np.clip((img - a) / (b - a) * 255, 0, 255).astype(np.uint8)

def ecualizar(img):
    return cv2.equalizeHist(img)

def otsu(img):
    _, binaria = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binaria

# ----------------------------------------------------
# SEGMENTACIÓN YOLO
# ----------------------------------------------------
def segmentar_yolo(img, class_name, umbral=0.7):
    h, w = img.shape[:2]

    res = model(img)[0]
    if res.masks is None:
        return None

    masks = res.masks.data.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)
    names = res.names

    class_id = [k for k, v in names.items() if v == class_name][0]
    idxs = np.where(cls == class_id)[0]

    if len(idxs) == 0:
        return None

    best = idxs[np.argmax([masks[i].sum() for i in idxs])]
    mask = (masks[best] >= umbral).astype(np.uint8)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return mask

# ----------------------------------------------------
# PROCESAMIENTO DE DATASETS
# ----------------------------------------------------
def procesar_datasets():

    for carpeta, cfg in DATASETS.items():

        input_dir = os.path.join(BASE_DATASET, carpeta)
        class_name = cfg["class_name"]
        prefix = cfg["prefix"]

        base_out = os.path.join(BASE_DATASET, f"{carpeta}_Procesada")

        out_gris = os.path.join(base_out, "grises")
        out_ecu = os.path.join(base_out, "ecualizada")
        out_bin = os.path.join(base_out, "binaria")

        for d in [out_gris, out_ecu, out_bin]:
            os.makedirs(d, exist_ok=True)

        files = sorted(os.listdir(input_dir))
        count = 0

        print(f"\nProcesando {carpeta.upper()}")

        for fname in files:

            if count >= MAX_IMGS:
                break

            if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            img_path = os.path.join(input_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # ---------- GRIS (siempre se guarda) ----------
            img_r = rescalar(img)
            gris = escala_grises(img_r)
            cv2.imwrite(
                os.path.join(out_gris, f"{prefix}_{count:03d}.png"),
                gris
            )

            # ---------- ECUALIZADA (siempre se guarda) ----------
            s = stretching(gris)
            ecu = ecualizar(s)
            cv2.imwrite(
                os.path.join(out_ecu, f"{prefix}_{count:03d}.png"),
                ecu
            )

            # ---------- BINARIA (solo si YOLO detecta) ----------
            ecu_bgr = cv2.cvtColor(ecu, cv2.COLOR_GRAY2BGR)
            mask = segmentar_yolo(ecu_bgr, class_name)

            if mask is not None:
                objeto = cv2.bitwise_and(ecu, ecu, mask=mask)
                binaria = otsu(objeto)
                cv2.imwrite(
                    os.path.join(out_bin, f"{prefix}_{count:03d}.png"),
                    binaria
                )

            count += 1
            print(f"✅ [{count}/{MAX_IMGS}] {prefix}_{count:03d}.png")

        print(f"{carpeta.upper()} terminado ({count} imágenes)")

    print("\nTodos los datasets procesados")