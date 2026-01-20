import os
import cv2
import numpy as np
from ultralytics import YOLO


# ====================================================
# CONFIGURACIÓN
# ====================================================
MODEL_PATH = "src/models/yolov8n-seg.pt"
OUTPUT_BASE = "dataset/animals_preprocesado"
MAX_IMGS = 100

CLASES = {
    "gatto": "cat",
    "cavallo": "horse",
    "elefante": "elephant"
}

EXT_VALIDAS = (".jpg", ".jpeg", ".png", ".bmp")

# ====================================================
# CARGAR MODELO
# ====================================================
model = YOLO(MODEL_PATH)


# ====================================================
# PIPELINE CLÁSICO
# ====================================================
def escala_grises(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def stretching(img):
    img = img / 255.0
    a, b = img.min(), img.max()
    if b - a == 0:
        return (img * 255).astype(np.uint8)
    return np.clip((img - a) / (b - a) * 255, 0, 255).astype(np.uint8)


def miAmpliH(img):
    return np.clip(img * 1.5, 0, 255).astype(np.uint8)


def miCuadrada(img):
    return np.clip((img / 255.0) ** 2 * 255, 0, 255).astype(np.uint8)


def miRaiz(img):
    return np.clip(np.sqrt(img / 255.0) * 255, 0, 255).astype(np.uint8)


def miEcualizador(img):
    return cv2.equalizeHist(img)


def otsu_manual(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()
    hist /= hist.sum()
    omega = np.cumsum(hist)
    mu = np.cumsum(hist * np.arange(256))
    mu_t = mu[-1]
    sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1 - omega) + 1e-10)
    t = np.argmax(sigma_b)
    _, binaria = cv2.threshold(img, t, 255, cv2.THRESH_BINARY)
    return binaria


def pipeline_clasico(img):
    g = escala_grises(img)
    s = stretching(g)
    a = miAmpliH(s)
    c = miCuadrada(a)
    r = miRaiz(c)
    e = miEcualizador(r)
    o = otsu_manual(e)
    return o


# ====================================================
# PROCESAMIENTO POR CLASE
# ====================================================
def procesar_clase(ruta_entrada, ruta_salida, class_name, prefijo):
    os.makedirs(ruta_salida, exist_ok=True)
    archivos = sorted(os.listdir(ruta_entrada))
    count = 0

    print(f"\nProcesando {prefijo.upper()}")

    for fname in archivos:
        if count >= MAX_IMGS:
            break
        if not fname.lower().endswith(EXT_VALIDAS):
            continue

        img_path = os.path.join(ruta_entrada, fname)
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

        class_id = [k for k, v in names.items() if v == class_name][0]
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

        out_name = f"{prefijo}_{count:03d}.png"
        cv2.imwrite(os.path.join(ruta_salida, out_name), img_final)
        print(out_name)

        count += 1

    print(f"{count} imágenes procesadas")

