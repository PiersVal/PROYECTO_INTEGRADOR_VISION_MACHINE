import cv2
import numpy as np

# ====================================================
#        FUNCIÓN PARA CONVERTIR A ESCALA DE GRISES
# ====================================================

def escala_grises(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ============================
#   1. ESTADÍSTICAS DE IMAGEN
# ============================

def miStat(P):
    media = np.mean(P)
    varianza = np.var(P)
    desviacion = np.std(P)
    return media, varianza, desviacion


# ============================
#   2. AMPLIACIÓN DE HISTOGRAMA
# ============================

def miAmpliH(P, a, b, L):
    P = np.array(P, dtype=float)
    X = L * ((P - a) / (b - a))
    X[X < 0] = 0
    X[X > L] = L
    return X.astype(np.uint8)


# ============================
#   3. TRANSFORMACIÓN CUADRÁTICA
# ============================

def micuadrada(P, L, O):
    P = np.array(P, dtype=float)

    if O == 0:
        X = P**2 / L
    elif O == 1:
        X = np.sqrt(L * P)
    else:
        raise ValueError("O debe ser 0 (cuadrática) u 1 (raíz)")

    X[X < 0] = 0
    X[X > L] = L
    return X.astype(np.uint8)


# ============================
#   4. ECUALIZACIÓN DE HISTOGRAMA
# ============================

def miEcualizador(P):
    L_max = 255
    N, M = P.shape
    total_pixeles = N * M

    hist, _ = np.histogram(P.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = np.round((L_max / total_pixeles) * cdf).astype(np.uint8)

    X = cdf_normalized[P]
    return X


# ================================================
#     PROCESAMIENTO BASADO EN PERCENTILES
#     Clasificación automática de imágenes
# ================================================

def procesar_prioridad_unica(img, percentiles):
    Lmax = 255

    p10 = percentiles[1]
    p25 = percentiles[2]
    p50 = percentiles[3]
    p75 = percentiles[4]
    p90 = percentiles[5]

    if p50 < 85 and p90 < 150:
        salida = micuadrada(img, Lmax, 1)
        return salida, "subexpo_raiz"

    if p50 > 170 and p10 > 100:
        salida = micuadrada(img, Lmax, 0)
        return salida, "sobreexpo_cuadratica"

    if (p90 - p10) < 40:
        a = np.min(img)
        b = np.max(img)
        salida = miAmpliH(img, a, b, Lmax)
        return salida, "bajo_contraste_stretching"

    return img, "original"


# ====================================================
#   API REQUERIDA POR EL BACKEND
#   Reemplaza la lógica interna con tu preprocesamiento real.
# ====================================================

def reescalar_imagen_bytes(image_bytes: bytes, size=(256, 256)) -> bytes:
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("No se pudo decodificar la imagen")

    resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    success, buffer = cv2.imencode(".png", resized)
    if not success:
        raise ValueError("No se pudo codificar la imagen reescalada")

    return buffer.tobytes()


def procesar_imagen_bytes(image_bytes: bytes) -> bytes:
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("No se pudo decodificar la imagen")

    resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

    # === EJEMPLO: convertir a escala de grises ===
    # Reemplaza este bloque por tu lógica real.
    procesada = escala_grises(resized)

    success, buffer = cv2.imencode(".png", procesada)
    if not success:
        raise ValueError("No se pudo codificar la imagen procesada")

    return buffer.tobytes()


def binarizar_imagen_bytes(image_bytes: bytes) -> bytes:
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("No se pudo decodificar la imagen")

    _, binaria = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    success, buffer = cv2.imencode(".png", binaria)
    if not success:
        raise ValueError("No se pudo codificar la imagen binarizada")

    return buffer.tobytes()


def calcular_momentos(image_bytes: bytes) -> dict:
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("No se pudo decodificar la imagen")

    _, binaria = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    moments = cv2.moments(binaria)

    return {
        "m00": moments["m00"],
        "m10": moments["m10"],
        "m01": moments["m01"],
        "m20": moments["m20"],
        "m11": moments["m11"],
        "m02": moments["m02"],
        "m30": moments["m30"],
        "m21": moments["m21"],
        "m12": moments["m12"],
        "m03": moments["m03"],
        "mu20": moments["mu20"],
        "mu11": moments["mu11"],
        "mu02": moments["mu02"],
        "mu30": moments["mu30"],
        "mu21": moments["mu21"],
        "mu12": moments["mu12"],
        "mu03": moments["mu03"],
        "nu20": moments["nu20"],
        "nu11": moments["nu11"],
        "nu02": moments["nu02"],
        "nu30": moments["nu30"],
        "nu21": moments["nu21"],
        "nu12": moments["nu12"],
        "nu03": moments["nu03"],
    }


def calcular_momentos_hu(image_bytes: bytes) -> dict:
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("No se pudo decodificar la imagen")

    _, binaria = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    moments = cv2.moments(binaria)
    hu_moments = cv2.HuMoments(moments).flatten().tolist()

    return {
        "hu1": hu_moments[0],
        "hu2": hu_moments[1],
        "hu3": hu_moments[2],
        "hu4": hu_moments[3],
        "hu5": hu_moments[4],
        "hu6": hu_moments[5],
        "hu7": hu_moments[6],
    }


def calcular_momentos_zernike(image_bytes: bytes, radius: int = 128) -> dict:
    try:
        import mahotas
    except ImportError:
        raise ValueError("Mahotas no está instalado")

    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("No se pudo decodificar la imagen")

    _, binaria = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    zernike = mahotas.features.zernike_moments(binaria, radius=radius, degree=8)

    result = {}
    for i, val in enumerate(zernike):
        result[f"z{i+1}"] = float(val)

    return result


def procesar_sift_con_descriptores(image_bytes: bytes):
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("No se pudo decodificar la imagen")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    salida = cv2.drawKeypoints(
        img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    success, buffer = cv2.imencode(".png", salida)
    if not success:
        raise ValueError("No se pudo codificar la imagen SIFT")

    if descriptors is None:
        desc_list = []
    else:
        desc_list = descriptors.astype(float).tolist()

    return buffer.tobytes(), desc_list


def procesar_hog_con_descriptores(image_bytes: bytes):
    try:
        from skimage.feature import hog
    except ImportError:
        raise ValueError("scikit-image no está instalado")

    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("No se pudo decodificar la imagen")

    features, hog_image = hog(
        img,
        orientations=6,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=True,
        feature_vector=True,
    )

    hog_norm = cv2.normalize(hog_image, None, 0, 255, cv2.NORM_MINMAX)
    hog_uint8 = hog_norm.astype(np.uint8)

    success, buffer = cv2.imencode(".png", hog_uint8)
    if not success:
        raise ValueError("No se pudo codificar la imagen HOG")

    return buffer.tobytes(), features.astype(float).tolist()
