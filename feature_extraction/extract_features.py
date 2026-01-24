import os
import math
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skimage.feature import hog


# ============================================================
# MAPEO DE CLASES A ETIQUETAS NUMÉRICAS
# ============================================================
CLASE_LABELS = {
    # Dataset 1 - Vehículos
    "Motorcycles": 0,
    "Planes": 1,
    "Ships": 2,
    # Dataset 2 - Animales
    "gatto": 3,
    "cavallo": 4,
    "elefante": 5
}


# ============================================================
# 1) UTILIDADES DE CARGA
# ============================================================
def listar_imagenes_binarias(data_root):
    """
    Busca en estructura:
      data/dataset_X/imagenes_procesadas/[clase]/binaria/
    Retorna: paths(list), labels(list), clase_names(list)
    """
    paths, labels, clase_names = [], [], []
    
    for dataset_num in [1, 2]:
        dataset_path = os.path.join(data_root, f"dataset_{dataset_num}", "imagenes_procesadas")
        
        if not os.path.isdir(dataset_path):
            print(f"⚠ No existe: {dataset_path}")
            continue
        
        for clase_folder in os.listdir(dataset_path):
            clase_path = os.path.join(dataset_path, clase_folder)
            if not os.path.isdir(clase_path):
                continue
            
            binaria_path = os.path.join(clase_path, "binaria")
            if not os.path.isdir(binaria_path):
                print(f"⚠ No existe carpeta binaria: {binaria_path}")
                continue
            
            if clase_folder not in CLASE_LABELS:
                print(f"⚠ Clase no reconocida: {clase_folder}")
                continue
            
            label = CLASE_LABELS[clase_folder]
            
            for fn in os.listdir(binaria_path):
                if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                    paths.append(os.path.join(binaria_path, fn))
                    labels.append(label)
                    clase_names.append(clase_folder)
    
    return paths, labels, clase_names


def leer_gris(path, size=(256, 256)):
    """Lee imagen en escala de grises"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo leer: {path}")
    if size is not None:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


# ============================================================
# 2) EXTRACCIÓN DE CARACTERÍSTICAS
# ============================================================
def features_momentos(img_gray):
    """
    Extrae 24 momentos: 10 regulares + 7 centrales + 7 normalizados
    Orden: m00,m10,m01,m20,m11,m02,m30,m21,m12,m03,
            mu20,mu11,mu02,mu30,mu21,mu12,mu03,
            nu20,nu11,nu02,nu30,nu21,nu12,nu03
    """
    M = cv2.moments(img_gray)
    
    # Momentos regulares
    m00 = M['m00']
    m10 = M['m10']
    m01 = M['m01']
    m20 = M['m20']
    m11 = M['m11']
    m02 = M['m02']
    m30 = M['m30']
    m21 = M['m21']
    m12 = M['m12']
    m03 = M['m03']
    
    # Momentos centrales
    mu20 = M['mu20']
    mu11 = M['mu11']
    mu02 = M['mu02']
    mu30 = M['mu30']
    mu21 = M['mu21']
    mu12 = M['mu12']
    mu03 = M['mu03']
    
    # Momentos centrales normalizados
    eps = 1e-10
    nu20 = mu20 / (m00 ** 2 + eps)
    nu11 = mu11 / (m00 ** 2 + eps)
    nu02 = mu02 / (m00 ** 2 + eps)
    nu30 = mu30 / (m00 ** 2.5 + eps)
    nu21 = mu21 / (m00 ** 2.5 + eps)
    nu12 = mu12 / (m00 ** 2.5 + eps)
    nu03 = mu03 / (m00 ** 2.5 + eps)
    
    feats = [
        m00, m10, m01, m20, m11, m02, m30, m21, m12, m03,
        mu20, mu11, mu02, mu30, mu21, mu12, mu03,
        nu20, nu11, nu02, nu30, nu21, nu12, nu03
    ]
    return np.array(feats, dtype=np.float64)


def features_hu(img_gray, log_scale=True, eps=1e-30):
    """
    Extrae 7 Momentos de Hu
    """
    M = cv2.moments(img_gray)
    hu = cv2.HuMoments(M).flatten().astype(np.float64)
    if log_scale:
        hu = -np.sign(hu) * np.log10(np.abs(hu) + eps)
    return hu


def features_zernike(img_gray, radius=21, degree=8):
    """
    Calcula Momentos de Zernike sin dependencias externas
    """
    h, w = img_gray.shape
    cy, cx = h // 2, w // 2
    
    # Crear malla de coordenadas
    y, x = np.ogrid[:h, :w]
    x = x - cx
    y = y - cy
    rho = np.sqrt(x**2 + y**2) / radius
    rho = np.clip(rho, 0, 1)
    theta = np.arctan2(y, x)
    
    # Normalizar imagen
    img_norm = img_gray.astype(np.float64)
    img_norm = img_norm / (np.max(img_norm) + 1e-10)
    
    zernike_feats = []
    
    # Calcular momentos de Zernike
    for n in range(degree + 1):
        for m in range(-n, n + 1, 2):
            # Polinomio radial Zernike
            vnm = 0
            for s in range((n - abs(m)) // 2 + 1):
                coeff = ((-1) ** s * math.factorial(n - s)) / (
                    math.factorial(s) * 
                    math.factorial((n - 2*s + abs(m)) // 2) * 
                    math.factorial((n - 2*s - abs(m)) // 2)
                )
                vnm += coeff * (rho ** (n - 2*s))
            
            # Función angular
            real_part = vnm * np.cos(m * theta)
            imag_part = vnm * np.sin(m * theta)
            
            # Calcular el momento
            zmn_real = np.sum(img_norm * real_part)
            zmn_imag = np.sum(img_norm * imag_part)
            zmn_mag = np.sqrt(zmn_real**2 + zmn_imag**2)
            
            zernike_feats.append(zmn_mag)
    
    return np.array(zernike_feats, dtype=np.float64)


def features_sift(img_gray):
    """
    Extrae características SIFT: estadísticas (mean, std, min, max) de los 128 descriptores
    Total: 512 características (128 dimensiones × 4 estadísticas)
    Si no hay keypoints: retorna vector de ceros
    """
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img_gray, None)
    
    # Si no hay descriptores, retornar vector de ceros
    if descriptors is None:
        return np.zeros(512, dtype=np.float64)
    
    # descriptors shape: (num_keypoints, 128)
    descriptors = descriptors.astype(np.float64)
    
    # Calcular estadísticas sobre cada una de las 128 dimensiones
    mean_desc = np.mean(descriptors, axis=0)  # 128
    std_desc = np.std(descriptors, axis=0)    # 128
    min_desc = np.min(descriptors, axis=0)    # 128
    max_desc = np.max(descriptors, axis=0)    # 128
    
    # Concatenar: mean + std + min + max = 512 características
    sift_feats = np.concatenate([mean_desc, std_desc, min_desc, max_desc])
    
    return sift_feats


def features_hog(img_gray):
    """
    Extrae características HOG (Histogram of Oriented Gradients)
    Parámetros: orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2)
    Retorna: vector de características HOG
    """
    # Asegurar que es grayscale
    if len(img_gray.shape) == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    
    # Extraer HOG
    features = hog(
        img_gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=True
    )
    
    return features.astype(np.float64)
    
    # Binarizar
    _, bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw01 = (bw > 0).astype(np.uint8)
    
    # Calcular Zernike
    z = mahotas.features.zernike_moments(bw01, radius=radius, degree=degree)
    return np.array(z, dtype=np.float64)


# ============================================================
# 3) CONSTRUCCIÓN DE DATASETS
# ============================================================
def construir_dataset(paths, labels, extractor_fn, nombres_clases=None, feature_names=None):
    """
    Extrae características de todas las imágenes
    """
    X = []
    valid_labels = []
    valid_nombres = []
    
    for i, p in enumerate(paths):
        try:
            img = leer_gris(p)
            feats = extractor_fn(img)
            X.append(feats)
            valid_labels.append(labels[i])
            if nombres_clases is not None:
                valid_nombres.append(nombres_clases[i])
        except Exception as e:
            print(f" Error procesando {p}: {e}")
            continue
    
    X = np.vstack(X)
    
    # Crear DataFrame con nombres de columnas si se proporcionan
    if feature_names is not None:
        df = pd.DataFrame(X, columns=feature_names)
    else:
        df = pd.DataFrame(X)
    
    if nombres_clases is not None:
        df.insert(0, "Clase", valid_nombres)  # Guarda el nombre de la clase
    else:
        df.insert(0, "Clase", np.array(valid_labels, dtype=int))
    
    return df


def generar_datasets_features(
    data_root="data",
    out_dir="data/features",
    z_radius=21,
    z_degree=8
):
    """
    Genera 3 datasets completos con características diferentes para cada dataset (1 y 2):
    1. Momentos clásicos
    2. Momentos de Hu
    3. Momentos de Zernike
    
    Estructura: data/features/dataset_1/, data/features/dataset_2/
    """
    
    os.makedirs(out_dir, exist_ok=True)
    
    for dataset_num in [1, 2]:
        print("\n" + "=" * 60)
        print(f"PROCESANDO DATASET {dataset_num}")
        print("=" * 60)
        
        # Crear carpeta específica del dataset
        dataset_out_dir = os.path.join(out_dir, f"dataset_{dataset_num}")
        os.makedirs(dataset_out_dir, exist_ok=True)
        
        # Leer imágenes del dataset específico
        paths, labels, clase_names = [], [], []
        dataset_path = os.path.join(data_root, f"dataset_{dataset_num}", "imagenes_procesadas")
        
        if not os.path.isdir(dataset_path):
            print(f" No existe: {dataset_path}")
            continue
        
        for clase_folder in os.listdir(dataset_path):
            clase_path = os.path.join(dataset_path, clase_folder)
            if not os.path.isdir(clase_path):
                continue
            
            binaria_path = os.path.join(clase_path, "binaria")
            if not os.path.isdir(binaria_path):
                continue
            
            if clase_folder not in CLASE_LABELS:
                continue
            
            label = CLASE_LABELS[clase_folder]
            
            for fn in os.listdir(binaria_path):
                if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                    paths.append(os.path.join(binaria_path, fn))
                    labels.append(label)
                    clase_names.append(clase_folder)
        
        if len(paths) == 0:
            print(f" No se encontraron imágenes en dataset {dataset_num}")
            continue
        
        print(f" Total de imágenes: {len(paths)}")
        print(f"Clases encontradas: {set(clase_names)}")
        
        # --- Dataset: Momentos ---
        print(f"\n⏳ Extrayendo Momentos clásicos...")
        feature_names_momentos = [
            "m00", "m10", "m01", "m20", "m11", "m02", "m30", "m21", "m12", "m03",
            "mu20", "mu11", "mu02", "mu30", "mu21", "mu12", "mu03",
            "nu20", "nu11", "nu02", "nu30", "nu21", "nu12", "nu03"
        ]
        df_momentos = construir_dataset(paths, labels, features_momentos, clase_names, feature_names_momentos)
        
        momentos_path = os.path.join(dataset_out_dir, "dataset_momentos.csv")
        df_momentos.to_csv(momentos_path, index=False)
        print(f" Guardado: {momentos_path}")
        print(f"   Tamaño: {df_momentos.shape}")
        
        # --- Dataset: Hu ---
        print(f"\n Extrayendo Momentos de Hu...")
        df_hu = construir_dataset(paths, labels, lambda im: features_hu(im, log_scale=True), clase_names)
        
        hu_path = os.path.join(dataset_out_dir, "dataset_hu.csv")
        df_hu.to_csv(hu_path, index=False)
        print(f" Guardado: {hu_path}")
        print(f"   Tamaño: {df_hu.shape}")
        
        # --- Dataset: Zernike ---
        print(f"\n Extrayendo Momentos de Zernike...")
        df_zernike = construir_dataset(
            paths, labels, 
            lambda im: features_zernike(im, radius=z_radius, degree=z_degree), 
            clase_names
        )
        
        zernike_path = os.path.join(dataset_out_dir, "dataset_zernike.csv")
        df_zernike.to_csv(zernike_path, index=False)
        print(f" Guardado: {zernike_path}")
        print(f"   Tamaño: {df_zernike.shape}")
    
    print("\n" + "=" * 60)
    print("EXTRACCIÓN COMPLETADA ")
    print("=" * 60)
    print(f"\nMapeo de clases:")
    for clase, label in sorted(CLASE_LABELS.items(), key=lambda x: x[1]):
        print(f"  {label}: {clase}")


def generar_datasets_sift(data_root="data", out_dir="data/features"):
    """
    Extrae características SIFT de las imágenes de contraste
    Genera dataset_sift.csv para cada dataset (1 y 2)
    """
    
    os.makedirs(out_dir, exist_ok=True)
    
    for dataset_num in [1, 2]:
        print("\n" + "=" * 60)
        print(f"PROCESANDO SIFT - DATASET {dataset_num}")
        print("=" * 60)
        
        # Crear carpeta específica del dataset
        dataset_out_dir = os.path.join(out_dir, f"dataset_{dataset_num}")
        os.makedirs(dataset_out_dir, exist_ok=True)
        
        # Leer imágenes de CONTRASTE del dataset específico
        paths, labels, clase_names = [], [], []
        dataset_path = os.path.join(data_root, f"dataset_{dataset_num}", "imagenes_procesadas")
        
        if not os.path.isdir(dataset_path):
            print(f" No existe: {dataset_path}")
            continue
        
        for clase_folder in os.listdir(dataset_path):
            clase_path = os.path.join(dataset_path, clase_folder)
            if not os.path.isdir(clase_path):
                continue
            
            # CAMBIO: usar carpeta "contraste" en lugar de "binaria"
            contraste_path = os.path.join(clase_path, "contraste")
            if not os.path.isdir(contraste_path):
                continue
            
            if clase_folder not in CLASE_LABELS:
                continue
            
            label = CLASE_LABELS[clase_folder]
            
            for fn in os.listdir(contraste_path):
                if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                    paths.append(os.path.join(contraste_path, fn))
                    labels.append(label)
                    clase_names.append(clase_folder)
        
        if len(paths) == 0:
            print(f" No se encontraron imágenes en dataset {dataset_num}")
            continue
        
        print(f" Total de imágenes: {len(paths)}")
        print(f"Clases encontradas: {set(clase_names)}")
        
        # --- Dataset: SIFT ---
        print(f"\n Extrayendo características SIFT...")
        df_sift = construir_dataset(paths, labels, features_sift, clase_names)
        
        sift_path = os.path.join(dataset_out_dir, "dataset_sift.csv")
        df_sift.to_csv(sift_path, index=False)
        print(f" Guardado: {sift_path}")
        print(f"   Tamaño: {df_sift.shape}")
    
    print("\n" + "=" * 60)
    print("EXTRACCIÓN SIFT COMPLETADA ")
    print("=" * 60)
    print(f"\nMapeo de clases:")
    for clase, label in sorted(CLASE_LABELS.items(), key=lambda x: x[1]):
        print(f"  {label}: {clase}")


def generar_datasets_hog(data_root="data", out_dir="data/features"):
    """
    Extrae características HOG de las imágenes de contraste
    Genera dataset_hog.csv para cada dataset (1 y 2)
    """
    
    os.makedirs(out_dir, exist_ok=True)
    
    for dataset_num in [1, 2]:
        print("\n" + "=" * 60)
        print(f"PROCESANDO HOG - DATASET {dataset_num}")
        print("=" * 60)
        
        # Crear carpeta específica del dataset
        dataset_out_dir = os.path.join(out_dir, f"dataset_{dataset_num}")
        os.makedirs(dataset_out_dir, exist_ok=True)
        
        # Leer imágenes de CONTRASTE del dataset específico
        paths, labels, clase_names = [], [], []
        dataset_path = os.path.join(data_root, f"dataset_{dataset_num}", "imagenes_procesadas")
        
        if not os.path.isdir(dataset_path):
            print(f" No existe: {dataset_path}")
            continue
        
        for clase_folder in os.listdir(dataset_path):
            clase_path = os.path.join(dataset_path, clase_folder)
            if not os.path.isdir(clase_path):
                continue
            
            # Usar carpeta "contraste"
            contraste_path = os.path.join(clase_path, "contraste")
            if not os.path.isdir(contraste_path):
                continue
            
            if clase_folder not in CLASE_LABELS:
                continue
            
            label = CLASE_LABELS[clase_folder]
            
            for fn in os.listdir(contraste_path):
                if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                    paths.append(os.path.join(contraste_path, fn))
                    labels.append(label)
                    clase_names.append(clase_folder)
        
        if len(paths) == 0:
            print(f" No se encontraron imágenes en dataset {dataset_num}")
            continue
        
        print(f" Total de imágenes: {len(paths)}")
        print(f"Clases encontradas: {set(clase_names)}")
        
        # --- Dataset: HOG ---
        print(f"\n Extrayendo características HOG...")
        df_hog = construir_dataset(paths, labels, features_hog, clase_names)
        
        hog_path = os.path.join(dataset_out_dir, "dataset_hog.csv")
        df_hog.to_csv(hog_path, index=False)
        print(f" Guardado: {hog_path}")
        print(f"   Tamaño: {df_hog.shape}")
    
    print("\n" + "=" * 60)
    print("EXTRACCIÓN HOG COMPLETADA ")
    print("=" * 60)
    print(f"\nMapeo de clases:")
    for clase, label in sorted(CLASE_LABELS.items(), key=lambda x: x[1]):
        print(f"  {label}: {clase}")


if __name__ == "__main__":
    generar_datasets_features(
        data_root="data",
        test_size=0.3,
        random_state=42,
        out_dir="feature_extraction/datasets"
    )
