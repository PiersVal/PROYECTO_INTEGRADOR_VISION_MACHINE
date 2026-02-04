import json
import os
import uuid

import numpy as np
from typing import List
from sklearn.preprocessing import normalize

from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from PROCESAMIENTO_IMG import (
    procesar_imagen_bytes,
    reescalar_imagen_bytes,
    binarizar_imagen_bytes,
    calcular_momentos,
    calcular_momentos_hu,
    calcular_momentos_zernike,
    procesar_sift_con_descriptores,
    procesar_hog_con_descriptores,
)
from cnn import procesar_cnn_con_descriptores
from clustering_online import LinksClusterCapacityOnline

from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score
)
from sklearn.preprocessing import LabelEncoder

ALLOWED_TYPES = {"image/jpeg": ".jpg", "image/png": ".png"}
MAX_FILE_SIZE = 10 * 1024 * 1024

DATA_DIR = os.getenv("DATA_DIR", "/data")
ORIGINAL_DIR = os.path.join(DATA_DIR, "originals")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
BINARIZED_DIR = os.path.join(DATA_DIR, "binarized")
INDEX_FILE = os.path.join(DATA_DIR, "index.json")
CLUSTER_STATE_FILE = os.path.join(DATA_DIR, "cluster_state.json")
CLUSTER_STATE_FILE_HU = os.path.join(DATA_DIR, "cluster_state_hu.json")
CLUSTER_STATE_FILE_ZERNIKE = os.path.join(DATA_DIR, "cluster_state_zernike.json")
CLUSTER_STATE_FILE_SIFT = os.path.join(DATA_DIR, "cluster_state_sift.json")
CLUSTER_STATE_FILE_HOG = os.path.join(DATA_DIR, "cluster_state_hog.json")
CLUSTER_STATE_FILE_CNN = os.path.join(DATA_DIR, "cluster_state_cnn.json")

os.makedirs(ORIGINAL_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(BINARIZED_DIR, exist_ok=True)

app = FastAPI()

# Configurar CORS para permitir frontend local y en producción

allowed_origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://127.0.0.1",
    "http://127.0.0.1:8080",
    "https://remontada-uzn6.onrender.com",  # backend mismo
    "https://remontada-1.onrender.com"      # frontend
]

# En desarrollo local (Docker), permitir todos los orígenes
if os.getenv("ENVIRONMENT", "development") == "development":
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


def load_index() -> List[dict]:
    if not os.path.exists(INDEX_FILE):
        return []
    try:
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def save_index(items: List[dict]) -> None:
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def save_cluster_state() -> None:
    """Guarda el estado del modelo de clustering"""
    if CLUSTER_MODEL:
        with open(CLUSTER_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(CLUSTER_MODEL.to_dict(), f, ensure_ascii=False, indent=2)


def save_cluster_state_hu() -> None:
    """Guarda el estado del modelo de clustering Hu"""
    if CLUSTER_MODEL_HU:
        with open(CLUSTER_STATE_FILE_HU, "w", encoding="utf-8") as f:
            json.dump(CLUSTER_MODEL_HU.to_dict(), f, ensure_ascii=False, indent=2)


def save_cluster_state_zernike() -> None:
    """Guarda el estado del modelo de clustering Zernike"""
    if CLUSTER_MODEL_ZERNIKE:
        with open(CLUSTER_STATE_FILE_ZERNIKE, "w", encoding="utf-8") as f:
            json.dump(CLUSTER_MODEL_ZERNIKE.to_dict(), f, ensure_ascii=False, indent=2)


def save_cluster_state_sift() -> None:
    """Guarda el estado del modelo de clustering SIFT"""
    if CLUSTER_MODEL_SIFT:
        with open(CLUSTER_STATE_FILE_SIFT, "w", encoding="utf-8") as f:
            json.dump(CLUSTER_MODEL_SIFT.to_dict(), f, ensure_ascii=False, indent=2)


def save_cluster_state_hog() -> None:
    """Guarda el estado del modelo de clustering HOG"""
    if CLUSTER_MODEL_HOG:
        with open(CLUSTER_STATE_FILE_HOG, "w", encoding="utf-8") as f:
            json.dump(CLUSTER_MODEL_HOG.to_dict(), f, ensure_ascii=False, indent=2)


def save_cluster_state_cnn() -> None:
    """Guarda el estado del modelo de clustering CNN"""
    if CLUSTER_MODEL_CNN:
        with open(CLUSTER_STATE_FILE_CNN, "w", encoding="utf-8") as f:
            json.dump(CLUSTER_MODEL_CNN.to_dict(), f, ensure_ascii=False, indent=2)


def load_cluster_state() -> None:
    """Carga el estado del modelo de clustering"""
    global CLUSTER_MODEL, CLUSTER_CAPACITIES
    # NO cargar el estado automáticamente al iniciar
    # Esto permite que cada /analyze cree un nuevo modelo
    pass


CLUSTER_MODEL: LinksClusterCapacityOnline | None = None
CLUSTER_CAPACITIES: list[int] | None = None

# Variables globales para el modelo de Hu Moments
CLUSTER_MODEL_HU: LinksClusterCapacityOnline | None = None
CLUSTER_CAPACITIES_HU: list[int] | None = None

# Variables globales para el modelo de Zernike Moments
CLUSTER_MODEL_ZERNIKE: LinksClusterCapacityOnline | None = None
CLUSTER_CAPACITIES_ZERNIKE: list[int] | None = None

# Variables globales para el modelo de SIFT
CLUSTER_MODEL_SIFT: LinksClusterCapacityOnline | None = None
CLUSTER_CAPACITIES_SIFT: list[int] | None = None

# Variables globales para el modelo de HOG
CLUSTER_MODEL_HOG: LinksClusterCapacityOnline | None = None
CLUSTER_CAPACITIES_HOG: list[int] | None = None

# Variables globales para el modelo de CNN
CLUSTER_MODEL_CNN: LinksClusterCapacityOnline | None = None
CLUSTER_CAPACITIES_CNN: list[int] | None = None

# Variables globales para Momentos con Etiquetas (External Metrics)
EXTERNAL_METRICS_STATE = None
CLUSTER_MODEL_MOMENTS_LABELED: LinksClusterCapacityOnline | None = None

# Variables globales para Hu con Etiquetas
EXTERNAL_METRICS_HU_STATE = None
CLUSTER_MODEL_HU_LABELED: LinksClusterCapacityOnline | None = None

# Variables globales para Zernike con Etiquetas
EXTERNAL_METRICS_ZERNIKE_STATE = None
CLUSTER_MODEL_ZERNIKE_LABELED: LinksClusterCapacityOnline | None = None

# Variables globales para SIFT con Etiquetas
EXTERNAL_METRICS_SIFT_STATE = None
CLUSTER_MODEL_SIFT_LABELED: LinksClusterCapacityOnline | None = None

# Variables globales para HOG con Etiquetas
EXTERNAL_METRICS_HOG_STATE = None
CLUSTER_MODEL_HOG_LABELED: LinksClusterCapacityOnline | None = None

# Variables globales para CNN con Etiquetas
EXTERNAL_METRICS_CNN_STATE = None
CLUSTER_MODEL_CNN_LABELED: LinksClusterCapacityOnline | None = None

# NO cargar estado del clustering al iniciar
# load_cluster_state()


def parse_capacities(capacities_text: str) -> list[int]:
    parts = [p.strip() for p in capacities_text.split(",") if p.strip()]
    if not parts:
        raise ValueError("capacities vacío")
    return [int(p) for p in parts]


def get_cluster_model(capacities: list[int]) -> LinksClusterCapacityOnline:
    global CLUSTER_MODEL, CLUSTER_CAPACITIES
    if CLUSTER_MODEL is None or CLUSTER_CAPACITIES != capacities:
        CLUSTER_MODEL = LinksClusterCapacityOnline(capacities=capacities)
        CLUSTER_CAPACITIES = capacities
    return CLUSTER_MODEL


@app.get("/images")
def list_images():
    return load_index()


@app.delete("/images")
def delete_images():
    items = load_index()
    for item in items:
        original_name = os.path.basename(item.get("original_url", ""))
        processed_name = os.path.basename(item.get("processed_url", ""))
        binarized_name = os.path.basename(item.get("binarized_url", ""))

        if original_name:
            original_path = os.path.join(ORIGINAL_DIR, original_name)
            if os.path.exists(original_path):
                os.remove(original_path)

        if processed_name:
            processed_path = os.path.join(PROCESSED_DIR, processed_name)
            if os.path.exists(processed_path):
                os.remove(processed_path)

        if binarized_name:
            binarized_path = os.path.join(BINARIZED_DIR, binarized_name)
            if os.path.exists(binarized_path):
                os.remove(binarized_path)

    save_index([])
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_images(
    files: List[UploadFile] = File(...),
    capacities: str | None = Form(None),
    clusters: int | None = Form(None),
    reset: bool = Form(False),
):
    if not files:
        raise HTTPException(status_code=400, detail="No se enviaron archivos")

    # Si reset=True o si se proporcionan nuevas capacidades, reiniciar el modelo
    if reset or capacities or clusters:
        global CLUSTER_MODEL
        CLUSTER_MODEL = None

    results = []

    if capacities is None and clusters is None:
        raise HTTPException(
            status_code=400,
            detail="Debes indicar capacities o número de clusters",
        )

    if capacities:
        try:
            caps = parse_capacities(capacities)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"capacities inválido: {exc}")
    else:
        try:
            k = int(clusters) if clusters is not None else 0
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"clusters inválido: {exc}")
        if k <= 0:
            raise HTTPException(status_code=400, detail="clusters inválido")
        caps = [100] * k

    model = get_cluster_model(caps)
    moment_keys = [
        "m00",
        "m10",
        "m01",
        "m20",
        "m11",
        "m02",
        "m30",
        "m21",
        "m12",
        "m03",
        "mu20",
        "mu11",
        "mu02",
        "mu30",
        "mu21",
        "mu12",
        "mu03",
        "nu20",
        "nu11",
        "nu02",
        "nu30",
        "nu21",
        "nu12",
        "nu03",
    ]

    for file in files:
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(status_code=400, detail=f"Tipo no permitido: {file.content_type}")

        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"Archivo demasiado grande: {file.filename}")

        ext = ALLOWED_TYPES[file.content_type]
        processed_ext = ".png"
        image_id = uuid.uuid4().hex
        original_name = f"{image_id}{ext}"
        processed_name = f"{image_id}_processed{processed_ext}"
        binarized_name = f"{image_id}_binarized{processed_ext}"

        original_path = os.path.join(ORIGINAL_DIR, original_name)
        processed_path = os.path.join(PROCESSED_DIR, processed_name)
        binarized_path = os.path.join(BINARIZED_DIR, binarized_name)

        try:
            resized_bytes = reescalar_imagen_bytes(content)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Error reescalando {file.filename}: {exc}")

        with open(original_path, "wb") as f:
            f.write(resized_bytes)

        try:
            processed_bytes = procesar_imagen_bytes(resized_bytes)
            binarized_bytes = binarizar_imagen_bytes(resized_bytes)
            # Calcular momentos usando la imagen binarizada
            momentos = calcular_momentos(binarized_bytes)
            vector = [float(momentos[k]) for k in moment_keys]
            vector_array = np.array(vector, dtype=float).reshape(1, -1)
            
            # Normalizar el vector con L2 (cada imagen se escala independientemente)
            vector_normalizado = normalize(vector_array, norm='l2')[0]
            
            cluster_id, last_centroid = model.predict_with_centroid(vector_normalizado)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error procesando {file.filename}: {exc}")

        with open(processed_path, "wb") as f:
            f.write(processed_bytes)

        with open(binarized_path, "wb") as f:
            f.write(binarized_bytes)

        result = {
            "id": image_id,
            "filename": file.filename,
            "original_url": f"/files/originals/{original_name}",
            "processed_url": f"/files/processed/{processed_name}",
            "binarized_url": f"/files/binarized/{binarized_name}",
            "momentos": momentos,
            "cluster_id": cluster_id,
            "ultimo_centroide": last_centroid.tolist(),
        }
        results.append(result)

        print(f"[CLUSTER] id={image_id} cluster={cluster_id} centroid={last_centroid.tolist()}")

    # Guardar estado del modelo después de procesar todas las imágenes
    save_cluster_state()
    
    # Calcular métricas de evaluación
    dunn_index = model.calculate_dunn_index()
    silhouette_coefficient = model.calculate_silhouette_coefficient()
    
    return {
        "results": results,
        "metrics": {
            "dunn_index": round(float(dunn_index), 4),
            "silhouette_coefficient": round(float(silhouette_coefficient), 4),
        }
    }
@app.post("/analyze-zernike")
async def analyze_images_zernike(
    files: List[UploadFile] = File(...),
    capacities: str = Form(None),
    clusters: int = Form(None)
):
    """
    Analiza imágenes con Momentos de Zernike y realiza clustering
    """
    global CLUSTER_MODEL_ZERNIKE, CLUSTER_CAPACITIES_ZERNIKE
    
    if not files:
        raise HTTPException(status_code=400, detail="No se enviaron archivos")

    # Inicializar el modelo de clustering si se proporcionan capacidades o clusters
    if capacities or clusters:
        try:
            caps = parse_capacities(capacities) if capacities else [10] * clusters
            CLUSTER_MODEL_ZERNIKE = LinksClusterCapacityOnline(capacities=caps)
            CLUSTER_CAPACITIES_ZERNIKE = caps
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Error inicializando clustering: {exc}")

    results = []
    zernike_keys = [f"z{i}" for i in range(1, 26)]  # Zernike tiene 25 momentos: z1 a z25

    for file in files:
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(status_code=400, detail=f"Tipo no permitido: {file.content_type}")

        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"Archivo demasiado grande: {file.filename}")

        ext = ALLOWED_TYPES[file.content_type]
        processed_ext = ".png"
        image_id = uuid.uuid4().hex
        original_name = f"{image_id}{ext}"
        processed_name = f"{image_id}_processed{processed_ext}"
        binarized_name = f"{image_id}_binarized{processed_ext}"

        original_path = os.path.join(ORIGINAL_DIR, original_name)
        processed_path = os.path.join(PROCESSED_DIR, processed_name)
        binarized_path = os.path.join(BINARIZED_DIR, binarized_name)

        try:
            resized_bytes = reescalar_imagen_bytes(content)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Error reescalando {file.filename}: {exc}")

        with open(original_path, "wb") as f:
            f.write(resized_bytes)

        try:
            processed_bytes = procesar_imagen_bytes(resized_bytes)
            binarized_bytes = binarizar_imagen_bytes(resized_bytes)
            momentos_zernike = calcular_momentos_zernike(resized_bytes)
            
            # Si hay modelo de clustering, hacer predicción
            cluster_id = None
            ultimo_centroide = None
            if CLUSTER_MODEL_ZERNIKE:
                # Extraer valores de los 25 momentos de Zernike
                vector = [float(momentos_zernike[k]) for k in zernike_keys]
                vector_array = np.array(vector, dtype=float).reshape(1, -1)
                
                # Normalizar el vector con L2
                vector_normalizado = normalize(vector_array, norm='l2')[0]
                
                cluster_id, ultimo_centroide = CLUSTER_MODEL_ZERNIKE.predict_with_centroid(
                    vector_normalizado,
                    allow_new_clusters=True,
                )
                
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error procesando {file.filename}: {exc}")

        with open(processed_path, "wb") as f:
            f.write(processed_bytes)

        with open(binarized_path, "wb") as f:
            f.write(binarized_bytes)

        result = {
            "id": image_id,
            "filename": file.filename,
            "original_url": f"/files/originals/{original_name}",
            "processed_url": f"/files/processed/{processed_name}",
            "binarized_url": f"/files/binarized/{binarized_name}",
            "momentos_zernike": momentos_zernike,
        }
        
        if cluster_id is not None:
            result["cluster_id"] = cluster_id
            result["ultimo_centroide"] = ultimo_centroide.tolist()
            print(f"[ZERNIKE-CLUSTER] id={image_id} cluster={cluster_id} centroid={ultimo_centroide.tolist()}")
        
        results.append(result)

    # Guardar estado del modelo si se hizo clustering
    if CLUSTER_MODEL_ZERNIKE:
        save_cluster_state_zernike()
        
        # Calcular métricas de evaluación
        dunn_index = CLUSTER_MODEL_ZERNIKE.calculate_dunn_index()
        silhouette_coefficient = CLUSTER_MODEL_ZERNIKE.calculate_silhouette_coefficient()
        
        return {
            "results": results,
            "metrics": {
                "dunn_index": round(float(dunn_index), 4),
                "silhouette_coefficient": round(float(silhouette_coefficient), 4),
            }
        }

    return {"results": results}


@app.post("/analyze-sift")
async def analyze_images_sift(
    files: List[UploadFile] = File(...),
    capacities: str = Form(None),
    clusters: int = Form(None)
):
    """
    Analiza imágenes con SIFT y realiza clustering usando promedio de descriptores
    """
    global CLUSTER_MODEL_SIFT, CLUSTER_CAPACITIES_SIFT
    
    if not files:
        raise HTTPException(status_code=400, detail="No se enviaron archivos")

    # Inicializar el modelo de clustering si se proporcionan capacidades o clusters
    if capacities or clusters:
        try:
            caps = parse_capacities(capacities) if capacities else [10] * clusters
            CLUSTER_MODEL_SIFT = LinksClusterCapacityOnline(capacities=caps)
            CLUSTER_CAPACITIES_SIFT = caps
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Error inicializando clustering: {exc}")

    results = []

    for file in files:
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(status_code=400, detail=f"Tipo no permitido: {file.content_type}")

        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"Archivo demasiado grande: {file.filename}")

        ext = ALLOWED_TYPES[file.content_type]
        processed_ext = ".png"
        image_id = uuid.uuid4().hex
        original_name = f"{image_id}{ext}"
        processed_name = f"{image_id}_sift{processed_ext}"

        original_path = os.path.join(ORIGINAL_DIR, original_name)
        processed_path = os.path.join(PROCESSED_DIR, processed_name)

        try:
            resized_bytes = reescalar_imagen_bytes(content)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Error reescalando {file.filename}: {exc}")

        with open(original_path, "wb") as f:
            f.write(resized_bytes)

        try:
            sift_bytes, descriptores = procesar_sift_con_descriptores(resized_bytes)
            
            # Si hay modelo de clustering, hacer predicción
            cluster_id = None
            ultimo_centroide = None
            if CLUSTER_MODEL_SIFT and descriptores:
                # Usar el promedio de todos los descriptores como vector característico
                descriptores_array = np.array(descriptores, dtype=float)
                vector = np.mean(descriptores_array, axis=0)  # Promedio de descriptores (128 dim)
                vector = vector.reshape(1, -1)
                
                # Normalizar el vector con L2
                vector_normalizado = normalize(vector, norm='l2')[0]
                
                cluster_id, ultimo_centroide = CLUSTER_MODEL_SIFT.predict_with_centroid(
                    vector_normalizado,
                    allow_new_clusters=True,
                )
                
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error procesando {file.filename}: {exc}")

        with open(processed_path, "wb") as f:
            f.write(sift_bytes)

        result = {
            "id": image_id,
            "filename": file.filename,
            "original_url": f"/files/originals/{original_name}",
            "processed_url": f"/files/processed/{processed_name}",
            "descriptores": descriptores,
            "num_keypoints": len(descriptores) if descriptores else 0,
        }
        
        if cluster_id is not None:
            result["cluster_id"] = cluster_id
            result["ultimo_centroide"] = ultimo_centroide.tolist()
            print(f"[SIFT-CLUSTER] id={image_id} cluster={cluster_id} keypoints={len(descriptores)} centroid={ultimo_centroide.tolist()}")
        
        results.append(result)

    # Guardar estado del modelo si se hizo clustering
    if CLUSTER_MODEL_SIFT:
        save_cluster_state_sift()
        
        # Calcular métricas de evaluación
        dunn_index = CLUSTER_MODEL_SIFT.calculate_dunn_index()
        silhouette_coefficient = CLUSTER_MODEL_SIFT.calculate_silhouette_coefficient()
        
        return {
            "results": results,
            "metrics": {
                "dunn_index": round(float(dunn_index), 4),
                "silhouette_coefficient": round(float(silhouette_coefficient), 4),
            }
        }

    return {"results": results}


@app.post("/analyze-hog")
async def analyze_images_hog(
    files: List[UploadFile] = File(...),
    capacities: str = Form(None),
    clusters: int = Form(None)
):
    """
    Analiza imágenes con HOG y realiza clustering usando los descriptores
    """
    global CLUSTER_MODEL_HOG, CLUSTER_CAPACITIES_HOG
    
    if not files:
        raise HTTPException(status_code=400, detail="No se enviaron archivos")

    # Inicializar el modelo de clustering si se proporcionan capacidades o clusters
    if capacities or clusters:
        try:
            caps = parse_capacities(capacities) if capacities else [10] * clusters
            CLUSTER_MODEL_HOG = LinksClusterCapacityOnline(capacities=caps)
            CLUSTER_CAPACITIES_HOG = caps
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Error inicializando clustering: {exc}")

    results = []

    for file in files:
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(status_code=400, detail=f"Tipo no permitido: {file.content_type}")

        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"Archivo demasiado grande: {file.filename}")

        ext = ALLOWED_TYPES[file.content_type]
        processed_ext = ".png"
        image_id = uuid.uuid4().hex
        original_name = f"{image_id}{ext}"
        processed_name = f"{image_id}_hog{processed_ext}"

        original_path = os.path.join(ORIGINAL_DIR, original_name)
        processed_path = os.path.join(PROCESSED_DIR, processed_name)

        try:
            resized_bytes = reescalar_imagen_bytes(content)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Error reescalando {file.filename}: {exc}")

        with open(original_path, "wb") as f:
            f.write(resized_bytes)

        try:
            hog_bytes, descriptores_hog = procesar_hog_con_descriptores(resized_bytes)
            
            # Si hay modelo de clustering, hacer predicción
            cluster_id = None
            ultimo_centroide = None
            if CLUSTER_MODEL_HOG and descriptores_hog:
                # HOG devuelve un vector de características directamente
                vector = np.array(descriptores_hog, dtype=float).reshape(1, -1)
                
                # Normalizar el vector con L2
                vector_normalizado = normalize(vector, norm='l2')[0]
                
                cluster_id, ultimo_centroide = CLUSTER_MODEL_HOG.predict_with_centroid(
                    vector_normalizado,
                    allow_new_clusters=True,
                )
                
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error procesando {file.filename}: {exc}")

        with open(processed_path, "wb") as f:
            f.write(hog_bytes)

        result = {
            "id": image_id,
            "filename": file.filename,
            "original_url": f"/files/originals/{original_name}",
            "processed_url": f"/files/processed/{processed_name}",
            "descriptores_hog": descriptores_hog,
            "num_features": len(descriptores_hog) if descriptores_hog else 0,
        }
        
        if cluster_id is not None:
            result["cluster_id"] = cluster_id
            result["ultimo_centroide"] = ultimo_centroide.tolist()
            print(f"[HOG-CLUSTER] id={image_id} cluster={cluster_id} features={len(descriptores_hog)} centroid={ultimo_centroide.tolist()}")
        
        results.append(result)

    # Guardar estado del modelo si se hizo clustering
    if CLUSTER_MODEL_HOG:
        save_cluster_state_hog()
        
        # Calcular métricas de evaluación
        dunn_index = CLUSTER_MODEL_HOG.calculate_dunn_index()
        silhouette_coefficient = CLUSTER_MODEL_HOG.calculate_silhouette_coefficient()
        
        return {
            "results": results,
            "metrics": {
                "dunn_index": round(float(dunn_index), 4),
                "silhouette_coefficient": round(float(silhouette_coefficient), 4),
            }
        }

    return {"results": results}


@app.post("/analyze-cnn")
async def analyze_images_cnn(
    files: List[UploadFile] = File(...),
    capacities: str = Form(None),
    clusters: int = Form(None)
):
    """
    Analiza imágenes con CNN (ResNet50) y realiza clustering usando los descriptores
    """
    global CLUSTER_MODEL_CNN, CLUSTER_CAPACITIES_CNN
    
    if not files:
        raise HTTPException(status_code=400, detail="No se enviaron archivos")

    # Inicializar el modelo de clustering si se proporcionan capacidades o clusters
    if capacities or clusters:
        try:
            caps = parse_capacities(capacities) if capacities else [10] * clusters
            CLUSTER_MODEL_CNN = LinksClusterCapacityOnline(capacities=caps)
            CLUSTER_CAPACITIES_CNN = caps
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Error inicializando clustering: {exc}")

    results = []

    for file in files:
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(status_code=400, detail=f"Tipo no permitido: {file.content_type}")

        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"Archivo demasiado grande: {file.filename}")

        ext = ALLOWED_TYPES[file.content_type]
        processed_ext = ".png"
        image_id = uuid.uuid4().hex
        original_name = f"{image_id}{ext}"
        processed_name = f"{image_id}_cnn{processed_ext}"

        original_path = os.path.join(ORIGINAL_DIR, original_name)
        processed_path = os.path.join(PROCESSED_DIR, processed_name)

        try:
            resized_bytes = reescalar_imagen_bytes(content)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Error reescalando {file.filename}: {exc}")

        with open(original_path, "wb") as f:
            f.write(resized_bytes)

        try:
            cnn_bytes, descriptores_cnn = procesar_cnn_con_descriptores(resized_bytes)
            
            # Si hay modelo de clustering, hacer predicción
            cluster_id = None
            ultimo_centroide = None
            if CLUSTER_MODEL_CNN and descriptores_cnn:
                # CNN devuelve un vector de características directamente
                vector = np.array(descriptores_cnn, dtype=float).reshape(1, -1)
                
                # Normalizar el vector con L2
                vector_normalizado = normalize(vector, norm='l2')[0]
                
                cluster_id, ultimo_centroide = CLUSTER_MODEL_CNN.predict_with_centroid(
                    vector_normalizado,
                    allow_new_clusters=True,
                )
                
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error procesando {file.filename}: {exc}")

        with open(processed_path, "wb") as f:
            f.write(cnn_bytes)

        result = {
            "id": image_id,
            "filename": file.filename,
            "original_url": f"/files/originals/{original_name}",
            "processed_url": f"/files/processed/{processed_name}",
            "descriptores_cnn": descriptores_cnn,
            "num_features": len(descriptores_cnn) if descriptores_cnn else 0,
        }
        
        if cluster_id is not None:
            result["cluster_id"] = cluster_id
            result["ultimo_centroide"] = ultimo_centroide.tolist()
            print(f"[CNN-CLUSTER] id={image_id} cluster={cluster_id} features={len(descriptores_cnn)} centroid={ultimo_centroide.tolist()}")
        
        results.append(result)

    # Guardar estado del modelo si se hizo clustering
    if CLUSTER_MODEL_CNN:
        save_cluster_state_cnn()
        
        # Calcular métricas de evaluación
        dunn_index = CLUSTER_MODEL_CNN.calculate_dunn_index()
        silhouette_coefficient = CLUSTER_MODEL_CNN.calculate_silhouette_coefficient()
        
        return {
            "results": results,
            "metrics": {
                "dunn_index": round(float(dunn_index), 4),
                "silhouette_coefficient": round(float(silhouette_coefficient), 4),
            }
        }

    return {"results": results}


@app.post("/analyze-hu")
async def analyze_images_hu(
    files: List[UploadFile] = File(...),
    capacities: str | None = Form(None),
    clusters: int | None = Form(None),
    reset: bool = Form(False),
):
    if not files:
        raise HTTPException(status_code=400, detail="No se enviaron archivos")

    # Si reset=True o si se proporcionan nuevas capacidades, reiniciar el modelo
    if reset or capacities or clusters:
        global CLUSTER_MODEL_HU
        CLUSTER_MODEL_HU = None

    results = []

    if capacities is None and clusters is None:
        raise HTTPException(
            status_code=400,
            detail="Debes indicar capacities o número de clusters",
        )

    if capacities:
        try:
            caps = parse_capacities(capacities)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"capacities inválido: {exc}")
    else:
        try:
            k = int(clusters) if clusters is not None else 0
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"clusters inválido: {exc}")
        if k <= 0:
            raise HTTPException(status_code=400, detail="clusters inválido")
        caps = [100] * k

    # Crear/obtener modelo para Hu
    global CLUSTER_CAPACITIES_HU
    if CLUSTER_MODEL_HU is None or CLUSTER_CAPACITIES_HU != caps:
        CLUSTER_MODEL_HU = LinksClusterCapacityOnline(capacities=caps)
        CLUSTER_CAPACITIES_HU = caps
    
    model = CLUSTER_MODEL_HU

    for file in files:
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(status_code=400, detail=f"Tipo no permitido: {file.content_type}")

        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"Archivo demasiado grande: {file.filename}")

        ext = ALLOWED_TYPES[file.content_type]
        processed_ext = ".png"
        image_id = uuid.uuid4().hex
        original_name = f"{image_id}{ext}"
        processed_name = f"{image_id}_processed{processed_ext}"
        binarized_name = f"{image_id}_binarized{processed_ext}"

        original_path = os.path.join(ORIGINAL_DIR, original_name)
        processed_path = os.path.join(PROCESSED_DIR, processed_name)
        binarized_path = os.path.join(BINARIZED_DIR, binarized_name)

        try:
            resized_bytes = reescalar_imagen_bytes(content)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Error reescalando {file.filename}: {exc}")

        with open(original_path, "wb") as f:
            f.write(resized_bytes)

        try:
            processed_bytes = procesar_imagen_bytes(resized_bytes)
            binarized_bytes = binarizar_imagen_bytes(resized_bytes)
            momentos_hu = calcular_momentos_hu(resized_bytes)
            
            # Normalizar el vector de Hu moments (7 valores: hu1 a hu7)
            vector = np.array([float(momentos_hu[f"hu{i}"]) for i in range(1, 8)], dtype=float).reshape(1, -1)
            vector_normalizado = normalize(vector, norm='l2')[0]
            
            cluster_id, last_centroid = model.predict_with_centroid(vector_normalizado)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error procesando {file.filename}: {exc}")

        with open(processed_path, "wb") as f:
            f.write(processed_bytes)

        with open(binarized_path, "wb") as f:
            f.write(binarized_bytes)

        result = {
            "id": image_id,
            "filename": file.filename,
            "original_url": f"/files/originals/{original_name}",
            "processed_url": f"/files/processed/{processed_name}",
            "binarized_url": f"/files/binarized/{binarized_name}",
            "momentos_hu": momentos_hu,
            "cluster_id": cluster_id,
            "ultimo_centroide": last_centroid.tolist(),
        }
        results.append(result)

        print(f"[CLUSTER-HU] id={image_id} cluster={cluster_id} centroid={last_centroid.tolist()}")

    # Guardar estado del modelo después de procesar todas las imágenes
    save_cluster_state_hu()
    
    # Calcular métricas de evaluación
    dunn_index = model.calculate_dunn_index()
    silhouette_coefficient = model.calculate_silhouette_coefficient()
    
    return {
        "results": results,
        "metrics": {
            "dunn_index": round(float(dunn_index), 4),
            "silhouette_coefficient": round(float(silhouette_coefficient), 4),
        }
    }


@app.post("/add-images-hu")
async def add_images_hu(files: List[UploadFile] = File(...)):
    """Agrega nuevas imágenes al clustering Hu existente"""
    global CLUSTER_MODEL_HU, CLUSTER_CAPACITIES_HU
    
    # Si el modelo está None, intentar restaurarlo desde el archivo guardado
    if CLUSTER_MODEL_HU is None:
        if os.path.exists(CLUSTER_STATE_FILE_HU):
            try:
                with open(CLUSTER_STATE_FILE_HU, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    CLUSTER_MODEL_HU = LinksClusterCapacityOnline.from_dict(data)
                    CLUSTER_CAPACITIES_HU = CLUSTER_MODEL_HU.capacities
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"No hay modelo de clustering activo. Usa /analyze-hu primero"
                )
        else:
            raise HTTPException(
                status_code=400, 
                detail="No hay modelo de clustering activo. Usa /analyze-hu primero"
            )
    
    if not files:
        raise HTTPException(status_code=400, detail="No se enviaron archivos")

    results = []

    for file in files:
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(status_code=400, detail=f"Tipo no permitido: {file.content_type}")

        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"Archivo demasiado grande: {file.filename}")

        ext = ALLOWED_TYPES[file.content_type]
        processed_ext = ".png"
        image_id = uuid.uuid4().hex
        original_name = f"{image_id}{ext}"
        processed_name = f"{image_id}_processed{processed_ext}"
        binarized_name = f"{image_id}_binarized{processed_ext}"

        original_path = os.path.join(ORIGINAL_DIR, original_name)
        processed_path = os.path.join(PROCESSED_DIR, processed_name)
        binarized_path = os.path.join(BINARIZED_DIR, binarized_name)

        try:
            resized_bytes = reescalar_imagen_bytes(content)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Error reescalando {file.filename}: {exc}")

        with open(original_path, "wb") as f:
            f.write(resized_bytes)

        try:
            processed_bytes = procesar_imagen_bytes(resized_bytes)
            binarized_bytes = binarizar_imagen_bytes(resized_bytes)
            momentos_hu = calcular_momentos_hu(resized_bytes)
            
            vector = np.array([float(momentos_hu[f"hu{i}"]) for i in range(1, 8)], dtype=float).reshape(1, -1)
            vector_normalizado = normalize(vector, norm='l2')[0]
            
            cluster_id, last_centroid = CLUSTER_MODEL_HU.predict_with_centroid(
                vector_normalizado,
                allow_new_clusters=False,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error procesando {file.filename}: {exc}")

        with open(processed_path, "wb") as f:
            f.write(processed_bytes)

        with open(binarized_path, "wb") as f:
            f.write(binarized_bytes)

        result = {
            "id": image_id,
            "filename": file.filename,
            "original_url": f"/files/originals/{original_name}",
            "processed_url": f"/files/processed/{processed_name}",
            "binarized_url": f"/files/binarized/{binarized_name}",
            "momentos_hu": momentos_hu,
            "cluster_id": cluster_id,
            "ultimo_centroide": last_centroid.tolist(),
        }
        results.append(result)

        print(f"[ADD-CLUSTER-HU] id={image_id} cluster={cluster_id} centroid={last_centroid.tolist()}")

    # Guardar estado del modelo después de agregar imágenes
    save_cluster_state_hu()
    
    # Calcular métricas de evaluación
    dunn_index = CLUSTER_MODEL_HU.calculate_dunn_index()
    silhouette_coefficient = CLUSTER_MODEL_HU.calculate_silhouette_coefficient()
    
    return {
        "results": results,
        "metrics": {
            "dunn_index": round(float(dunn_index), 4),
            "silhouette_coefficient": round(float(silhouette_coefficient), 4),
        }
    }


@app.post("/update-capacities-hu")
async def update_capacities_hu(capacities: str = Form(...)):
    """Actualiza las capacidades de los clusters Hu existentes"""
    global CLUSTER_MODEL_HU, CLUSTER_CAPACITIES_HU
    
    # Si el modelo está None, intentar restaurarlo desde el archivo
    if CLUSTER_MODEL_HU is None:
        if os.path.exists(CLUSTER_STATE_FILE_HU):
            try:
                with open(CLUSTER_STATE_FILE_HU, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    CLUSTER_MODEL_HU = LinksClusterCapacityOnline.from_dict(data)
                    CLUSTER_CAPACITIES_HU = CLUSTER_MODEL_HU.capacities
            except:
                raise HTTPException(status_code=400, detail="No hay modelo activo")
        else:
            raise HTTPException(status_code=400, detail="No hay modelo activo. Ejecuta /analyze-hu primero")
    
    try:
        new_caps = parse_capacities(capacities)
        if len(new_caps) != len(CLUSTER_MODEL_HU.capacities):
            raise HTTPException(
                status_code=400,
                detail=f"Debe proporcionar {len(CLUSTER_MODEL_HU.capacities)} capacidades (tienes {len(CLUSTER_MODEL_HU.clusters)} clusters)"
            )
        
        CLUSTER_MODEL_HU.capacities = new_caps
        save_cluster_state_hu()
        
        # Calcular métricas de evaluación
        dunn_index = CLUSTER_MODEL_HU.calculate_dunn_index()
        silhouette_coefficient = CLUSTER_MODEL_HU.calculate_silhouette_coefficient()
        
        return {
            "status": "ok",
            "new_capacities": new_caps,
            "current_counts": CLUSTER_MODEL_HU.cluster_counts,
            "available_spaces": [
                cap - count 
                for cap, count in zip(new_caps, CLUSTER_MODEL_HU.cluster_counts)
            ],
            "metrics": {
                "dunn_index": round(float(dunn_index), 4),
                "silhouette_coefficient": round(float(silhouette_coefficient), 4),
            }
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/cluster-status-hu")
def get_cluster_status_hu():
    """Retorna el estado actual del clustering Hu"""
    global CLUSTER_MODEL_HU
    
    # Si el modelo está None, intentar restaurarlo desde el archivo
    if CLUSTER_MODEL_HU is None:
        if os.path.exists(CLUSTER_STATE_FILE_HU):
            try:
                with open(CLUSTER_STATE_FILE_HU, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    CLUSTER_MODEL_HU = LinksClusterCapacityOnline.from_dict(data)
            except:
                return {"active": False}
        else:
            return {"active": False}
    
    if not CLUSTER_MODEL_HU:
        return {"active": False}
    
    return {
        "active": True,
        "num_clusters": len(CLUSTER_MODEL_HU.clusters),
        "capacities": CLUSTER_MODEL_HU.capacities,
        "current_counts": CLUSTER_MODEL_HU.cluster_counts,
        "available_spaces": [
            cap - count 
            for cap, count in zip(CLUSTER_MODEL_HU.capacities, CLUSTER_MODEL_HU.cluster_counts)
        ]
    }


@app.post("/add-images-zernike")
async def add_images_zernike(files: List[UploadFile] = File(...)):
    """Agrega nuevas imágenes al clustering existente de Zernike"""
    global CLUSTER_MODEL_ZERNIKE, CLUSTER_CAPACITIES_ZERNIKE
    
    # Si el modelo está None, intentar restaurarlo desde el archivo guardado
    if CLUSTER_MODEL_ZERNIKE is None:
        if os.path.exists(CLUSTER_STATE_FILE_ZERNIKE):
            try:
                with open(CLUSTER_STATE_FILE_ZERNIKE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    CLUSTER_MODEL_ZERNIKE = LinksClusterCapacityOnline.from_dict(data)
                    CLUSTER_CAPACITIES_ZERNIKE = CLUSTER_MODEL_ZERNIKE.capacities
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"No hay modelo de clustering activo. Usa /analyze-zernike primero"
                )
        else:
            raise HTTPException(
                status_code=400, 
                detail="No hay modelo de clustering activo. Usa /analyze-zernike primero"
            )
    
    if not files:
        raise HTTPException(status_code=400, detail="No se enviaron archivos")

    results = []
    zernike_keys = [f"z{i}" for i in range(1, 26)]  # z1 a z25

    for file in files:
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(status_code=400, detail=f"Tipo no permitido: {file.content_type}")

        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"Archivo demasiado grande: {file.filename}")

        ext = ALLOWED_TYPES[file.content_type]
        processed_ext = ".png"
        image_id = uuid.uuid4().hex
        original_name = f"{image_id}{ext}"
        processed_name = f"{image_id}_processed{processed_ext}"
        binarized_name = f"{image_id}_binarized{processed_ext}"

        original_path = os.path.join(ORIGINAL_DIR, original_name)
        processed_path = os.path.join(PROCESSED_DIR, processed_name)
        binarized_path = os.path.join(BINARIZED_DIR, binarized_name)

        try:
            resized_bytes = reescalar_imagen_bytes(content)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Error reescalando {file.filename}: {exc}")

        with open(original_path, "wb") as f:
            f.write(resized_bytes)

        try:
            processed_bytes = procesar_imagen_bytes(resized_bytes)
            binarized_bytes = binarizar_imagen_bytes(resized_bytes)
            momentos_zernike = calcular_momentos_zernike(resized_bytes)
            vector = [float(momentos_zernike[k]) for k in zernike_keys]
            vector_array = np.array(vector, dtype=float).reshape(1, -1)
            
            # Normalizar el vector con L2
            vector_normalizado = normalize(vector_array, norm='l2')[0]
            
            cluster_id, last_centroid = CLUSTER_MODEL_ZERNIKE.predict_with_centroid(
                vector_normalizado,
                allow_new_clusters=False,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error procesando {file.filename}: {exc}")

        with open(processed_path, "wb") as f:
            f.write(processed_bytes)

        with open(binarized_path, "wb") as f:
            f.write(binarized_bytes)

        result = {
            "id": image_id,
            "filename": file.filename,
            "original_url": f"/files/originals/{original_name}",
            "processed_url": f"/files/processed/{processed_name}",
            "binarized_url": f"/files/binarized/{binarized_name}",
            "momentos_zernike": momentos_zernike,
            "cluster_id": cluster_id,
            "ultimo_centroide": last_centroid.tolist(),
        }
        results.append(result)

        print(f"[ADD-ZERNIKE-CLUSTER] id={image_id} cluster={cluster_id} centroid={last_centroid.tolist()}")

    # Guardar estado del modelo después de agregar imágenes
    save_cluster_state_zernike()
    
    # Calcular métricas de evaluación
    dunn_index = CLUSTER_MODEL_ZERNIKE.calculate_dunn_index()
    silhouette_coefficient = CLUSTER_MODEL_ZERNIKE.calculate_silhouette_coefficient()
    
    return {
        "results": results,
        "metrics": {
            "dunn_index": round(float(dunn_index), 4),
            "silhouette_coefficient": round(float(silhouette_coefficient), 4),
        }
    }


@app.post("/update-capacities-zernike")
async def update_capacities_zernike(capacities: str = Form(...)):
    """Actualiza las capacidades de los clusters existentes de Zernike"""
    global CLUSTER_MODEL_ZERNIKE, CLUSTER_CAPACITIES_ZERNIKE
    
    # Si el modelo está None, intentar restaurarlo desde el archivo
    if CLUSTER_MODEL_ZERNIKE is None:
        if os.path.exists(CLUSTER_STATE_FILE_ZERNIKE):
            try:
                with open(CLUSTER_STATE_FILE_ZERNIKE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    CLUSTER_MODEL_ZERNIKE = LinksClusterCapacityOnline.from_dict(data)
                    CLUSTER_CAPACITIES_ZERNIKE = CLUSTER_MODEL_ZERNIKE.capacities
            except:
                raise HTTPException(status_code=400, detail="No hay modelo activo")
        else:
            raise HTTPException(status_code=400, detail="No hay modelo activo. Ejecuta /analyze-zernike primero")
    
    try:
        new_caps = parse_capacities(capacities)
        if len(new_caps) != len(CLUSTER_MODEL_ZERNIKE.capacities):
            raise HTTPException(
                status_code=400,
                detail=f"Debe proporcionar {len(CLUSTER_MODEL_ZERNIKE.capacities)} capacidades (tienes {len(CLUSTER_MODEL_ZERNIKE.clusters)} clusters)"
            )
        
        CLUSTER_MODEL_ZERNIKE.capacities = new_caps
        save_cluster_state_zernike()
        
        # Calcular métricas de evaluación
        dunn_index = CLUSTER_MODEL_ZERNIKE.calculate_dunn_index()
        silhouette_coefficient = CLUSTER_MODEL_ZERNIKE.calculate_silhouette_coefficient()
        
        return {
            "status": "ok",
            "new_capacities": new_caps,
            "current_counts": CLUSTER_MODEL_ZERNIKE.cluster_counts,
            "available_spaces": [
                cap - count 
                for cap, count in zip(new_caps, CLUSTER_MODEL_ZERNIKE.cluster_counts)
            ],
            "metrics": {
                "dunn_index": round(float(dunn_index), 4),
                "silhouette_coefficient": round(float(silhouette_coefficient), 4),
            }
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/cluster-status-zernike")
def get_cluster_status_zernike():
    """Retorna el estado actual del clustering Zernike"""
    global CLUSTER_MODEL_ZERNIKE
    
    # Si el modelo está None, intentar restaurarlo desde el archivo
    if CLUSTER_MODEL_ZERNIKE is None:
        if os.path.exists(CLUSTER_STATE_FILE_ZERNIKE):
            try:
                with open(CLUSTER_STATE_FILE_ZERNIKE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    CLUSTER_MODEL_ZERNIKE = LinksClusterCapacityOnline.from_dict(data)
            except:
                return {"active": False}
        else:
            return {"active": False}
    
    if not CLUSTER_MODEL_ZERNIKE:
        return {"active": False}
    
    return {
        "active": True,
        "num_clusters": len(CLUSTER_MODEL_ZERNIKE.clusters),
        "capacities": CLUSTER_MODEL_ZERNIKE.capacities,
        "current_counts": CLUSTER_MODEL_ZERNIKE.cluster_counts,
        "available_spaces": [
            cap - count 
            for cap, count in zip(CLUSTER_MODEL_ZERNIKE.capacities, CLUSTER_MODEL_ZERNIKE.cluster_counts)
        ]
    }


@app.post("/add-images-sift")
async def add_images_sift(files: List[UploadFile] = File(...)):
    """Agrega nuevas imágenes al clustering existente de SIFT"""
    global CLUSTER_MODEL_SIFT, CLUSTER_CAPACITIES_SIFT
    
    # Si el modelo está None, intentar restaurarlo desde el archivo guardado
    if CLUSTER_MODEL_SIFT is None:
        if os.path.exists(CLUSTER_STATE_FILE_SIFT):
            try:
                with open(CLUSTER_STATE_FILE_SIFT, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    CLUSTER_MODEL_SIFT = LinksClusterCapacityOnline.from_dict(data)
                    CLUSTER_CAPACITIES_SIFT = CLUSTER_MODEL_SIFT.capacities
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"No hay modelo de clustering activo. Usa /analyze-sift primero"
                )
        else:
            raise HTTPException(
                status_code=400, 
                detail="No hay modelo de clustering activo. Usa /analyze-sift primero"
            )
    
    if not files:
        raise HTTPException(status_code=400, detail="No se enviaron archivos")

    results = []

    for file in files:
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(status_code=400, detail=f"Tipo no permitido: {file.content_type}")

        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"Archivo demasiado grande: {file.filename}")

        ext = ALLOWED_TYPES[file.content_type]
        processed_ext = ".png"
        image_id = uuid.uuid4().hex
        original_name = f"{image_id}{ext}"
        processed_name = f"{image_id}_sift{processed_ext}"

        original_path = os.path.join(ORIGINAL_DIR, original_name)
        processed_path = os.path.join(PROCESSED_DIR, processed_name)

        try:
            resized_bytes = reescalar_imagen_bytes(content)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Error reescalando {file.filename}: {exc}")

        with open(original_path, "wb") as f:
            f.write(resized_bytes)

        try:
            sift_bytes, descriptores = procesar_sift_con_descriptores(resized_bytes)
            
            if not descriptores:
                raise ValueError("No se encontraron keypoints SIFT en la imagen")
            
            # Usar el promedio de todos los descriptores como vector característico
            descriptores_array = np.array(descriptores, dtype=float)
            vector = np.mean(descriptores_array, axis=0)
            vector = vector.reshape(1, -1)
            
            # Normalizar el vector con L2
            vector_normalizado = normalize(vector, norm='l2')[0]
            
            cluster_id, last_centroid = CLUSTER_MODEL_SIFT.predict_with_centroid(
                vector_normalizado,
                allow_new_clusters=False,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error procesando {file.filename}: {exc}")

        with open(processed_path, "wb") as f:
            f.write(sift_bytes)

        result = {
            "id": image_id,
            "filename": file.filename,
            "original_url": f"/files/originals/{original_name}",
            "processed_url": f"/files/processed/{processed_name}",
            "descriptores": descriptores,
            "num_keypoints": len(descriptores),
            "cluster_id": cluster_id,
            "ultimo_centroide": last_centroid.tolist(),
        }
        results.append(result)

        print(f"[ADD-SIFT-CLUSTER] id={image_id} cluster={cluster_id} keypoints={len(descriptores)} centroid={last_centroid.tolist()}")

    # Guardar estado del modelo después de agregar imágenes
    save_cluster_state_sift()
    
    # Calcular métricas de evaluación
    dunn_index = CLUSTER_MODEL_SIFT.calculate_dunn_index()
    silhouette_coefficient = CLUSTER_MODEL_SIFT.calculate_silhouette_coefficient()
    
    return {
        "results": results,
        "metrics": {
            "dunn_index": round(float(dunn_index), 4),
            "silhouette_coefficient": round(float(silhouette_coefficient), 4),
        }
    }


@app.post("/update-capacities-sift")
async def update_capacities_sift(capacities: str = Form(...)):
    """Actualiza las capacidades de los clusters existentes de SIFT"""
    global CLUSTER_MODEL_SIFT, CLUSTER_CAPACITIES_SIFT
    
    # Si el modelo está None, intentar restaurarlo desde el archivo
    if CLUSTER_MODEL_SIFT is None:
        if os.path.exists(CLUSTER_STATE_FILE_SIFT):
            try:
                with open(CLUSTER_STATE_FILE_SIFT, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    CLUSTER_MODEL_SIFT = LinksClusterCapacityOnline.from_dict(data)
                    CLUSTER_CAPACITIES_SIFT = CLUSTER_MODEL_SIFT.capacities
            except:
                raise HTTPException(status_code=400, detail="No hay modelo activo")
        else:
            raise HTTPException(status_code=400, detail="No hay modelo activo. Ejecuta /analyze-sift primero")
    
    try:
        new_caps = parse_capacities(capacities)
        if len(new_caps) != len(CLUSTER_MODEL_SIFT.capacities):
            raise HTTPException(
                status_code=400,
                detail=f"Debe proporcionar {len(CLUSTER_MODEL_SIFT.capacities)} capacidades (tienes {len(CLUSTER_MODEL_SIFT.clusters)} clusters)"
            )
        
        CLUSTER_MODEL_SIFT.capacities = new_caps
        save_cluster_state_sift()
        
        # Calcular métricas de evaluación
        dunn_index = CLUSTER_MODEL_SIFT.calculate_dunn_index()
        silhouette_coefficient = CLUSTER_MODEL_SIFT.calculate_silhouette_coefficient()
        
        return {
            "status": "ok",
            "new_capacities": new_caps,
            "current_counts": CLUSTER_MODEL_SIFT.cluster_counts,
            "available_spaces": [
                cap - count 
                for cap, count in zip(new_caps, CLUSTER_MODEL_SIFT.cluster_counts)
            ],
            "metrics": {
                "dunn_index": round(float(dunn_index), 4),
                "silhouette_coefficient": round(float(silhouette_coefficient), 4),
            }
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/cluster-status-sift")
def get_cluster_status_sift():
    """Retorna el estado actual del clustering SIFT"""
    global CLUSTER_MODEL_SIFT
    
    # Si el modelo está None, intentar restaurarlo desde el archivo
    if CLUSTER_MODEL_SIFT is None:
        if os.path.exists(CLUSTER_STATE_FILE_SIFT):
            try:
                with open(CLUSTER_STATE_FILE_SIFT, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    CLUSTER_MODEL_SIFT = LinksClusterCapacityOnline.from_dict(data)
            except:
                return {"active": False}
        else:
            return {"active": False}
    
    if not CLUSTER_MODEL_SIFT:
        return {"active": False}
    
    return {
        "active": True,
        "num_clusters": len(CLUSTER_MODEL_SIFT.clusters),
        "capacities": CLUSTER_MODEL_SIFT.capacities,
        "current_counts": CLUSTER_MODEL_SIFT.cluster_counts,
        "available_spaces": [
            cap - count 
            for cap, count in zip(CLUSTER_MODEL_SIFT.capacities, CLUSTER_MODEL_SIFT.cluster_counts)
        ]
    }


@app.post("/add-images-hog")
async def add_images_hog(files: List[UploadFile] = File(...)):
    """Agrega nuevas imágenes al clustering existente de HOG"""
    global CLUSTER_MODEL_HOG, CLUSTER_CAPACITIES_HOG
    
    # Si el modelo está None, intentar restaurarlo desde el archivo guardado
    if CLUSTER_MODEL_HOG is None:
        if os.path.exists(CLUSTER_STATE_FILE_HOG):
            try:
                with open(CLUSTER_STATE_FILE_HOG, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    CLUSTER_MODEL_HOG = LinksClusterCapacityOnline.from_dict(data)
                    CLUSTER_CAPACITIES_HOG = CLUSTER_MODEL_HOG.capacities
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"No hay modelo de clustering activo. Usa /analyze-hog primero"
                )
        else:
            raise HTTPException(
                status_code=400, 
                detail="No hay modelo de clustering activo. Usa /analyze-hog primero"
            )
    
    if not files:
        raise HTTPException(status_code=400, detail="No se enviaron archivos")

    results = []

    for file in files:
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(status_code=400, detail=f"Tipo no permitido: {file.content_type}")

        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"Archivo demasiado grande: {file.filename}")

        ext = ALLOWED_TYPES[file.content_type]
        processed_ext = ".png"
        image_id = uuid.uuid4().hex
        original_name = f"{image_id}{ext}"
        processed_name = f"{image_id}_hog{processed_ext}"

        original_path = os.path.join(ORIGINAL_DIR, original_name)
        processed_path = os.path.join(PROCESSED_DIR, processed_name)

        try:
            resized_bytes = reescalar_imagen_bytes(content)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Error reescalando {file.filename}: {exc}")

        with open(original_path, "wb") as f:
            f.write(resized_bytes)

        try:
            hog_bytes, descriptores_hog = procesar_hog_con_descriptores(resized_bytes)
            
            if not descriptores_hog:
                raise ValueError("No se pudieron extraer características HOG de la imagen")
            
            # HOG devuelve un vector de características directamente
            vector = np.array(descriptores_hog, dtype=float).reshape(1, -1)
            
            # Normalizar el vector con L2
            vector_normalizado = normalize(vector, norm='l2')[0]
            
            cluster_id, last_centroid = CLUSTER_MODEL_HOG.predict_with_centroid(
                vector_normalizado,
                allow_new_clusters=False,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error procesando {file.filename}: {exc}")

        with open(processed_path, "wb") as f:
            f.write(hog_bytes)

        result = {
            "id": image_id,
            "filename": file.filename,
            "original_url": f"/files/originals/{original_name}",
            "processed_url": f"/files/processed/{processed_name}",
            "descriptores_hog": descriptores_hog,
            "num_features": len(descriptores_hog),
            "cluster_id": cluster_id,
            "ultimo_centroide": last_centroid.tolist(),
        }
        results.append(result)

        print(f"[ADD-HOG-CLUSTER] id={image_id} cluster={cluster_id} features={len(descriptores_hog)} centroid={last_centroid.tolist()}")

    # Guardar estado del modelo después de agregar imágenes
    save_cluster_state_hog()
    
    # Calcular métricas de evaluación
    dunn_index = CLUSTER_MODEL_HOG.calculate_dunn_index()
    silhouette_coefficient = CLUSTER_MODEL_HOG.calculate_silhouette_coefficient()
    
    return {
        "results": results,
        "metrics": {
            "dunn_index": round(float(dunn_index), 4),
            "silhouette_coefficient": round(float(silhouette_coefficient), 4),
        }
    }


@app.post("/update-capacities-hog")
async def update_capacities_hog(capacities: str = Form(...)):
    """Actualiza las capacidades de los clusters existentes de HOG"""
    global CLUSTER_MODEL_HOG, CLUSTER_CAPACITIES_HOG
    
    # Si el modelo está None, intentar restaurarlo desde el archivo
    if CLUSTER_MODEL_HOG is None:
        if os.path.exists(CLUSTER_STATE_FILE_HOG):
            try:
                with open(CLUSTER_STATE_FILE_HOG, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    CLUSTER_MODEL_HOG = LinksClusterCapacityOnline.from_dict(data)
                    CLUSTER_CAPACITIES_HOG = CLUSTER_MODEL_HOG.capacities
            except:
                raise HTTPException(status_code=400, detail="No hay modelo activo")
        else:
            raise HTTPException(status_code=400, detail="No hay modelo activo. Ejecuta /analyze-hog primero")
    
    try:
        new_caps = parse_capacities(capacities)
        if len(new_caps) != len(CLUSTER_MODEL_HOG.capacities):
            raise HTTPException(
                status_code=400,
                detail=f"Debe proporcionar {len(CLUSTER_MODEL_HOG.capacities)} capacidades (tienes {len(CLUSTER_MODEL_HOG.clusters)} clusters)"
            )
        
        CLUSTER_MODEL_HOG.capacities = new_caps
        save_cluster_state_hog()
        
        # Calcular métricas de evaluación
        dunn_index = CLUSTER_MODEL_HOG.calculate_dunn_index()
        silhouette_coefficient = CLUSTER_MODEL_HOG.calculate_silhouette_coefficient()
        
        return {
            "status": "ok",
            "new_capacities": new_caps,
            "current_counts": CLUSTER_MODEL_HOG.cluster_counts,
            "available_spaces": [
                cap - count 
                for cap, count in zip(new_caps, CLUSTER_MODEL_HOG.cluster_counts)
            ],
            "metrics": {
                "dunn_index": round(float(dunn_index), 4),
                "silhouette_coefficient": round(float(silhouette_coefficient), 4),
            }
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/cluster-status-hog")
def get_cluster_status_hog():
    """Retorna el estado actual del clustering HOG"""
    global CLUSTER_MODEL_HOG
    
    # Si el modelo está None, intentar restaurarlo desde el archivo
    if CLUSTER_MODEL_HOG is None:
        if os.path.exists(CLUSTER_STATE_FILE_HOG):
            try:
                with open(CLUSTER_STATE_FILE_HOG, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    CLUSTER_MODEL_HOG = LinksClusterCapacityOnline.from_dict(data)
            except:
                return {"active": False}
        else:
            return {"active": False}
    
    if not CLUSTER_MODEL_HOG:
        return {"active": False}
    
    return {
        "active": True,
        "num_clusters": len(CLUSTER_MODEL_HOG.clusters),
        "capacities": CLUSTER_MODEL_HOG.capacities,
        "current_counts": CLUSTER_MODEL_HOG.cluster_counts,
        "available_spaces": [
            cap - count 
            for cap, count in zip(CLUSTER_MODEL_HOG.capacities, CLUSTER_MODEL_HOG.cluster_counts)
        ]
    }


@app.post("/add-images-cnn")
async def add_images_cnn(files: List[UploadFile] = File(...)):
    """Agrega nuevas imágenes al clustering existente de CNN"""
    global CLUSTER_MODEL_CNN, CLUSTER_CAPACITIES_CNN
    
    # Si el modelo está None, intentar restaurarlo desde el archivo guardado
    if CLUSTER_MODEL_CNN is None:
        if os.path.exists(CLUSTER_STATE_FILE_CNN):
            try:
                with open(CLUSTER_STATE_FILE_CNN, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    CLUSTER_MODEL_CNN = LinksClusterCapacityOnline.from_dict(data)
                    CLUSTER_CAPACITIES_CNN = CLUSTER_MODEL_CNN.capacities
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error cargando modelo: {e}")
        else:
            raise HTTPException(status_code=400, detail="No hay modelo activo. Ejecuta /analyze-cnn primero")
    
    if not files:
        raise HTTPException(status_code=400, detail="No se enviaron archivos")

    results = []

    for file in files:
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(status_code=400, detail=f"Tipo no permitido: {file.content_type}")

        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"Archivo demasiado grande: {file.filename}")

        ext = ALLOWED_TYPES[file.content_type]
        processed_ext = ".png"
        image_id = uuid.uuid4().hex
        original_name = f"{image_id}{ext}"
        processed_name = f"{image_id}_cnn{processed_ext}"

        original_path = os.path.join(ORIGINAL_DIR, original_name)
        processed_path = os.path.join(PROCESSED_DIR, processed_name)

        try:
            resized_bytes = reescalar_imagen_bytes(content)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Error reescalando {file.filename}: {exc}")

        with open(original_path, "wb") as f:
            f.write(resized_bytes)

        try:
            cnn_bytes, descriptores_cnn = procesar_cnn_con_descriptores(resized_bytes)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error procesando {file.filename}: {exc}")

        with open(processed_path, "wb") as f:
            f.write(cnn_bytes)

        # Clustering: predecir cluster
        vector = np.array(descriptores_cnn, dtype=float).reshape(1, -1)
        vector_normalizado = normalize(vector, norm='l2')[0]
        
        cluster_id, last_centroid = CLUSTER_MODEL_CNN.predict_with_centroid(
            vector_normalizado,
            allow_new_clusters=False
        )

        result = {
            "id": image_id,
            "filename": file.filename,
            "original_url": f"/files/originals/{original_name}",
            "processed_url": f"/files/processed/{processed_name}",
            "descriptores_cnn": descriptores_cnn,
            "num_features": len(descriptores_cnn) if descriptores_cnn else 0,
            "cluster_id": cluster_id,
            "ultimo_centroide": last_centroid.tolist(),
        }
        
        results.append(result)
        print(f"[ADD-CNN-CLUSTER] id={image_id} cluster={cluster_id} features={len(descriptores_cnn)} centroid={last_centroid.tolist()}")

    # Guardar estado del modelo después de agregar imágenes
    save_cluster_state_cnn()
    
    # Calcular métricas de evaluación
    dunn_index = CLUSTER_MODEL_CNN.calculate_dunn_index()
    silhouette_coefficient = CLUSTER_MODEL_CNN.calculate_silhouette_coefficient()
    
    return {
        "results": results,
        "metrics": {
            "dunn_index": round(float(dunn_index), 4),
            "silhouette_coefficient": round(float(silhouette_coefficient), 4),
        }
    }


@app.post("/update-capacities-cnn")
async def update_capacities_cnn(capacities: str = Form(...)):
    """Actualiza las capacidades de los clusters existentes de CNN"""
    global CLUSTER_MODEL_CNN, CLUSTER_CAPACITIES_CNN
    
    # Si el modelo está None, intentar restaurarlo desde el archivo
    if CLUSTER_MODEL_CNN is None:
        if os.path.exists(CLUSTER_STATE_FILE_CNN):
            try:
                with open(CLUSTER_STATE_FILE_CNN, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    CLUSTER_MODEL_CNN = LinksClusterCapacityOnline.from_dict(data)
                    CLUSTER_CAPACITIES_CNN = CLUSTER_MODEL_CNN.capacities
            except:
                raise HTTPException(status_code=400, detail="No hay modelo activo")
        else:
            raise HTTPException(status_code=400, detail="No hay modelo activo. Ejecuta /analyze-cnn primero")
    
    try:
        new_caps = parse_capacities(capacities)
        if len(new_caps) != len(CLUSTER_MODEL_CNN.capacities):
            raise HTTPException(
                status_code=400,
                detail=f"Debe proporcionar {len(CLUSTER_MODEL_CNN.capacities)} capacidades (tienes {len(CLUSTER_MODEL_CNN.clusters)} clusters)"
            )
        
        CLUSTER_MODEL_CNN.capacities = new_caps
        save_cluster_state_cnn()
        
        # Calcular métricas de evaluación
        dunn_index = CLUSTER_MODEL_CNN.calculate_dunn_index()
        silhouette_coefficient = CLUSTER_MODEL_CNN.calculate_silhouette_coefficient()
        
        return {
            "status": "ok",
            "new_capacities": new_caps,
            "current_counts": CLUSTER_MODEL_CNN.cluster_counts,
            "available_spaces": [
                cap - count 
                for cap, count in zip(new_caps, CLUSTER_MODEL_CNN.cluster_counts)
            ],
            "metrics": {
                "dunn_index": round(float(dunn_index), 4),
                "silhouette_coefficient": round(float(silhouette_coefficient), 4),
            }
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/cluster-status-cnn")
def get_cluster_status_cnn():
    """Retorna el estado actual del clustering CNN"""
    global CLUSTER_MODEL_CNN
    
    # Si el modelo está None, intentar restaurarlo desde el archivo
    if CLUSTER_MODEL_CNN is None:
        if os.path.exists(CLUSTER_STATE_FILE_CNN):
            try:
                with open(CLUSTER_STATE_FILE_CNN, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    CLUSTER_MODEL_CNN = LinksClusterCapacityOnline.from_dict(data)
            except:
                return {"active": False}
        else:
            return {"active": False}
    
    if not CLUSTER_MODEL_CNN:
        return {"active": False}
    
    return {
        "active": True,
        "num_clusters": len(CLUSTER_MODEL_CNN.clusters),
        "capacities": CLUSTER_MODEL_CNN.capacities,
        "current_counts": CLUSTER_MODEL_CNN.cluster_counts,
        "available_spaces": [
            cap - count 
            for cap, count in zip(CLUSTER_MODEL_CNN.capacities, CLUSTER_MODEL_CNN.cluster_counts)
        ]
    }


@app.post("/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No se enviaron archivos")

    items = load_index()
    new_items = []

    for file in files:
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(status_code=400, detail=f"Tipo no permitido: {file.content_type}")

        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"Archivo demasiado grande: {file.filename}")

        ext = ALLOWED_TYPES[file.content_type]
        processed_ext = ".png"
        image_id = uuid.uuid4().hex
        original_name = f"{image_id}{ext}"
        processed_name = f"{image_id}_processed{processed_ext}"
        binarized_name = f"{image_id}_binarized{processed_ext}"

        original_path = os.path.join(ORIGINAL_DIR, original_name)
        processed_path = os.path.join(PROCESSED_DIR, processed_name)
        binarized_path = os.path.join(BINARIZED_DIR, binarized_name)

        try:
            resized_bytes = reescalar_imagen_bytes(content)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Error reescalando {file.filename}: {exc}")

        with open(original_path, "wb") as f:
            f.write(resized_bytes)

        try:
            processed_bytes = procesar_imagen_bytes(resized_bytes)
            binarized_bytes = binarizar_imagen_bytes(resized_bytes)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error procesando {file.filename}: {exc}")

        with open(processed_path, "wb") as f:
            f.write(processed_bytes)

        with open(binarized_path, "wb") as f:
            f.write(binarized_bytes)

        item = {
            "id": image_id,
            "original_url": f"/files/originals/{original_name}",
            "processed_url": f"/files/processed/{processed_name}",
            "binarized_url": f"/files/binarized/{binarized_name}",
            "filename": file.filename,
        }
        items.append(item)
        new_items.append(item)

    save_index(items)
    return {"items": new_items}


@app.get("/files/originals/{filename}")
def get_original(filename: str):
    path = os.path.join(ORIGINAL_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    return FileResponse(path)


@app.get("/files/processed/{filename}")
def get_processed(filename: str):
    path = os.path.join(PROCESSED_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    return FileResponse(path)


@app.get("/files/binarized/{filename}")
def get_binarized(filename: str):
    path = os.path.join(BINARIZED_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    return FileResponse(path)


@app.get("/cluster-status")
def get_cluster_status():
    """Obtiene el estado actual del clustering"""
    global CLUSTER_MODEL, CLUSTER_CAPACITIES
    
    # Si el modelo está None, intentar restaurarlo desde el archivo
    if CLUSTER_MODEL is None:
        if os.path.exists(CLUSTER_STATE_FILE):
            try:
                with open(CLUSTER_STATE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    CLUSTER_MODEL = LinksClusterCapacityOnline.from_dict(data)
                    CLUSTER_CAPACITIES = CLUSTER_MODEL.capacities
            except:
                return {"active": False}
        else:
            return {"active": False}
    
    return {
        "active": True,
        "num_clusters": len(CLUSTER_MODEL.clusters),
        "capacities": CLUSTER_MODEL.capacities,
        "current_counts": CLUSTER_MODEL.cluster_counts,
        "available_spaces": [
            cap - count 
            for cap, count in zip(CLUSTER_MODEL.capacities, CLUSTER_MODEL.cluster_counts)
        ]
    }


@app.post("/update-capacities")
async def update_capacities(capacities: str = Form(...)):
    """Actualiza las capacidades de los clusters existentes"""
    global CLUSTER_MODEL, CLUSTER_CAPACITIES
    
    # Si el modelo está None, intentar restaurarlo desde el archivo
    if CLUSTER_MODEL is None:
        if os.path.exists(CLUSTER_STATE_FILE):
            try:
                with open(CLUSTER_STATE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    CLUSTER_MODEL = LinksClusterCapacityOnline.from_dict(data)
                    CLUSTER_CAPACITIES = CLUSTER_MODEL.capacities
            except:
                raise HTTPException(status_code=400, detail="No hay modelo activo")
        else:
            raise HTTPException(status_code=400, detail="No hay modelo activo. Ejecuta /analyze primero")
    
    try:
        new_caps = parse_capacities(capacities)
        if len(new_caps) != len(CLUSTER_MODEL.capacities):
            raise HTTPException(
                status_code=400,
                detail=f"Debe proporcionar {len(CLUSTER_MODEL.capacities)} capacidades (tienes {len(CLUSTER_MODEL.clusters)} clusters)"
            )
        
        CLUSTER_MODEL.capacities = new_caps
        save_cluster_state()
        
        # Calcular métricas de evaluación
        dunn_index = CLUSTER_MODEL.calculate_dunn_index()
        silhouette_coefficient = CLUSTER_MODEL.calculate_silhouette_coefficient()
        
        return {
            "status": "ok",
            "new_capacities": new_caps,
            "current_counts": CLUSTER_MODEL.cluster_counts,
            "available_spaces": [
                cap - count 
                for cap, count in zip(new_caps, CLUSTER_MODEL.cluster_counts)
            ],
            "metrics": {
                "dunn_index": round(float(dunn_index), 4),
                "silhouette_coefficient": round(float(silhouette_coefficient), 4),
            }
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/add-images")
async def add_images(files: List[UploadFile] = File(...)):
    """Agrega nuevas imágenes al clustering existente"""
    global CLUSTER_MODEL, CLUSTER_CAPACITIES
    
    # Si el modelo está None, intentar restaurarlo desde el archivo guardado
    if CLUSTER_MODEL is None:
        if os.path.exists(CLUSTER_STATE_FILE):
            try:
                with open(CLUSTER_STATE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    CLUSTER_MODEL = LinksClusterCapacityOnline.from_dict(data)
                    CLUSTER_CAPACITIES = CLUSTER_MODEL.capacities
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"No hay modelo de clustering activo. Usa /analyze primero"
                )
        else:
            raise HTTPException(
                status_code=400, 
                detail="No hay modelo de clustering activo. Usa /analyze primero"
            )
    
    if not files:
        raise HTTPException(status_code=400, detail="No se enviaron archivos")

    results = []
    moment_keys = [
        "m00", "m10", "m01", "m20", "m11", "m02", "m30", "m21", "m12", "m03",
        "mu20", "mu11", "mu02", "mu30", "mu21", "mu12", "mu03",
        "nu20", "nu11", "nu02", "nu30", "nu21", "nu12", "nu03",
    ]

    for file in files:
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(status_code=400, detail=f"Tipo no permitido: {file.content_type}")

        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"Archivo demasiado grande: {file.filename}")

        ext = ALLOWED_TYPES[file.content_type]
        processed_ext = ".png"
        image_id = uuid.uuid4().hex
        original_name = f"{image_id}{ext}"
        processed_name = f"{image_id}_processed{processed_ext}"
        binarized_name = f"{image_id}_binarized{processed_ext}"

        original_path = os.path.join(ORIGINAL_DIR, original_name)
        processed_path = os.path.join(PROCESSED_DIR, processed_name)
        binarized_path = os.path.join(BINARIZED_DIR, binarized_name)

        try:
            resized_bytes = reescalar_imagen_bytes(content)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Error reescalando {file.filename}: {exc}")

        with open(original_path, "wb") as f:
            f.write(resized_bytes)

        try:
            processed_bytes = procesar_imagen_bytes(resized_bytes)
            binarized_bytes = binarizar_imagen_bytes(resized_bytes)
            momentos = calcular_momentos(resized_bytes)
            vector = [float(momentos[k]) for k in moment_keys]
            vector_array = np.array(vector, dtype=float).reshape(1, -1)
            
            # Normalizar el vector con L2
            vector_normalizado = normalize(vector_array, norm='l2')[0]
            
            cluster_id, last_centroid = CLUSTER_MODEL.predict_with_centroid(
                vector_normalizado,
                allow_new_clusters=False,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error procesando {file.filename}: {exc}")

        with open(processed_path, "wb") as f:
            f.write(processed_bytes)

        with open(binarized_path, "wb") as f:
            f.write(binarized_bytes)

        result = {
            "id": image_id,
            "filename": file.filename,
            "original_url": f"/files/originals/{original_name}",
            "processed_url": f"/files/processed/{processed_name}",
            "binarized_url": f"/files/binarized/{binarized_name}",
            "momentos": momentos,
            "cluster_id": cluster_id,
            "ultimo_centroide": last_centroid.tolist(),
        }
        results.append(result)

        print(f"[ADD-CLUSTER] id={image_id} cluster={cluster_id} centroid={last_centroid.tolist()}")

    # Guardar estado del modelo después de agregar imágenes
    save_cluster_state()
    
    # Calcular métricas de evaluación
    dunn_index = CLUSTER_MODEL.calculate_dunn_index()
    silhouette_coefficient = CLUSTER_MODEL.calculate_silhouette_coefficient()
    
    return {
        "results": results,
        "metrics": {
            "dunn_index": round(float(dunn_index), 4),
            "silhouette_coefficient": round(float(silhouette_coefficient), 4),
        }
    }


# ==========================================
# SECCIÓN: MOMENTOS CON ETIQUETAS (EXTERNAL METRICS - ARI, AMI, NMI)
# ==========================================

@app.post("/external-metrics/initialize")
async def initialize_external_metrics(num_clusters: int = Form(...)):
    """
    Inicializa el apartado de Momentos con Etiquetas.
    El usuario especifica cuántos clusters desea.
    """
    global EXTERNAL_METRICS_STATE
    
    try:
        k = int(num_clusters)
        if k <= 0:
            raise ValueError("Debe ser > 0")
    except:
        raise HTTPException(status_code=400, detail="num_clusters inválido")
    
    EXTERNAL_METRICS_STATE = {
        "num_clusters": k,
        "labeled_data": {},  # {group_id: {"label": str, "images": [...]}}
        "predictions": [],  # etiquetas predichas por el modelo
        "true_labels": [],  # etiquetas verdaderas
        "metrics": {}
    }
    
    return {
        "status": "ok",
        "num_clusters": k,
        "ready_for_upload": True,
        "message": f"Listo para subir imágenes en {k} grupos"
    }


@app.post("/external-metrics/upload-group")
async def upload_group_images(
    group_id: int = Form(...),
    label: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    Sube imágenes de un grupo específico con su etiqueta verdadera
    """
    global EXTERNAL_METRICS_STATE
    
    if EXTERNAL_METRICS_STATE is None:
        raise HTTPException(status_code=400, detail="No hay sesión inicializada. Inicia primero indicando cuántos grupos.")
    
    if group_id < 0 or group_id >= EXTERNAL_METRICS_STATE["num_clusters"]:
        raise HTTPException(status_code=400, detail=f"group_id debe estar entre 0 y {EXTERNAL_METRICS_STATE['num_clusters']-1}")
    
    if not files:
        raise HTTPException(status_code=400, detail="No se enviaron archivos")
    
    if not label or len(label.strip()) == 0:
        raise HTTPException(status_code=400, detail="label vacía")
    
    moment_keys = [
        "m00", "m10", "m01", "m20", "m11", "m02", "m30", "m21", "m12", "m03",
        "mu20", "mu11", "mu02", "mu30", "mu21", "mu12", "mu03",
        "nu20", "nu11", "nu02", "nu30", "nu21", "nu12", "nu03",
    ]
    
    if group_id not in EXTERNAL_METRICS_STATE["labeled_data"]:
        EXTERNAL_METRICS_STATE["labeled_data"][group_id] = {
            "label": label.strip(),
            "images": []
        }
    
    results = []
    
    for file in files:
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(status_code=400, detail=f"Tipo no permitido: {file.content_type}")
        
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"Archivo demasiado grande")
        
        try:
            resized_bytes = reescalar_imagen_bytes(content)
            processed_bytes = procesar_imagen_bytes(resized_bytes)
            binarized_bytes = binarizar_imagen_bytes(resized_bytes)
            momentos = calcular_momentos(resized_bytes)
            
            # Crear vector de características normalizado
            vector = [float(momentos[k]) for k in moment_keys]
            vector_array = np.array(vector, dtype=float).reshape(1, -1)
            vector_normalizado = normalize(vector_array, norm='l2')[0]
            
            # Guardar archivos
            ext = ALLOWED_TYPES[file.content_type]
            image_id = uuid.uuid4().hex
            original_name = f"{image_id}{ext}"
            processed_name = f"{image_id}_processed.png"
            binarized_name = f"{image_id}_binarized.png"
            
            original_path = os.path.join(ORIGINAL_DIR, original_name)
            processed_path = os.path.join(PROCESSED_DIR, processed_name)
            binarized_path = os.path.join(BINARIZED_DIR, binarized_name)
            
            with open(original_path, "wb") as f:
                f.write(resized_bytes)
            with open(processed_path, "wb") as f:
                f.write(processed_bytes)
            with open(binarized_path, "wb") as f:
                f.write(binarized_bytes)

            image_data = {
                "filename": file.filename,
                "vector": vector_normalizado.tolist(),
                "momentos": momentos,
                "true_label": label.strip(),
                "original_url": f"/files/originals/{original_name}",
                "processed_url": f"/files/processed/{processed_name}",
                "binarized_url": f"/files/binarized/{binarized_name}",
            }
            
            EXTERNAL_METRICS_STATE["labeled_data"][group_id]["images"].append(image_data)
            
            results.append({
                "filename": file.filename,
                "original_url": f"/files/originals/{original_name}",
                "processed_url": f"/files/processed/{processed_name}",
                "binarized_url": f"/files/binarized/{binarized_name}"
            })
            
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error procesando {file.filename}: {exc}")
    
    return {
        "status": "ok",
        "group_id": group_id,
        "label": label,
        "num_images_uploaded": len(results),
        "total_images_in_group": len(EXTERNAL_METRICS_STATE["labeled_data"][group_id]["images"]),
        "uploaded": results
    }


@app.post("/external-metrics/calculate")
async def calculate_external_metrics(capacities: str | None = Form(None)):
    """
    Calcula ARI, AMI y NMI comparando etiquetas verdaderas con predicciones del modelo de clustering
    """
    global EXTERNAL_METRICS_STATE, CLUSTER_MODEL_MOMENTS_LABELED
    
    if EXTERNAL_METRICS_STATE is None or not EXTERNAL_METRICS_STATE["labeled_data"]:
        raise HTTPException(status_code=400, detail="No hay datos etiquetados")
    
    # Recopilar todos los vectores y etiquetas verdaderas
    all_vectors = []
    true_labels_list = []
    image_info = []
    
    for group_id in sorted(EXTERNAL_METRICS_STATE["labeled_data"].keys()):
        group_data = EXTERNAL_METRICS_STATE["labeled_data"][group_id]
        label = group_data["label"]
        for img_data in group_data["images"]:
            vector = np.array(img_data["vector"])
            all_vectors.append(vector)
            true_labels_list.append(label)
            image_info.append({
                "filename": img_data.get("filename"),
                "true_label": label,
                "original_url": img_data.get("original_url"),
                "processed_url": img_data.get("processed_url"),
                "binarized_url": img_data.get("binarized_url"),
            })
    
    if len(all_vectors) == 0:
        raise HTTPException(status_code=400, detail="No hay imágenes para analizar")

    # Randomizar el orden de entrada para evitar secuencias 1,1,1,2,2,2
    perm = np.random.permutation(len(all_vectors))
    all_vectors = [all_vectors[i] for i in perm]
    true_labels_list = [true_labels_list[i] for i in perm]
    image_info = [image_info[i] for i in perm]
    
    # Crear modelo de clustering con capacidades
    num_clusters = EXTERNAL_METRICS_STATE["num_clusters"]
    total_images = len(all_vectors)
    if capacities:
        try:
            capacities_list = parse_capacities(capacities)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"capacities inválido: {exc}")
        if len(capacities_list) != num_clusters:
            raise HTTPException(
                status_code=400,
                detail=f"Debe proporcionar {num_clusters} capacidades (tienes {len(capacities_list)})",
            )
        capacities = capacities_list
    else:
        capacity_per_cluster = max(total_images // num_clusters + 2, 5)
        capacities = [capacity_per_cluster] * num_clusters
    
    CLUSTER_MODEL_MOMENTS_LABELED = LinksClusterCapacityOnline(capacities=capacities)
    
    # Predecir clusters para todos los vectores
    predicted_labels_list = []
    image_urls = []
    
    for idx, vector in enumerate(all_vectors):
        try:
            cluster_id, _ = CLUSTER_MODEL_MOMENTS_LABELED.predict_with_centroid(vector)
            predicted_labels_list.append(cluster_id)
        except:
            # Si hay error, asignar al cluster 0
            predicted_labels_list.append(0)
        
        # Guardar URLs de las imágenes
        image_data = image_info[idx] if idx < len(image_info) else {"filename": f"img_{idx}"}
        image_urls.append({
            "filename": image_data.get("filename", f"img_{idx}"),
            "true_label": image_data.get("true_label", "unknown"),
            "predicted_cluster": predicted_labels_list[idx],
            "original_url": image_data.get("original_url"),
            "processed_url": image_data.get("processed_url"),
            "binarized_url": image_data.get("binarized_url"),
        })
    
    # Convertir etiquetas a índices numéricos para las métricas
    le_true = LabelEncoder()
    y_true_numeric = le_true.fit_transform(true_labels_list)
    y_pred_numeric = np.array(predicted_labels_list)
    
    # Calcular métricas
    ari = adjusted_rand_score(y_true_numeric, y_pred_numeric)
    ami = adjusted_mutual_info_score(y_true_numeric, y_pred_numeric)
    nmi = normalized_mutual_info_score(y_true_numeric, y_pred_numeric)
    
    # Calcular métricas internas también
    dunn_index = CLUSTER_MODEL_MOMENTS_LABELED.calculate_dunn_index()
    silhouette_coefficient = CLUSTER_MODEL_MOMENTS_LABELED.calculate_silhouette_coefficient()
    
    # Agrupar imágenes por cluster predicho
    clusters_visualization = {}
    for idx, cluster_id in enumerate(predicted_labels_list):
        if cluster_id not in clusters_visualization:
            clusters_visualization[cluster_id] = []
        
        clusters_visualization[cluster_id].append({
            "index": idx,
            "filename": image_urls[idx]["filename"],
            "true_label": image_urls[idx]["true_label"],
            "predicted_cluster": cluster_id,
            "original_url": image_urls[idx].get("original_url"),
            "processed_url": image_urls[idx].get("processed_url"),
            "binarized_url": image_urls[idx].get("binarized_url"),
        })
    
    # Almacenar resultados
    EXTERNAL_METRICS_STATE["metrics"] = {
        "ARI": float(ari),
        "AMI": float(ami),
        "NMI": float(nmi),
        "dunn_index": float(dunn_index),
        "silhouette_coefficient": float(silhouette_coefficient),
        "num_images": len(all_vectors),
        "num_clusters": num_clusters,
        "true_labels_count": len(set(true_labels_list)),
        "predicted_labels_count": len(set(predicted_labels_list))
    }
    
    EXTERNAL_METRICS_STATE["true_labels"] = true_labels_list
    EXTERNAL_METRICS_STATE["predictions"] = predicted_labels_list
    EXTERNAL_METRICS_STATE["image_info"] = image_info
    EXTERNAL_METRICS_STATE["clusters_visualization"] = clusters_visualization
    
    return {
        "status": "ok",
        "external_metrics": {
            "ARI": round(ari, 4),
            "AMI": round(ami, 4),
            "NMI": round(nmi, 4)
        },
        "internal_metrics": {
            "dunn_index": round(dunn_index, 4),
            "silhouette_coefficient": round(silhouette_coefficient, 4)
        },
        "clusters": clusters_visualization,
        "summary": {
            "num_images": len(all_vectors),
            "num_clusters": num_clusters,
            "true_groups": len(set(true_labels_list)),
            "predicted_clusters": len(set(predicted_labels_list))
        }
    }


@app.get("/external-metrics/status")
def get_external_metrics_status():
    """Retorna el estado actual de las métricas externas"""
    global EXTERNAL_METRICS_STATE
    
    if EXTERNAL_METRICS_STATE is None:
        return {"initialized": False}
    
    group_summary = {}
    for group_id, data in EXTERNAL_METRICS_STATE["labeled_data"].items():
        group_summary[f"grupo_{group_id}"] = {
            "label": data["label"],
            "num_images": len(data["images"])
        }
    
    return {
        "initialized": True,
        "num_clusters": EXTERNAL_METRICS_STATE["num_clusters"],
        "groups_with_data": group_summary,
        "metrics": EXTERNAL_METRICS_STATE.get("metrics", {}),
        "total_images": sum(
            len(data["images"]) 
            for data in EXTERNAL_METRICS_STATE["labeled_data"].values()
        )
    }


@app.delete("/external-metrics/reset")
def reset_external_metrics():
    """Reinicia la sesión de métricas externas"""
    global EXTERNAL_METRICS_STATE, CLUSTER_MODEL_MOMENTS_LABELED
    EXTERNAL_METRICS_STATE = None
    CLUSTER_MODEL_MOMENTS_LABELED = None
    return {"status": "reset", "message": "Sesión de métricas externa reiniciada"}


# ==================== HELPER PARA PROCESAR DESCRIPTORES POR TIPO ====================

def process_image_and_extract_features(resized_bytes: bytes, method_type: str):
    """
    Procesa una imagen y extrae sus características según el método especificado
    Returns: (processed_bytes, vector, method_name)
    """
    if method_type == "hu":
        momentos_hu = calcular_momentos_hu(resized_bytes)
        processed_bytes = procesar_imagen_bytes(resized_bytes)
        vector = [m for m in momentos_hu.values()]
        return processed_bytes, vector, "hu"
    
    elif method_type == "zernike":
        momentos_zernike = calcular_momentos_zernike(resized_bytes)
        processed_bytes = procesar_imagen_bytes(resized_bytes)
        vector = [m for m in momentos_zernike.values()]
        return processed_bytes, vector, "zernike"
    
    elif method_type == "sift":
        processed_bytes, descriptores_sift = procesar_sift_con_descriptores(resized_bytes)
        if not descriptores_sift:
            raise ValueError("No se pudieron extraer características SIFT")
        return processed_bytes, descriptores_sift, "sift"
    
    elif method_type == "hog":
        processed_bytes, descriptores_hog = procesar_hog_con_descriptores(resized_bytes)
        if not descriptores_hog:
            raise ValueError("No se pudieron extraer características HOG")
        return processed_bytes, descriptores_hog, "hog"
    
    elif method_type == "cnn":
        processed_bytes, descriptores_cnn = procesar_cnn_con_descriptores(resized_bytes)
        if not descriptores_cnn:
            raise ValueError("No se pudieron extraer características CNN")
        return processed_bytes, descriptores_cnn, "cnn"
    
    else:
        raise ValueError(f"Tipo de método no soportado: {method_type}")


def get_external_state(method_type: str):
    """Retorna el estado global correspondiente al método"""
    method_map = {
        "hu": ("EXTERNAL_METRICS_HU_STATE", "CLUSTER_MODEL_HU_LABELED"),
        "zernike": ("EXTERNAL_METRICS_ZERNIKE_STATE", "CLUSTER_MODEL_ZERNIKE_LABELED"),
        "sift": ("EXTERNAL_METRICS_SIFT_STATE", "CLUSTER_MODEL_SIFT_LABELED"),
        "hog": ("EXTERNAL_METRICS_HOG_STATE", "CLUSTER_MODEL_HOG_LABELED"),
        "cnn": ("EXTERNAL_METRICS_CNN_STATE", "CLUSTER_MODEL_CNN_LABELED"),
    }
    return method_map.get(method_type, (None, None))


# ==================== HU CON ETIQUETAS (ARI/AMI/NMI) ====================

@app.post("/external-metrics-hu/initialize")
async def initialize_external_metrics_hu(num_clusters: int = Form(...)):
    global EXTERNAL_METRICS_HU_STATE
    
    try:
        k = int(num_clusters)
        if k <= 0:
            raise ValueError("Debe ser > 0")
    except:
        raise HTTPException(status_code=400, detail="num_clusters inválido")
    
    EXTERNAL_METRICS_HU_STATE = {
        "num_clusters": k,
        "labeled_data": {},
        "predictions": [],
        "true_labels": [],
        "metrics": {}
    }
    
    return {
        "status": "ok",
        "num_clusters": k,
        "ready_for_upload": True,
        "message": f"Listo para subir imágenes Hu en {k} grupos"
    }


@app.post("/external-metrics-hu/upload-group")
async def upload_group_images_hu(
    group_id: int = Form(...),
    label: str = Form(...),
    files: List[UploadFile] = File(...)
):
    global EXTERNAL_METRICS_HU_STATE
    
    if EXTERNAL_METRICS_HU_STATE is None:
        raise HTTPException(status_code=400, detail="No hay sesión inicializada")
    
    if group_id < 0 or group_id >= EXTERNAL_METRICS_HU_STATE["num_clusters"]:
        raise HTTPException(status_code=400, detail=f"group_id inválido")
    
    if not files:
        raise HTTPException(status_code=400, detail="No se enviaron archivos")
    
    if group_id not in EXTERNAL_METRICS_HU_STATE["labeled_data"]:
        EXTERNAL_METRICS_HU_STATE["labeled_data"][group_id] = {
            "label": label.strip(),
            "images": []
        }
    
    results = []
    
    for file in files:
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(status_code=400, detail=f"Tipo no permitido")
        
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"Archivo demasiado grande")
        
        try:
            resized_bytes = reescalar_imagen_bytes(content)
            processed_bytes, vector, _ = process_image_and_extract_features(resized_bytes, "hu")
            
            vector_array = np.array(vector, dtype=float).reshape(1, -1)
            vector_normalizado = normalize(vector_array, norm='l2')[0]
            
            ext = ALLOWED_TYPES[file.content_type]
            image_id = uuid.uuid4().hex
            original_name = f"{image_id}{ext}"
            processed_name = f"{image_id}_hu.png"
            
            with open(os.path.join(ORIGINAL_DIR, original_name), "wb") as f:
                f.write(resized_bytes)
            with open(os.path.join(PROCESSED_DIR, processed_name), "wb") as f:
                f.write(processed_bytes)

            EXTERNAL_METRICS_HU_STATE["labeled_data"][group_id]["images"].append({
                "filename": file.filename,
                "vector": vector_normalizado.tolist(),
                "true_label": label.strip(),
                "original_url": f"/files/originals/{original_name}",
                "processed_url": f"/files/processed/{processed_name}",
            })
            
            results.append({
                "filename": file.filename,
                "original_url": f"/files/originals/{original_name}",
                "processed_url": f"/files/processed/{processed_name}"
            })
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error: {exc}")
    
    return {
        "status": "ok",
        "group_id": group_id,
        "label": label,
        "num_images_uploaded": len(results)
    }


@app.post("/external-metrics-hu/calculate")
async def calculate_external_metrics_hu(capacities: str | None = Form(None)):
    global EXTERNAL_METRICS_HU_STATE, CLUSTER_MODEL_HU_LABELED
    
    if EXTERNAL_METRICS_HU_STATE is None or not EXTERNAL_METRICS_HU_STATE["labeled_data"]:
        raise HTTPException(status_code=400, detail="No hay datos etiquetados")
    
    all_vectors = []
    true_labels_list = []
    image_info = []
    
    for group_id in sorted(EXTERNAL_METRICS_HU_STATE["labeled_data"].keys()):
        group_data = EXTERNAL_METRICS_HU_STATE["labeled_data"][group_id]
        label = group_data["label"]
        for img_data in group_data["images"]:
            all_vectors.append(np.array(img_data["vector"]))
            true_labels_list.append(label)
            image_info.append({
                "filename": img_data.get("filename"),
                "true_label": label,
                "original_url": img_data.get("original_url"),
                "processed_url": img_data.get("processed_url"),
            })
    
    if len(all_vectors) == 0:
        raise HTTPException(status_code=400, detail="No hay imágenes")

    perm = np.random.permutation(len(all_vectors))
    all_vectors = [all_vectors[i] for i in perm]
    true_labels_list = [true_labels_list[i] for i in perm]
    image_info = [image_info[i] for i in perm]
    
    num_clusters = EXTERNAL_METRICS_HU_STATE["num_clusters"]
    total_images = len(all_vectors)
    if capacities:
        capacities_list = parse_capacities(capacities)
        if len(capacities_list) != num_clusters:
            raise HTTPException(status_code=400, detail=f"Debe proporcionar {num_clusters} capacidades")
        capacities = capacities_list
    else:
        capacities = [max(total_images // num_clusters + 2, 5)] * num_clusters
    
    CLUSTER_MODEL_HU_LABELED = LinksClusterCapacityOnline(capacities=capacities)
    
    predicted_labels_list = []
    image_urls = []
    
    for idx, vector in enumerate(all_vectors):
        try:
            cluster_id, _ = CLUSTER_MODEL_HU_LABELED.predict_with_centroid(vector)
            predicted_labels_list.append(cluster_id)
        except:
            predicted_labels_list.append(0)
        
        image_urls.append({
            "filename": image_info[idx].get("filename", f"img_{idx}"),
            "true_label": image_info[idx].get("true_label", "unknown"),
            "predicted_cluster": predicted_labels_list[idx],
            "original_url": image_info[idx].get("original_url"),
            "processed_url": image_info[idx].get("processed_url"),
        })
    
    le_true = LabelEncoder()
    y_true_numeric = le_true.fit_transform(true_labels_list)
    y_pred_numeric = np.array(predicted_labels_list)
    
    ari = adjusted_rand_score(y_true_numeric, y_pred_numeric)
    ami = adjusted_mutual_info_score(y_true_numeric, y_pred_numeric)
    nmi = normalized_mutual_info_score(y_true_numeric, y_pred_numeric)
    
    dunn_index = CLUSTER_MODEL_HU_LABELED.calculate_dunn_index()
    silhouette_coefficient = CLUSTER_MODEL_HU_LABELED.calculate_silhouette_coefficient()
    
    clusters_visualization = {}
    for idx, cluster_id in enumerate(predicted_labels_list):
        if cluster_id not in clusters_visualization:
            clusters_visualization[cluster_id] = []
        
        clusters_visualization[cluster_id].append({
            "index": idx,
            "filename": image_urls[idx]["filename"],
            "true_label": image_urls[idx]["true_label"],
            "predicted_cluster": cluster_id,
            "original_url": image_urls[idx].get("original_url"),
            "processed_url": image_urls[idx].get("processed_url"),
        })
    
    EXTERNAL_METRICS_HU_STATE["metrics"] = {
        "ARI": float(ari),
        "AMI": float(ami),
        "NMI": float(nmi),
        "dunn_index": float(dunn_index),
        "silhouette_coefficient": float(silhouette_coefficient),
    }
    
    return {
        "status": "ok",
        "external_metrics": {
            "ARI": round(ari, 4),
            "AMI": round(ami, 4),
            "NMI": round(nmi, 4)
        },
        "internal_metrics": {
            "dunn_index": round(dunn_index, 4),
            "silhouette_coefficient": round(silhouette_coefficient, 4)
        },
        "clusters": clusters_visualization,
        "summary": {
            "num_images": len(all_vectors),
            "num_clusters": num_clusters,
            "true_groups": len(set(true_labels_list)),
            "predicted_clusters": len(set(predicted_labels_list))
        }
    }


@app.delete("/external-metrics-hu/reset")
def reset_external_metrics_hu():
    global EXTERNAL_METRICS_HU_STATE, CLUSTER_MODEL_HU_LABELED
    EXTERNAL_METRICS_HU_STATE = None
    CLUSTER_MODEL_HU_LABELED = None
    return {"status": "reset", "message": "Sesión Hu reiniciada"}


# ==================== ZERNIKE CON ETIQUETAS (ARI/AMI/NMI) ====================

@app.post("/external-metrics-zernike/initialize")
async def initialize_external_metrics_zernike(num_clusters: int = Form(...)):
    global EXTERNAL_METRICS_ZERNIKE_STATE
    
    try:
        k = int(num_clusters)
        if k <= 0:
            raise ValueError("Debe ser > 0")
    except:
        raise HTTPException(status_code=400, detail="num_clusters inválido")
    
    EXTERNAL_METRICS_ZERNIKE_STATE = {
        "num_clusters": k,
        "labeled_data": {},
        "predictions": [],
        "true_labels": [],
        "metrics": {}
    }
    
    return {"status": "initialized", "num_clusters": k}


@app.post("/external-metrics-zernike/upload-group")
async def upload_group_zernike(
    group_id: int = Form(...),
    label: str = Form(...),
    files: list[UploadFile] = File(...)
):
    global EXTERNAL_METRICS_ZERNIKE_STATE
    
    if EXTERNAL_METRICS_ZERNIKE_STATE is None:
        raise HTTPException(status_code=400, detail="No hay sesión inicializada")
    
    if group_id < 0 or group_id >= EXTERNAL_METRICS_ZERNIKE_STATE["num_clusters"]:
        raise HTTPException(status_code=400, detail=f"group_id inválido")
    
    if not files:
        raise HTTPException(status_code=400, detail="No se enviaron archivos")
    
    if group_id not in EXTERNAL_METRICS_ZERNIKE_STATE["labeled_data"]:
        EXTERNAL_METRICS_ZERNIKE_STATE["labeled_data"][group_id] = {
            "label": label.strip(),
            "images": []
        }
    
    results = []
    
    for file in files:
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(status_code=400, detail=f"Tipo no permitido")
        
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"Archivo demasiado grande")
        
        try:
            resized_bytes = reescalar_imagen_bytes(content)
            processed_bytes, vector, _ = process_image_and_extract_features(resized_bytes, "zernike")
            
            vector_array = np.array(vector, dtype=float).reshape(1, -1)
            vector_normalizado = normalize(vector_array, norm='l2')[0]
            
            ext = ALLOWED_TYPES[file.content_type]
            image_id = uuid.uuid4().hex
            original_name = f"{image_id}{ext}"
            processed_name = f"{image_id}_zernike.png"
            
            with open(os.path.join(ORIGINAL_DIR, original_name), "wb") as f:
                f.write(resized_bytes)
            with open(os.path.join(PROCESSED_DIR, processed_name), "wb") as f:
                f.write(processed_bytes)

            EXTERNAL_METRICS_ZERNIKE_STATE["labeled_data"][group_id]["images"].append({
                "filename": file.filename,
                "vector": vector_normalizado.tolist(),
                "true_label": label.strip(),
                "original_url": f"/files/originals/{original_name}",
                "processed_url": f"/files/processed/{processed_name}",
            })
            
            results.append({
                "filename": file.filename,
                "original_url": f"/files/originals/{original_name}",
                "processed_url": f"/files/processed/{processed_name}"
            })
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error: {exc}")
    
    return {
        "status": "ok",
        "group_id": group_id,
        "label": label,
        "num_images_uploaded": len(results)
    }


@app.post("/external-metrics-zernike/calculate")
async def calculate_external_metrics_zernike(capacities: str | None = Form(None)):
    global EXTERNAL_METRICS_ZERNIKE_STATE, CLUSTER_MODEL_ZERNIKE_LABELED
    
    if EXTERNAL_METRICS_ZERNIKE_STATE is None or not EXTERNAL_METRICS_ZERNIKE_STATE["labeled_data"]:
        raise HTTPException(status_code=400, detail="No hay datos etiquetados")
    
    all_vectors = []
    true_labels_list = []
    image_info = []
    
    for group_id in sorted(EXTERNAL_METRICS_ZERNIKE_STATE["labeled_data"].keys()):
        group_data = EXTERNAL_METRICS_ZERNIKE_STATE["labeled_data"][group_id]
        label = group_data["label"]
        for img_data in group_data["images"]:
            all_vectors.append(np.array(img_data["vector"]))
            true_labels_list.append(label)
            image_info.append({
                "filename": img_data.get("filename"),
                "true_label": label,
                "original_url": img_data.get("original_url"),
                "processed_url": img_data.get("processed_url"),
            })
    
    if len(all_vectors) == 0:
        raise HTTPException(status_code=400, detail="No hay imágenes")

    perm = np.random.permutation(len(all_vectors))
    all_vectors = [all_vectors[i] for i in perm]
    true_labels_list = [true_labels_list[i] for i in perm]
    image_info = [image_info[i] for i in perm]
    
    num_clusters = EXTERNAL_METRICS_ZERNIKE_STATE["num_clusters"]
    total_images = len(all_vectors)
    if capacities:
        capacities_list = parse_capacities(capacities)
        if len(capacities_list) != num_clusters:
            raise HTTPException(status_code=400, detail=f"Debe proporcionar {num_clusters} capacidades")
        capacities = capacities_list
    else:
        capacities = [max(total_images // num_clusters + 2, 5)] * num_clusters
    
    CLUSTER_MODEL_ZERNIKE_LABELED = LinksClusterCapacityOnline(capacities=capacities)
    
    predicted_labels_list = []
    image_urls = []
    
    for idx, vector in enumerate(all_vectors):
        try:
            cluster_id, _ = CLUSTER_MODEL_ZERNIKE_LABELED.predict_with_centroid(vector)
            predicted_labels_list.append(cluster_id)
        except:
            predicted_labels_list.append(0)
        
        image_urls.append({
            "filename": image_info[idx].get("filename", f"img_{idx}"),
            "true_label": image_info[idx].get("true_label", "unknown"),
            "predicted_cluster": predicted_labels_list[idx],
            "original_url": image_info[idx].get("original_url"),
            "processed_url": image_info[idx].get("processed_url"),
        })
    
    le_true = LabelEncoder()
    y_true_numeric = le_true.fit_transform(true_labels_list)
    y_pred_numeric = np.array(predicted_labels_list)
    
    ari = adjusted_rand_score(y_true_numeric, y_pred_numeric)
    ami = adjusted_mutual_info_score(y_true_numeric, y_pred_numeric)
    nmi = normalized_mutual_info_score(y_true_numeric, y_pred_numeric)
    
    dunn_index = CLUSTER_MODEL_ZERNIKE_LABELED.calculate_dunn_index()
    silhouette_coefficient = CLUSTER_MODEL_ZERNIKE_LABELED.calculate_silhouette_coefficient()
    
    clusters_visualization = {}
    for idx, cluster_id in enumerate(predicted_labels_list):
        if cluster_id not in clusters_visualization:
            clusters_visualization[cluster_id] = []
        
        clusters_visualization[cluster_id].append({
            "index": idx,
            "filename": image_urls[idx]["filename"],
            "true_label": image_urls[idx]["true_label"],
            "predicted_cluster": cluster_id,
            "original_url": image_urls[idx].get("original_url"),
            "processed_url": image_urls[idx].get("processed_url"),
        })
    
    EXTERNAL_METRICS_ZERNIKE_STATE["metrics"] = {
        "ARI": float(ari),
        "AMI": float(ami),
        "NMI": float(nmi),
        "dunn_index": float(dunn_index),
        "silhouette_coefficient": float(silhouette_coefficient),
    }
    
    return {
        "status": "ok",
        "external_metrics": {
            "ARI": round(ari, 4),
            "AMI": round(ami, 4),
            "NMI": round(nmi, 4)
        },
        "internal_metrics": {
            "dunn_index": round(dunn_index, 4),
            "silhouette_coefficient": round(silhouette_coefficient, 4)
        },
        "clusters": clusters_visualization,
        "summary": {
            "num_images": len(all_vectors),
            "num_clusters": num_clusters,
            "true_groups": len(set(true_labels_list)),
            "predicted_clusters": len(set(predicted_labels_list))
        }
    }


@app.delete("/external-metrics-zernike/reset")
def reset_external_metrics_zernike():
    global EXTERNAL_METRICS_ZERNIKE_STATE, CLUSTER_MODEL_ZERNIKE_LABELED
    EXTERNAL_METRICS_ZERNIKE_STATE = None
    CLUSTER_MODEL_ZERNIKE_LABELED = None
    return {"status": "reset", "message": "Sesión Zernike reiniciada"}


# ==================== SIFT CON ETIQUETAS (ARI/AMI/NMI) ====================

@app.post("/external-metrics-sift/initialize")
async def initialize_external_metrics_sift(num_clusters: int = Form(...)):
    global EXTERNAL_METRICS_SIFT_STATE
    
    try:
        k = int(num_clusters)
        if k <= 0:
            raise ValueError("Debe ser > 0")
    except:
        raise HTTPException(status_code=400, detail="num_clusters inválido")
    
    EXTERNAL_METRICS_SIFT_STATE = {
        "num_clusters": k,
        "labeled_data": {},
        "predictions": [],
        "true_labels": [],
        "metrics": {}
    }
    
    return {"status": "initialized", "num_clusters": k}


@app.post("/external-metrics-sift/upload-group")
async def upload_group_sift(
    group_id: int = Form(...),
    label: str = Form(...),
    files: list[UploadFile] = File(...)
):
    global EXTERNAL_METRICS_SIFT_STATE
    
    if EXTERNAL_METRICS_SIFT_STATE is None:
        raise HTTPException(status_code=400, detail="No hay sesión inicializada")
    
    if group_id < 0 or group_id >= EXTERNAL_METRICS_SIFT_STATE["num_clusters"]:
        raise HTTPException(status_code=400, detail=f"group_id inválido")
    
    if not files:
        raise HTTPException(status_code=400, detail="No se enviaron archivos")
    
    if group_id not in EXTERNAL_METRICS_SIFT_STATE["labeled_data"]:
        EXTERNAL_METRICS_SIFT_STATE["labeled_data"][group_id] = {
            "label": label.strip(),
            "images": []
        }
    
    results = []
    
    for file in files:
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(status_code=400, detail=f"Tipo no permitido")
        
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"Archivo demasiado grande")
        
        try:
            resized_bytes = reescalar_imagen_bytes(content)
            processed_bytes, descriptores, _ = process_image_and_extract_features(resized_bytes, "sift")
            
            # SIFT devuelve múltiples descriptores, promediarlos
            if not descriptores:
                raise ValueError("No se pudieron extraer características SIFT")
            descriptores_array = np.array(descriptores, dtype=float)
            vector = np.mean(descriptores_array, axis=0)  # Promedio de descriptores
            
            vector_array = vector.reshape(1, -1)
            vector_normalizado = normalize(vector_array, norm='l2')[0]
            
            ext = ALLOWED_TYPES[file.content_type]
            image_id = uuid.uuid4().hex
            original_name = f"{image_id}{ext}"
            processed_name = f"{image_id}_sift.png"
            
            with open(os.path.join(ORIGINAL_DIR, original_name), "wb") as f:
                f.write(resized_bytes)
            with open(os.path.join(PROCESSED_DIR, processed_name), "wb") as f:
                f.write(processed_bytes)

            EXTERNAL_METRICS_SIFT_STATE["labeled_data"][group_id]["images"].append({
                "filename": file.filename,
                "vector": vector_normalizado.tolist(),
                "true_label": label.strip(),
                "original_url": f"/files/originals/{original_name}",
                "processed_url": f"/files/processed/{processed_name}",
            })
            
            results.append({
                "filename": file.filename,
                "original_url": f"/files/originals/{original_name}",
                "processed_url": f"/files/processed/{processed_name}"
            })
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error: {exc}")
    
    return {
        "status": "ok",
        "group_id": group_id,
        "label": label,
        "num_images_uploaded": len(results)
    }


@app.post("/external-metrics-sift/calculate")
async def calculate_external_metrics_sift(capacities: str | None = Form(None)):
    global EXTERNAL_METRICS_SIFT_STATE, CLUSTER_MODEL_SIFT_LABELED
    
    if EXTERNAL_METRICS_SIFT_STATE is None or not EXTERNAL_METRICS_SIFT_STATE["labeled_data"]:
        raise HTTPException(status_code=400, detail="No hay datos etiquetados")
    
    all_vectors = []
    true_labels_list = []
    image_info = []
    
    for group_id in sorted(EXTERNAL_METRICS_SIFT_STATE["labeled_data"].keys()):
        group_data = EXTERNAL_METRICS_SIFT_STATE["labeled_data"][group_id]
        label = group_data["label"]
        for img_data in group_data["images"]:
            all_vectors.append(np.array(img_data["vector"]))
            true_labels_list.append(label)
            image_info.append({
                "filename": img_data.get("filename"),
                "true_label": label,
                "original_url": img_data.get("original_url"),
                "processed_url": img_data.get("processed_url"),
            })
    
    if len(all_vectors) == 0:
        raise HTTPException(status_code=400, detail="No hay imágenes")

    perm = np.random.permutation(len(all_vectors))
    all_vectors = [all_vectors[i] for i in perm]
    true_labels_list = [true_labels_list[i] for i in perm]
    image_info = [image_info[i] for i in perm]
    
    num_clusters = EXTERNAL_METRICS_SIFT_STATE["num_clusters"]
    total_images = len(all_vectors)
    if capacities:
        capacities_list = parse_capacities(capacities)
        if len(capacities_list) != num_clusters:
            raise HTTPException(status_code=400, detail=f"Debe proporcionar {num_clusters} capacidades")
        capacities = capacities_list
    else:
        capacities = [max(total_images // num_clusters + 2, 5)] * num_clusters
    
    CLUSTER_MODEL_SIFT_LABELED = LinksClusterCapacityOnline(capacities=capacities)
    
    predicted_labels_list = []
    image_urls = []
    
    for idx, vector in enumerate(all_vectors):
        try:
            cluster_id, _ = CLUSTER_MODEL_SIFT_LABELED.predict_with_centroid(vector)
            predicted_labels_list.append(cluster_id)
        except:
            predicted_labels_list.append(0)
        
        image_urls.append({
            "filename": image_info[idx].get("filename", f"img_{idx}"),
            "true_label": image_info[idx].get("true_label", "unknown"),
            "predicted_cluster": predicted_labels_list[idx],
            "original_url": image_info[idx].get("original_url"),
            "processed_url": image_info[idx].get("processed_url"),
        })
    
    le_true = LabelEncoder()
    y_true_numeric = le_true.fit_transform(true_labels_list)
    y_pred_numeric = np.array(predicted_labels_list)
    
    ari = adjusted_rand_score(y_true_numeric, y_pred_numeric)
    ami = adjusted_mutual_info_score(y_true_numeric, y_pred_numeric)
    nmi = normalized_mutual_info_score(y_true_numeric, y_pred_numeric)
    
    dunn_index = CLUSTER_MODEL_SIFT_LABELED.calculate_dunn_index()
    silhouette_coefficient = CLUSTER_MODEL_SIFT_LABELED.calculate_silhouette_coefficient()
    
    clusters_visualization = {}
    for idx, cluster_id in enumerate(predicted_labels_list):
        if cluster_id not in clusters_visualization:
            clusters_visualization[cluster_id] = []
        
        clusters_visualization[cluster_id].append({
            "index": idx,
            "filename": image_urls[idx]["filename"],
            "true_label": image_urls[idx]["true_label"],
            "predicted_cluster": cluster_id,
            "original_url": image_urls[idx].get("original_url"),
            "processed_url": image_urls[idx].get("processed_url"),
        })
    
    EXTERNAL_METRICS_SIFT_STATE["metrics"] = {
        "ARI": float(ari),
        "AMI": float(ami),
        "NMI": float(nmi),
        "dunn_index": float(dunn_index),
        "silhouette_coefficient": float(silhouette_coefficient),
    }
    
    return {
        "status": "ok",
        "external_metrics": {
            "ARI": round(ari, 4),
            "AMI": round(ami, 4),
            "NMI": round(nmi, 4)
        },
        "internal_metrics": {
            "dunn_index": round(dunn_index, 4),
            "silhouette_coefficient": round(silhouette_coefficient, 4)
        },
        "clusters": clusters_visualization,
        "summary": {
            "num_images": len(all_vectors),
            "num_clusters": num_clusters,
            "true_groups": len(set(true_labels_list)),
            "predicted_clusters": len(set(predicted_labels_list))
        }
    }


@app.delete("/external-metrics-sift/reset")
def reset_external_metrics_sift():
    global EXTERNAL_METRICS_SIFT_STATE, CLUSTER_MODEL_SIFT_LABELED
    EXTERNAL_METRICS_SIFT_STATE = None
    CLUSTER_MODEL_SIFT_LABELED = None
    return {"status": "reset", "message": "Sesión SIFT reiniciada"}


# ==================== HOG CON ETIQUETAS (ARI/AMI/NMI) ====================

@app.post("/external-metrics-hog/initialize")
async def initialize_external_metrics_hog(num_clusters: int = Form(...)):
    global EXTERNAL_METRICS_HOG_STATE
    
    try:
        k = int(num_clusters)
        if k <= 0:
            raise ValueError("Debe ser > 0")
    except:
        raise HTTPException(status_code=400, detail="num_clusters inválido")
    
    EXTERNAL_METRICS_HOG_STATE = {
        "num_clusters": k,
        "labeled_data": {},
        "predictions": [],
        "true_labels": [],
        "metrics": {}
    }
    
    return {"status": "initialized", "num_clusters": k}


@app.post("/external-metrics-hog/upload-group")
async def upload_group_hog(
    group_id: int = Form(...),
    label: str = Form(...),
    files: list[UploadFile] = File(...)
):
    global EXTERNAL_METRICS_HOG_STATE
    
    if EXTERNAL_METRICS_HOG_STATE is None:
        raise HTTPException(status_code=400, detail="No hay sesión inicializada")
    
    if group_id < 0 or group_id >= EXTERNAL_METRICS_HOG_STATE["num_clusters"]:
        raise HTTPException(status_code=400, detail=f"group_id inválido")
    
    if not files:
        raise HTTPException(status_code=400, detail="No se enviaron archivos")
    
    if group_id not in EXTERNAL_METRICS_HOG_STATE["labeled_data"]:
        EXTERNAL_METRICS_HOG_STATE["labeled_data"][group_id] = {
            "label": label.strip(),
            "images": []
        }
    
    results = []
    
    for file in files:
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(status_code=400, detail=f"Tipo no permitido")
        
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"Archivo demasiado grande")
        
        try:
            resized_bytes = reescalar_imagen_bytes(content)
            processed_bytes, vector, _ = process_image_and_extract_features(resized_bytes, "hog")
            
            vector_array = np.array(vector, dtype=float).reshape(1, -1)
            vector_normalizado = normalize(vector_array, norm='l2')[0]
            
            ext = ALLOWED_TYPES[file.content_type]
            image_id = uuid.uuid4().hex
            original_name = f"{image_id}{ext}"
            processed_name = f"{image_id}_hog.png"
            
            with open(os.path.join(ORIGINAL_DIR, original_name), "wb") as f:
                f.write(resized_bytes)
            with open(os.path.join(PROCESSED_DIR, processed_name), "wb") as f:
                f.write(processed_bytes)

            EXTERNAL_METRICS_HOG_STATE["labeled_data"][group_id]["images"].append({
                "filename": file.filename,
                "vector": vector_normalizado.tolist(),
                "true_label": label.strip(),
                "original_url": f"/files/originals/{original_name}",
                "processed_url": f"/files/processed/{processed_name}",
            })
            
            results.append({
                "filename": file.filename,
                "original_url": f"/files/originals/{original_name}",
                "processed_url": f"/files/processed/{processed_name}"
            })
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error: {exc}")
    
    return {
        "status": "ok",
        "group_id": group_id,
        "label": label,
        "num_images_uploaded": len(results)
    }


@app.post("/external-metrics-hog/calculate")
async def calculate_external_metrics_hog(capacities: str | None = Form(None)):
    global EXTERNAL_METRICS_HOG_STATE, CLUSTER_MODEL_HOG_LABELED
    
    if EXTERNAL_METRICS_HOG_STATE is None or not EXTERNAL_METRICS_HOG_STATE["labeled_data"]:
        raise HTTPException(status_code=400, detail="No hay datos etiquetados")
    
    all_vectors = []
    true_labels_list = []
    image_info = []
    
    for group_id in sorted(EXTERNAL_METRICS_HOG_STATE["labeled_data"].keys()):
        group_data = EXTERNAL_METRICS_HOG_STATE["labeled_data"][group_id]
        label = group_data["label"]
        for img_data in group_data["images"]:
            all_vectors.append(np.array(img_data["vector"]))
            true_labels_list.append(label)
            image_info.append({
                "filename": img_data.get("filename"),
                "true_label": label,
                "original_url": img_data.get("original_url"),
                "processed_url": img_data.get("processed_url"),
            })
    
    if len(all_vectors) == 0:
        raise HTTPException(status_code=400, detail="No hay imágenes")

    perm = np.random.permutation(len(all_vectors))
    all_vectors = [all_vectors[i] for i in perm]
    true_labels_list = [true_labels_list[i] for i in perm]
    image_info = [image_info[i] for i in perm]
    
    num_clusters = EXTERNAL_METRICS_HOG_STATE["num_clusters"]
    total_images = len(all_vectors)
    if capacities:
        capacities_list = parse_capacities(capacities)
        if len(capacities_list) != num_clusters:
            raise HTTPException(status_code=400, detail=f"Debe proporcionar {num_clusters} capacidades")
        capacities = capacities_list
    else:
        capacities = [max(total_images // num_clusters + 2, 5)] * num_clusters
    
    CLUSTER_MODEL_HOG_LABELED = LinksClusterCapacityOnline(capacities=capacities)
    
    predicted_labels_list = []
    image_urls = []
    
    for idx, vector in enumerate(all_vectors):
        try:
            cluster_id, _ = CLUSTER_MODEL_HOG_LABELED.predict_with_centroid(vector)
            predicted_labels_list.append(cluster_id)
        except:
            predicted_labels_list.append(0)
        
        image_urls.append({
            "filename": image_info[idx].get("filename", f"img_{idx}"),
            "true_label": image_info[idx].get("true_label", "unknown"),
            "predicted_cluster": predicted_labels_list[idx],
            "original_url": image_info[idx].get("original_url"),
            "processed_url": image_info[idx].get("processed_url"),
        })
    
    le_true = LabelEncoder()
    y_true_numeric = le_true.fit_transform(true_labels_list)
    y_pred_numeric = np.array(predicted_labels_list)
    
    ari = adjusted_rand_score(y_true_numeric, y_pred_numeric)
    ami = adjusted_mutual_info_score(y_true_numeric, y_pred_numeric)
    nmi = normalized_mutual_info_score(y_true_numeric, y_pred_numeric)
    
    dunn_index = CLUSTER_MODEL_HOG_LABELED.calculate_dunn_index()
    silhouette_coefficient = CLUSTER_MODEL_HOG_LABELED.calculate_silhouette_coefficient()
    
    clusters_visualization = {}
    for idx, cluster_id in enumerate(predicted_labels_list):
        if cluster_id not in clusters_visualization:
            clusters_visualization[cluster_id] = []
        
        clusters_visualization[cluster_id].append({
            "index": idx,
            "filename": image_urls[idx]["filename"],
            "true_label": image_urls[idx]["true_label"],
            "predicted_cluster": cluster_id,
            "original_url": image_urls[idx].get("original_url"),
            "processed_url": image_urls[idx].get("processed_url"),
        })
    
    EXTERNAL_METRICS_HOG_STATE["metrics"] = {
        "ARI": float(ari),
        "AMI": float(ami),
        "NMI": float(nmi),
        "dunn_index": float(dunn_index),
        "silhouette_coefficient": float(silhouette_coefficient),
    }
    
    return {
        "status": "ok",
        "external_metrics": {
            "ARI": round(ari, 4),
            "AMI": round(ami, 4),
            "NMI": round(nmi, 4)
        },
        "internal_metrics": {
            "dunn_index": round(dunn_index, 4),
            "silhouette_coefficient": round(silhouette_coefficient, 4)
        },
        "clusters": clusters_visualization,
        "summary": {
            "num_images": len(all_vectors),
            "num_clusters": num_clusters,
            "true_groups": len(set(true_labels_list)),
            "predicted_clusters": len(set(predicted_labels_list))
        }
    }


@app.delete("/external-metrics-hog/reset")
def reset_external_metrics_hog():
    global EXTERNAL_METRICS_HOG_STATE, CLUSTER_MODEL_HOG_LABELED
    EXTERNAL_METRICS_HOG_STATE = None
    CLUSTER_MODEL_HOG_LABELED = None
    return {"status": "reset", "message": "Sesión HOG reiniciada"}


# ==================== CNN CON ETIQUETAS (ARI/AMI/NMI) ====================

@app.post("/external-metrics-cnn/initialize")
async def initialize_external_metrics_cnn(num_clusters: int = Form(...)):
    global EXTERNAL_METRICS_CNN_STATE
    
    try:
        k = int(num_clusters)
        if k <= 0:
            raise ValueError("Debe ser > 0")
    except:
        raise HTTPException(status_code=400, detail="num_clusters inválido")
    
    EXTERNAL_METRICS_CNN_STATE = {
        "num_clusters": k,
        "labeled_data": {},
        "predictions": [],
        "true_labels": [],
        "metrics": {}
    }
    
    return {"status": "initialized", "num_clusters": k}


@app.post("/external-metrics-cnn/upload-group")
async def upload_group_cnn(
    group_id: int = Form(...),
    label: str = Form(...),
    files: list[UploadFile] = File(...)
):
    global EXTERNAL_METRICS_CNN_STATE
    
    if EXTERNAL_METRICS_CNN_STATE is None:
        raise HTTPException(status_code=400, detail="No hay sesión inicializada")
    
    if group_id < 0 or group_id >= EXTERNAL_METRICS_CNN_STATE["num_clusters"]:
        raise HTTPException(status_code=400, detail=f"group_id inválido")
    
    if not files:
        raise HTTPException(status_code=400, detail="No se enviaron archivos")
    
    if group_id not in EXTERNAL_METRICS_CNN_STATE["labeled_data"]:
        EXTERNAL_METRICS_CNN_STATE["labeled_data"][group_id] = {
            "label": label.strip(),
            "images": []
        }
    
    results = []
    
    for file in files:
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(status_code=400, detail=f"Tipo no permitido")
        
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"Archivo demasiado grande")
        
        try:
            resized_bytes = reescalar_imagen_bytes(content)
            processed_bytes, vector, _ = process_image_and_extract_features(resized_bytes, "cnn")
            
            vector_array = np.array(vector, dtype=float).reshape(1, -1)
            vector_normalizado = normalize(vector_array, norm='l2')[0]
            
            ext = ALLOWED_TYPES[file.content_type]
            image_id = uuid.uuid4().hex
            original_name = f"{image_id}{ext}"
            processed_name = f"{image_id}_cnn.png"
            
            with open(os.path.join(ORIGINAL_DIR, original_name), "wb") as f:
                f.write(resized_bytes)
            with open(os.path.join(PROCESSED_DIR, processed_name), "wb") as f:
                f.write(processed_bytes)

            EXTERNAL_METRICS_CNN_STATE["labeled_data"][group_id]["images"].append({
                "filename": file.filename,
                "vector": vector_normalizado.tolist(),
                "true_label": label.strip(),
                "original_url": f"/files/originals/{original_name}",
                "processed_url": f"/files/processed/{processed_name}",
            })
            
            results.append({
                "filename": file.filename,
                "original_url": f"/files/originals/{original_name}",
                "processed_url": f"/files/processed/{processed_name}"
            })
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error: {exc}")
    
    return {
        "status": "ok",
        "group_id": group_id,
        "label": label,
        "num_images_uploaded": len(results)
    }


@app.post("/external-metrics-cnn/calculate")
async def calculate_external_metrics_cnn(capacities: str | None = Form(None)):
    global EXTERNAL_METRICS_CNN_STATE, CLUSTER_MODEL_CNN_LABELED
    
    if EXTERNAL_METRICS_CNN_STATE is None or not EXTERNAL_METRICS_CNN_STATE["labeled_data"]:
        raise HTTPException(status_code=400, detail="No hay datos etiquetados")
    
    all_vectors = []
    true_labels_list = []
    image_info = []
    
    for group_id in sorted(EXTERNAL_METRICS_CNN_STATE["labeled_data"].keys()):
        group_data = EXTERNAL_METRICS_CNN_STATE["labeled_data"][group_id]
        label = group_data["label"]
        for img_data in group_data["images"]:
            all_vectors.append(np.array(img_data["vector"]))
            true_labels_list.append(label)
            image_info.append({
                "filename": img_data.get("filename"),
                "true_label": label,
                "original_url": img_data.get("original_url"),
                "processed_url": img_data.get("processed_url"),
            })
    
    if len(all_vectors) == 0:
        raise HTTPException(status_code=400, detail="No hay imágenes")

    perm = np.random.permutation(len(all_vectors))
    all_vectors = [all_vectors[i] for i in perm]
    true_labels_list = [true_labels_list[i] for i in perm]
    image_info = [image_info[i] for i in perm]
    
    num_clusters = EXTERNAL_METRICS_CNN_STATE["num_clusters"]
    total_images = len(all_vectors)
    if capacities:
        capacities_list = parse_capacities(capacities)
        if len(capacities_list) != num_clusters:
            raise HTTPException(status_code=400, detail=f"Debe proporcionar {num_clusters} capacidades")
        capacities = capacities_list
    else:
        capacities = [max(total_images // num_clusters + 2, 5)] * num_clusters
    
    CLUSTER_MODEL_CNN_LABELED = LinksClusterCapacityOnline(capacities=capacities)
    
    predicted_labels_list = []
    image_urls = []
    
    for idx, vector in enumerate(all_vectors):
        try:
            cluster_id, _ = CLUSTER_MODEL_CNN_LABELED.predict_with_centroid(vector)
            predicted_labels_list.append(cluster_id)
        except:
            predicted_labels_list.append(0)
        
        image_urls.append({
            "filename": image_info[idx].get("filename", f"img_{idx}"),
            "true_label": image_info[idx].get("true_label", "unknown"),
            "predicted_cluster": predicted_labels_list[idx],
            "original_url": image_info[idx].get("original_url"),
            "processed_url": image_info[idx].get("processed_url"),
        })
    
    le_true = LabelEncoder()
    y_true_numeric = le_true.fit_transform(true_labels_list)
    y_pred_numeric = np.array(predicted_labels_list)
    
    ari = adjusted_rand_score(y_true_numeric, y_pred_numeric)
    ami = adjusted_mutual_info_score(y_true_numeric, y_pred_numeric)
    nmi = normalized_mutual_info_score(y_true_numeric, y_pred_numeric)
    
    dunn_index = CLUSTER_MODEL_CNN_LABELED.calculate_dunn_index()
    silhouette_coefficient = CLUSTER_MODEL_CNN_LABELED.calculate_silhouette_coefficient()
    
    clusters_visualization = {}
    for idx, cluster_id in enumerate(predicted_labels_list):
        if cluster_id not in clusters_visualization:
            clusters_visualization[cluster_id] = []
        
        clusters_visualization[cluster_id].append({
            "index": idx,
            "filename": image_urls[idx]["filename"],
            "true_label": image_urls[idx]["true_label"],
            "predicted_cluster": cluster_id,
            "original_url": image_urls[idx].get("original_url"),
            "processed_url": image_urls[idx].get("processed_url"),
        })
    
    EXTERNAL_METRICS_CNN_STATE["metrics"] = {
        "ARI": float(ari),
        "AMI": float(ami),
        "NMI": float(nmi),
        "dunn_index": float(dunn_index),
        "silhouette_coefficient": float(silhouette_coefficient),
    }
    
    return {
        "status": "ok",
        "external_metrics": {
            "ARI": round(ari, 4),
            "AMI": round(ami, 4),
            "NMI": round(nmi, 4)
        },
        "internal_metrics": {
            "dunn_index": round(dunn_index, 4),
            "silhouette_coefficient": round(silhouette_coefficient, 4)
        },
        "clusters": clusters_visualization,
        "summary": {
            "num_images": len(all_vectors),
            "num_clusters": num_clusters,
            "true_groups": len(set(true_labels_list)),
            "predicted_clusters": len(set(predicted_labels_list))
        }
    }


@app.delete("/external-metrics-cnn/reset")
def reset_external_metrics_cnn():
    global EXTERNAL_METRICS_CNN_STATE, CLUSTER_MODEL_CNN_LABELED
    EXTERNAL_METRICS_CNN_STATE = None
    CLUSTER_MODEL_CNN_LABELED = None
    return {"status": "reset", "message": "Sesión CNN reiniciada"}
