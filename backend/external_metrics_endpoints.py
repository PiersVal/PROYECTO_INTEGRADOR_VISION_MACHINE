# Este archivo contiene código para generar endpoints de métricas externas
# Se utilizará para copiar y pegar en main.py

# Template para cada método (Hu, Zernike, SIFT, HOG, CNN)

TEMPLATE = """
# ==================== {METHOD_NAME} CON ETIQUETAS (ARI/AMI/NMI) ====================

@app.post("/external-metrics-{method_lower}/initialize")
async def initialize_external_metrics_{method_lower}(num_clusters: int = Form(...)):
    global EXTERNAL_METRICS_{METHOD_UPPER}_STATE
    
    try:
        k = int(num_clusters)
        if k <= 0:
            raise ValueError("Debe ser > 0")
    except:
        raise HTTPException(status_code=400, detail="num_clusters inválido")
    
    EXTERNAL_METRICS_{METHOD_UPPER}_STATE = {{
        "num_clusters": k,
        "labeled_data": {{}},
        "predictions": [],
        "true_labels": [],
        "metrics": {{}}
    }}
    
    return {{
        "status": "ok",
        "num_clusters": k,
        "ready_for_upload": True,
        "message": f"Listo para subir imágenes {METHOD_NAME} en {{k}} grupos"
    }}


@app.post("/external-metrics-{method_lower}/upload-group")
async def upload_group_images_{method_lower}(
    group_id: int = Form(...),
    label: str = Form(...),
    files: List[UploadFile] = File(...)
):
    global EXTERNAL_METRICS_{METHOD_UPPER}_STATE
    
    if EXTERNAL_METRICS_{METHOD_UPPER}_STATE is None:
        raise HTTPException(status_code=400, detail="No hay sesión inicializada. Inicia primero.")
    
    if group_id < 0 or group_id >= EXTERNAL_METRICS_{METHOD_UPPER}_STATE["num_clusters"]:
        raise HTTPException(status_code=400, detail=f"group_id debe estar entre 0 y {{EXTERNAL_METRICS_{METHOD_UPPER}_STATE['num_clusters']-1}}")
    
    if not files:
        raise HTTPException(status_code=400, detail="No se enviaron archivos")
    
    if not label or len(label.strip()) == 0:
        raise HTTPException(status_code=400, detail="label vacía")
    
    if group_id not in EXTERNAL_METRICS_{METHOD_UPPER}_STATE["labeled_data"]:
        EXTERNAL_METRICS_{METHOD_UPPER}_STATE["labeled_data"][group_id] = {{
            "label": label.strip(),
            "images": []
        }}
    
    results = []
    
    for file in files:
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(status_code=400, detail=f"Tipo no permitido: {{file.content_type}}")
        
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"Archivo demasiado grande")
        
        try:
            resized_bytes = reescalar_imagen_bytes(content)
            {PROCESSING_CODE}
            
            vector_array = np.array(vector, dtype=float).reshape(1, -1)
            vector_normalizado = normalize(vector_array, norm='l2')[0]
            
            ext = ALLOWED_TYPES[file.content_type]
            image_id = uuid.uuid4().hex
            original_name = f"{{image_id}}{{ext}}"
            processed_name = f"{{image_id}}_{method_lower}.png"
            
            original_path = os.path.join(ORIGINAL_DIR, original_name)
            processed_path = os.path.join(PROCESSED_DIR, processed_name)
            
            with open(original_path, "wb") as f:
                f.write(resized_bytes)
            with open(processed_path, "wb") as f:
                f.write(processed_bytes)

            image_data = {{
                "filename": file.filename,
                "vector": vector_normalizado.tolist(),
                "true_label": label.strip(),
                "original_url": f"/files/originals/{{original_name}}",
                "processed_url": f"/files/processed/{{processed_name}}",
            }}
            
            EXTERNAL_METRICS_{METHOD_UPPER}_STATE["labeled_data"][group_id]["images"].append(image_data)
            
            results.append({{
                "filename": file.filename,
                "original_url": f"/files/originals/{{original_name}}",
                "processed_url": f"/files/processed/{{processed_name}}"
            }})
            
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error procesando {{file.filename}}: {{exc}}")
    
    return {{
        "status": "ok",
        "group_id": group_id,
        "label": label,
        "num_images_uploaded": len(results),
        "total_images_in_group": len(EXTERNAL_METRICS_{METHOD_UPPER}_STATE["labeled_data"][group_id]["images"]),
        "uploaded": results
    }}


@app.post("/external-metrics-{method_lower}/calculate")
async def calculate_external_metrics_{method_lower}(capacities: str | None = Form(None)):
    global EXTERNAL_METRICS_{METHOD_UPPER}_STATE, CLUSTER_MODEL_{METHOD_UPPER}_LABELED
    
    if EXTERNAL_METRICS_{METHOD_UPPER}_STATE is None or not EXTERNAL_METRICS_{METHOD_UPPER}_STATE["labeled_data"]:
        raise HTTPException(status_code=400, detail="No hay datos etiquetados")
    
    all_vectors = []
    true_labels_list = []
    image_info = []
    
    for group_id in sorted(EXTERNAL_METRICS_{METHOD_UPPER}_STATE["labeled_data"].keys()):
        group_data = EXTERNAL_METRICS_{METHOD_UPPER}_STATE["labeled_data"][group_id]
        label = group_data["label"]
        for img_data in group_data["images"]:
            vector = np.array(img_data["vector"])
            all_vectors.append(vector)
            true_labels_list.append(label)
            image_info.append({{
                "filename": img_data.get("filename"),
                "true_label": label,
                "original_url": img_data.get("original_url"),
                "processed_url": img_data.get("processed_url"),
            }})
    
    if len(all_vectors) == 0:
        raise HTTPException(status_code=400, detail="No hay imágenes para analizar")

    perm = np.random.permutation(len(all_vectors))
    all_vectors = [all_vectors[i] for i in perm]
    true_labels_list = [true_labels_list[i] for i in perm]
    image_info = [image_info[i] for i in perm]
    
    num_clusters = EXTERNAL_METRICS_{METHOD_UPPER}_STATE["num_clusters"]
    total_images = len(all_vectors)
    if capacities:
        try:
            capacities_list = parse_capacities(capacities)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"capacities inválido: {{exc}}")
        if len(capacities_list) != num_clusters:
            raise HTTPException(status_code=400, detail=f"Debe proporcionar {{num_clusters}} capacidades")
        capacities = capacities_list
    else:
        capacity_per_cluster = max(total_images // num_clusters + 2, 5)
        capacities = [capacity_per_cluster] * num_clusters
    
    CLUSTER_MODEL_{METHOD_UPPER}_LABELED = LinksClusterCapacityOnline(capacities=capacities)
    
    predicted_labels_list = []
    image_urls = []
    
    for idx, vector in enumerate(all_vectors):
        try:
            cluster_id, _ = CLUSTER_MODEL_{METHOD_UPPER}_LABELED.predict_with_centroid(vector)
            predicted_labels_list.append(cluster_id)
        except:
            predicted_labels_list.append(0)
        
        image_data = image_info[idx] if idx < len(image_info) else {{"filename": f"img_{{idx}}"}}
        image_urls.append({{
            "filename": image_data.get("filename", f"img_{{idx}}"),
            "true_label": image_data.get("true_label", "unknown"),
            "predicted_cluster": predicted_labels_list[idx],
            "original_url": image_data.get("original_url"),
            "processed_url": image_data.get("processed_url"),
        }})
    
    le_true = LabelEncoder()
    y_true_numeric = le_true.fit_transform(true_labels_list)
    y_pred_numeric = np.array(predicted_labels_list)
    
    ari = adjusted_rand_score(y_true_numeric, y_pred_numeric)
    ami = adjusted_mutual_info_score(y_true_numeric, y_pred_numeric)
    nmi = normalized_mutual_info_score(y_true_numeric, y_pred_numeric)
    
    dunn_index = CLUSTER_MODEL_{METHOD_UPPER}_LABELED.calculate_dunn_index()
    silhouette_coefficient = CLUSTER_MODEL_{METHOD_UPPER}_LABELED.calculate_silhouette_coefficient()
    
    clusters_visualization = {{}}
    for idx, cluster_id in enumerate(predicted_labels_list):
        if cluster_id not in clusters_visualization:
            clusters_visualization[cluster_id] = []
        
        clusters_visualization[cluster_id].append({{
            "index": idx,
            "filename": image_urls[idx]["filename"],
            "true_label": image_urls[idx]["true_label"],
            "predicted_cluster": cluster_id,
            "original_url": image_urls[idx].get("original_url"),
            "processed_url": image_urls[idx].get("processed_url"),
        }})
    
    EXTERNAL_METRICS_{METHOD_UPPER}_STATE["metrics"] = {{
        "ARI": float(ari),
        "AMI": float(ami),
        "NMI": float(nmi),
        "dunn_index": float(dunn_index),
        "silhouette_coefficient": float(silhouette_coefficient),
        "num_images": len(all_vectors),
        "num_clusters": num_clusters,
        "true_labels_count": len(set(true_labels_list)),
        "predicted_labels_count": len(set(predicted_labels_list))
    }}
    
    EXTERNAL_METRICS_{METHOD_UPPER}_STATE["true_labels"] = true_labels_list
    EXTERNAL_METRICS_{METHOD_UPPER}_STATE["predictions"] = predicted_labels_list
    EXTERNAL_METRICS_{METHOD_UPPER}_STATE["image_info"] = image_info
    EXTERNAL_METRICS_{METHOD_UPPER}_STATE["clusters_visualization"] = clusters_visualization
    
    return {{
        "status": "ok",
        "external_metrics": {{
            "ARI": round(ari, 4),
            "AMI": round(ami, 4),
            "NMI": round(nmi, 4)
        }},
        "internal_metrics": {{
            "dunn_index": round(dunn_index, 4),
            "silhouette_coefficient": round(silhouette_coefficient, 4)
        }},
        "clusters": clusters_visualization,
        "summary": {{
            "num_images": len(all_vectors),
            "num_clusters": num_clusters,
            "true_groups": len(set(true_labels_list)),
            "predicted_clusters": len(set(predicted_labels_list))
        }}
    }}


@app.get("/external-metrics-{method_lower}/status")
def get_external_metrics_status_{method_lower}():
    global EXTERNAL_METRICS_{METHOD_UPPER}_STATE
    
    if EXTERNAL_METRICS_{METHOD_UPPER}_STATE is None:
        return {{"initialized": False}}
    
    group_summary = {{}}
    for group_id, data in EXTERNAL_METRICS_{METHOD_UPPER}_STATE["labeled_data"].items():
        group_summary[f"grupo_{{group_id}}"] = {{
            "label": data["label"],
            "num_images": len(data["images"])
        }}
    
    return {{
        "initialized": True,
        "num_clusters": EXTERNAL_METRICS_{METHOD_UPPER}_STATE["num_clusters"],
        "groups_with_data": group_summary,
        "metrics": EXTERNAL_METRICS_{METHOD_UPPER}_STATE.get("metrics", {{}}),
        "total_images": sum(
            len(data["images"]) 
            for data in EXTERNAL_METRICS_{METHOD_UPPER}_STATE["labeled_data"].values()
        )
    }}


@app.delete("/external-metrics-{method_lower}/reset")
def reset_external_metrics_{method_lower}():
    global EXTERNAL_METRICS_{METHOD_UPPER}_STATE, CLUSTER_MODEL_{METHOD_UPPER}_LABELED
    EXTERNAL_METRICS_{METHOD_UPPER}_STATE = None
    CLUSTER_MODEL_{METHOD_UPPER}_LABELED = None
    return {{"status": "reset", "message": "Sesión de métricas {METHOD_NAME} reiniciada"}}

"""

# Configuración para cada método
METHODS = {
    "hu": {
        "METHOD_NAME": "Hu",
        "METHOD_UPPER": "HU",
        "method_lower": "hu",
        "PROCESSING_CODE": """momentos_hu = calcular_momentos_hu(resized_bytes)
            processed_bytes = procesar_imagen_bytes(resized_bytes)
            vector = momentos_hu"""
    },
    "zernike": {
        "METHOD_NAME": "Zernike",
        "METHOD_UPPER": "ZERNIKE",
        "method_lower": "zernike",
        "PROCESSING_CODE": """momentos_zernike = calcular_momentos_zernike(resized_bytes)
            processed_bytes = procesar_imagen_bytes(resized_bytes)
            vector = [m for m in momentos_zernike.values()]"""
    },
    "sift": {
        "METHOD_NAME": "SIFT",
        "METHOD_UPPER": "SIFT",
        "method_lower": "sift",
        "PROCESSING_CODE": """processed_bytes, descriptores_sift = procesar_sift_con_descriptores(resized_bytes)
            if not descriptores_sift:
                raise ValueError("No se pudieron extraer características SIFT")
            vector = descriptores_sift"""
    },
    "hog": {
        "METHOD_NAME": "HOG",
        "METHOD_UPPER": "HOG",
        "method_lower": "hog",
        "PROCESSING_CODE": """processed_bytes, descriptores_hog = procesar_hog_con_descriptores(resized_bytes)
            if not descriptores_hog:
                raise ValueError("No se pudieron extraer características HOG")
            vector = descriptores_hog"""
    },
    "cnn": {
        "METHOD_NAME": "CNN",
        "METHOD_UPPER": "CNN",
        "method_lower": "cnn",
        "PROCESSING_CODE": """processed_bytes, descriptores_cnn = procesar_cnn_con_descriptores(resized_bytes)
            if not descriptores_cnn:
                raise ValueError("No se pudieron extraer características CNN")
            vector = descriptores_cnn"""
    }
}

# Generar código para todos los métodos
print("# COPIAR EL SIGUIENTE CÓDIGO AL FINAL DE main.py\n")
print("# " + "="*80)
for method_key, method_config in METHODS.items():
    code = TEMPLATE.format(**method_config)
    print(code)
    print("# " + "="*80)
