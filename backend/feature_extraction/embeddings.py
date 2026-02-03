# src/preprocesamiento/extract_embeddings.py
import os
from pathlib import Path
import time
import random
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# Número máximo de imágenes por clase
MAX_IMGS = 100

def extraer_embeddings(dataset_path, clases, max_imgs=MAX_IMGS, img_size=(256,256), batch_size=32):
    """
    dataset_path : Path o str : carpeta base 'data/dataset_X/'
    clases : dict : {'nombre_carpeta_cruda': {'prefix': 'prefijo', 'class_name': 'YOLO'}}
    """
    dataset_path = Path(dataset_path)
    all_images = []
    class_labels = []

    # ----------------------------
    # Recopilar imágenes y etiquetas
    # ----------------------------
    for carpeta, cfg in clases.items():
        input_dir = dataset_path / "imagenes_crudas" / carpeta
        if not input_dir.exists():
            print(f"⚠ Carpeta no encontrada: {input_dir}")
            continue
        image_extensions = ("*.jpg", "*.jpeg", "*.png")
        imgs = [p for ext in image_extensions for p in input_dir.glob(ext)]
        # Seleccionar aleatoriamente max_imgs
        imgs_sel = random.sample(imgs, min(max_imgs, len(imgs)))

        all_images.extend(imgs_sel)
        class_labels.extend([cfg['prefix']] * len(imgs_sel))

    print(f"\nTotal imágenes a procesar para embeddings: {len(all_images)}")

    # ----------------------------
    # Cargar modelo ResNet50
    # ----------------------------
    print("\nCargando modelo ResNet50 preentrenado...")
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(img_size[0], img_size[1],3))
    print("Modelo cargado ✅")

    # ----------------------------
    # Listas para almacenar resultados
    # ----------------------------
    embeddings = []
    image_paths = []

    start_time = time.time()
    print("\nComenzando extracción de embeddings...")

    # ----------------------------
    # Procesar por lotes para mayor eficiencia
    # ----------------------------
    for i in range(0, len(all_images), batch_size):
        batch_paths = all_images[i:i + batch_size]
        batch_images = []

        for img_path in batch_paths:
            try:
                # Cargar y preprocesar imagen
                img = load_img(img_path, target_size=img_size)
                x = img_to_array(img)
                x = preprocess_input(x)
                batch_images.append(x)
            except Exception as e:
                print(f"Error cargando {img_path}: {e}")
                continue

        if batch_images:  # Si hay imágenes válidas en el batch
            # Procesar batch completo
            batch_array = np.array(batch_images)
            batch_embeddings = model.predict(batch_array, verbose=0)

            # Agregar resultados
            embeddings.extend(batch_embeddings)

            # Guardar rutas correspondientes
            image_paths.extend([str(p) for p in batch_paths])

        # Mostrar progreso cada 5 lotes
        if (i // batch_size) % 5 == 0 and i > 0:
            elapsed = time.time() - start_time
            print(f"  Procesadas {min(i + batch_size, len(all_images))}/{len(all_images)} imágenes "
                  f"({elapsed:.1f}s)")

    # ----------------------------
    # Convertir a arrays numpy
    # ----------------------------
    embeddings = np.array(embeddings)
    class_labels = np.array(class_labels)
    image_paths = np.array(image_paths)

    print(f"\nExtracción completada!")
    print(f"════════════════════════════════════════")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Número de imágenes procesadas: {len(embeddings)}")
    print(f"Tiempo total: {time.time() - start_time:.2f} segundos")

    # ----------------------------
    # Distribución por clases
    # ----------------------------
    print(f"\nDistribución por clases:")
    unique_classes, counts = np.unique(class_labels, return_counts=True)
    for clase, count in zip(unique_classes, counts):
        print(f"  {clase}: {count} imágenes")

    # ----------------------------
    # Guardar resultados
    # ----------------------------
    print(f"\nGuardando resultados...")
    output_dir = dataset_path / "embeddings"
    output_dir.mkdir(exist_ok=True)

    # Guardar embeddings
    np.save(output_dir / "Embeddings.npy", embeddings)

    # Guardar etiquetas
    np.save(output_dir / "Labels.npy", class_labels)

    # Guardar rutas de imágenes
    np.save(output_dir / "image_paths.npy", image_paths)

    # Crear archivo CSV con metadatos
    metadata = pd.DataFrame({
        'image_path': image_paths,
        'class': class_labels
    })
    metadata.to_csv(output_dir / "metadata.csv", index=False)

    print(f"Resultados guardados en: {output_dir}")
    print(f"   - embeddings.npy (shape: {embeddings.shape})")
    print(f"   - class_labels.npy ({len(class_labels)} etiquetas)")
    print(f"   - image_paths.npy ({len(image_paths)} rutas)")
    print(f"   - metadata.csv")

    # ----------------------------
    # Información adicional
    # ----------------------------
    print(f"\nInformación de embeddings:")
    print(f"   Dimensión por imagen: {embeddings.shape[1]} features")
    print(f"   Memoria aprox: {(embeddings.nbytes / 1024**2):.2f} MB")

    return embeddings, class_labels, image_paths

