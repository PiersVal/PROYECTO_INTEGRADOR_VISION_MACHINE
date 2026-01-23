import os
import shutil
import kagglehub
import random

CLASES = {
    "cat": "gatto",
    "horse": "cavallo",
    "elephant": "elefante"
}

BASE_OUTPUT = "dataset/animales"
MAX_IMGS = 100
SEED = 42

def descargar_dataset():
    
    random.seed(SEED)

    path = kagglehub.dataset_download("alessiocorrado99/animals10")
    os.makedirs(BASE_OUTPUT, exist_ok=True)

    rutas = {}

    for root, dirs, _ in os.walk(path):
        for clase, carpeta in CLASES.items():
            if carpeta in dirs:
                origen = os.path.join(root, carpeta)
                destino = os.path.join(BASE_OUTPUT, carpeta)
                os.makedirs(destino, exist_ok=True)

                # imágenes válidas
                imgs = [
                    f for f in os.listdir(origen)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]

                if len(imgs) == 0:
                    raise RuntimeError(f"No hay imágenes en {carpeta}")

                seleccion = random.sample(
                    imgs,
                    min(MAX_IMGS, len(imgs))
                )

                for img in seleccion:
                    shutil.copy2(
                        os.path.join(origen, img),
                        os.path.join(destino, img)
                    )

                print(f"{carpeta}: {len(seleccion)} imágenes copiadas")

                rutas[clase] = destino

        if len(rutas) == len(CLASES):
            break

    if len(rutas) != len(CLASES):
        raise RuntimeError("No se encontraron todas las carpetas")

    return rutas