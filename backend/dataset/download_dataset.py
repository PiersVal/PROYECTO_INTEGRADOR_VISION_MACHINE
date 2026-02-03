import os
import shutil
import random
import kagglehub

SEED = 42
MAX_IMGS = 200


def descargar_dataset_generico(
    dataset_id: str,
    clases: dict,
    dataset_num: int,
    max_imgs: int = MAX_IMGS,
    seed: int = SEED
):
    """
    Descarga un dataset de Kaggle y copia un número limitado de imágenes por clase
    directamente dentro de datos/dataset_X/imagenes_crudas/
    
    Parameters
    ----------
    dataset_id : str
        ID del dataset en Kaggle
    clases : dict
        {"nombre_clase_modelo": "nombre_carpeta_dataset"}
    dataset_num : int
        Número de dataset (1 o 2) para crear carpeta dat/dataset_X
    max_imgs : int
        Máximo de imágenes por clase
    seed : int
        Semilla para reproducibilidad
    """

    random.seed(seed)

    base_output = f"data/dataset_{dataset_num}/imagenes_crudas"
    os.makedirs(base_output, exist_ok=True)

    print(f"\n⬇ Descargando dataset: {dataset_id}")
    path = kagglehub.dataset_download(dataset_id)

    rutas = {}

    for root, dirs, _ in os.walk(path):
        for clase, carpeta in clases.items():
            if carpeta in dirs and clase not in rutas:

                origen = os.path.join(root, carpeta)
                destino = os.path.join(base_output, carpeta)
                os.makedirs(destino, exist_ok=True)

                imgs = [
                    f for f in os.listdir(origen)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]

                if len(imgs) == 0:
                    raise RuntimeError(f"No hay imágenes en {carpeta}")

                seleccion = random.sample(
                    imgs, min(max_imgs, len(imgs))
                )

                for img in seleccion:
                    shutil.copy2(
                        os.path.join(origen, img),
                        os.path.join(destino, img)
                    )

                print(f"📁 {carpeta}: {len(seleccion)} imágenes copiadas")
                rutas[clase] = destino

        if len(rutas) == len(clases):
            break

    if len(rutas) != len(clases):
        raise RuntimeError("No se encontraron todas las clases")

    return rutas

def descargar_dataset_1():
    clases_vehiculos = {
        "Motorcycles": "Motorcycles",
        "Planes": "Planes",
        "Ships": "Ships"
    }
    return descargar_dataset_generico(
        dataset_id="mohamedmaher5/vehicle-classification",
        clases=clases_vehiculos,
        dataset_num=1
    )


def descargar_dataset_2():
    clases_animales = {
        "cat": "gatto",
        "horse": "cavallo",
        "elephant": "elefante"
    }
    return descargar_dataset_generico(
        dataset_id="alessiocorrado99/animals10",
        clases=clases_animales,
        dataset_num=2
    )
