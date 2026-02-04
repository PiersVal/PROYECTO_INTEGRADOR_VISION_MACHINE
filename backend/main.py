from dataset.download_dataset import descargar_dataset_1, descargar_dataset_2
from preprocessing.preprocesamiento import procesar_imagenes
import os
## End of backend/main.py
## Start of backend/main.py


def main():
    # Descargar datasets
    print("⬇ Descargando dataset 1...")
    descargar_dataset_1()

    print("⬇ Descargando dataset 2...")
    descargar_dataset_2()

    # Configuración de preprocesamiento
    datasets_config = {
        1: {
            "Motorcycles": {"class_name": "motorcycle", "prefix": "moto"},
            "Planes": {"class_name": "airplane", "prefix": "avion"},
            "Ships": {"class_name": "boat", "prefix": "barco"}
        },
        2: {
            "gatto": {"class_name": "cat", "prefix": "gato"},
            "cavallo": {"class_name": "horse", "prefix": "caballo"},
            "elefante": {"class_name": "elephant", "prefix": "elefante"}
        }
    }

    # Procesar todos los datasets
    print("\n" + "="*50)
    print("INICIANDO PREPROCESAMIENTO")
    print("="*50)
    
    for dataset_num in [1, 2]:
        clases = datasets_config[dataset_num]
        
        for carpeta, cfg in clases.items():
            print(f"\n📁 Procesando Dataset {dataset_num} - {carpeta}")
            
            # Rutas
            carpeta_entrada = os.path.join("data", f"dataset_{dataset_num}", "imagenes_crudas", carpeta)
            base_salida = os.path.join("data", f"dataset_{dataset_num}", "imagenes_procesadas", carpeta)
            
            carpeta_gris = os.path.join(base_salida, "grises")
            carpeta_contraste = os.path.join(base_salida, "contraste")
            carpeta_binaria = os.path.join(base_salida, "binaria")
            
            # Verificar si existe carpeta de entrada
            if not os.path.exists(carpeta_entrada):
                print(f"⚠ No se encontró: {carpeta_entrada}")
                continue
            
            # Procesar
            total = procesar_imagenes(
                carpeta_entrada=carpeta_entrada,
                carpeta_salida_gris=carpeta_gris,
                carpeta_salida_contraste=carpeta_contraste,
                carpeta_salida_binaria=carpeta_binaria
            )
            
            print(f"✅ Procesadas: {total} imágenes")
    
    print("\n" + "="*50)
    print("PREPROCESAMIENTO COMPLETADO")
    print("="*50)

if __name__ == "__main__":
    main()
