from src.features.procesamiento_vehiculos import procesar_datasets as generar_dataset_vehiculos
from src.features.procesamiento_animales import procesar_datasets as generar_dataset_animales


def main():

    print("Generando dataset de Vehiculos")
    generar_dataset_vehiculos()
    print("Dataset de vehiculos generado correctamente")

    print("Generando dataset de Animales")
    generar_dataset_animales()
    print("Dataset de animales generado correctamente")


if __name__ == "__main__":
    main()
    