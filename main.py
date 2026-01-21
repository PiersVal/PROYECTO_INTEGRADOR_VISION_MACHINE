from genera_dataset.dataset_animales import generar_datos_animales as generar_dataset_animales


def main():

    print("Generando dataset de animales (YOLO + pipeline clásico)...")
    generar_dataset_animales()
    print("Dataset de animales generado correctamente")


if __name__ == "__main__":
    main()