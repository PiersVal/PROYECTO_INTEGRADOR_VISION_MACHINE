from genera_dataset.dataset_rps import generar_datos_rps as generar_dataset_rps
from genera_dataset.dataset_animales import generar_datos_animales as generar_dataset_animales


def main():
    print("Generando dataset RPS...")
    generar_dataset_rps()
    print("Dataset RPS generado\n")

    print("Generando dataset de animales (YOLO + pipeline clásico)...")
    generar_dataset_animales()
    print("Dataset de animales generado correctamente")


if __name__ == "__main__":
    main()