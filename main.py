from src.features.procesamiento_vehiculos import procesar_datasets as generar_dataset_vehiculos

def main():

    print("Generando dataset de Vehiculos")
    generar_dataset_vehiculos()
    print("Dataset de vehiculos generado correctamente")

if __name__ == "__main__":
    main()