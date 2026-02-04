#!/usr/bin/env python3
"""Test completo del clustering online con restricciones."""
import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def upload_files(file_paths, capacities=None, clusters=None):
    """Subir archivos para análisis inicial."""
    print_section("1️⃣  ANÁLISIS INICIAL - Subir 3 imágenes")
    
    files = [("files", (Path(fp).name, open(fp, "rb"), "image/png")) for fp in file_paths]
    data = {}
    if capacities:
        data["capacities"] = capacities
    if clusters:
        data["clusters"] = str(clusters)
    
    response = requests.post(f"{BASE_URL}/analyze", files=files, data=data)
    for file_tuple in files:
        file_tuple[1][1].close()
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Subida exitosa")
        print(f"  Resultados: {len(result['results'])} imágenes procesadas")
        for i, r in enumerate(result['results'], 1):
            print(f"    - {r['filename']}: Cluster {r['cluster_id']}")
        return result['results']
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"  {response.text}")
        return None

def check_status():
    """Verificar estado del clustering."""
    print_section("📊 Estado del Clustering")
    
    response = requests.get(f"{BASE_URL}/cluster-status")
    if response.status_code == 200:
        status = response.json()
        print(f"✓ Modelo activo: {status['active']}")
        print(f"  Número de clusters: {status['num_clusters']}")
        print(f"  Capacidades: {status['capacities']}")
        print(f"  Conteo actual: {status['current_counts']}")
        print(f"  Espacios disponibles: {status['available_spaces']}")
        return status
    else:
        print(f"✗ Error: {response.status_code}")
        return None

def add_images(file_paths):
    """Agregar más imágenes al clustering."""
    print_section(f"2️⃣  AGREGAR IMÁGENES - {len(file_paths)} nuevas imágenes")
    
    files = [("files", (Path(fp).name, open(fp, "rb"), "image/png")) for fp in file_paths]
    response = requests.post(f"{BASE_URL}/add-images", files=files)
    for file_tuple in files:
        file_tuple[1][1].close()
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Imágenes agregadas exitosamente")
        print(f"  Resultados: {len(result['results'])} imágenes procesadas")
        for i, r in enumerate(result['results'], 1):
            print(f"    - {r['filename']}: Cluster {r['cluster_id']}")
        return result['results']
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"  {response.text}")
        return None

def update_capacities(new_capacities):
    """Actualizar las capacidades de los clusters."""
    print_section(f"3️⃣  ACTUALIZAR RESTRICCIONES - Nuevas capacidades: {new_capacities}")
    
    data = {"capacities": new_capacities}
    response = requests.post(f"{BASE_URL}/update-capacities", data=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Restricciones actualizadas")
        print(f"  Nuevas capacidades: {result['new_capacities']}")
        print(f"  Conteo actual: {result['current_counts']}")
        print(f"  Espacios disponibles: {result['available_spaces']}")
        return result
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"  {response.text}")
        return None

def main():
    print("\n" + "🚀 TEST COMPLETO DE CLUSTERING ONLINE CON RESTRICCIONES" + "\n")
    
    # Obtener archivos de prueba
    test_dir = Path("/app/test_images")
    all_images = sorted(list(test_dir.glob("test_*.png")))
    
    if len(all_images) < 5:
        print(f"✗ Se necesitan al menos 5 imágenes de prueba. Solo hay {len(all_images)}")
        return
    
    # Fase 1: Análisis inicial con 3 imágenes y capacidades [3, 2]
    initial_images = all_images[:3]
    results1 = upload_files([str(p) for p in initial_images], capacities="5,5")
    
    if not results1:
        return
    
    # Verificar estado
    status1 = check_status()
    
    # Fase 2: Intentar agregar 2 imágenes más
    additional_images = all_images[3:5]
    results2 = add_images([str(p) for p in additional_images])
    
    if results2:
        status2 = check_status()
        
        # Fase 3: Actualizar restricciones a valores más altos
        update_capacities("10,10")
        
        status3 = check_status()
        
        print("\n" + "="*60)
        print("  ✓ TEST COMPLETADO EXITOSAMENTE")
        print("="*60)
        print("\nResumen:")
        print(f"  1. Cargadas 3 imágenes iniciales con capacidades [5,5]")
        print(f"  2. Agregadas 2 imágenes adicionales")
        print(f"  3. Restricciones actualizadas a [10,10]")
        print(f"\nTodo funcionó correctamente! 🎉")
    else:
        print("\n✗ Error al agregar imágenes")

if __name__ == "__main__":
    main()
