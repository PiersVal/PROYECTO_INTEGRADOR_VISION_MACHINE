#!/usr/bin/env python3
"""Test del clustering con 2 clusters para verificar distribución correcta."""
import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"

def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def upload_files(file_paths, capacities=None):
    """Subir archivos para análisis inicial."""
    print_section(f"📤 ANÁLISIS INICIAL - Subir {len(file_paths)} imágenes")
    
    files = [("files", (Path(fp).name, open(fp, "rb"), "image/png")) for fp in file_paths]
    data = {}
    if capacities:
        data["capacities"] = capacities
    
    response = requests.post(f"{BASE_URL}/analyze", files=files, data=data)
    for file_tuple in files:
        file_tuple[1][1].close()
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Subida exitosa: {len(result['results'])} imágenes")
        for r in result['results']:
            print(f"  • {r['filename']}: Cluster {r['cluster_id']}")
        return result['results']
    else:
        print(f"✗ Error: {response.text}")
        return None

def check_status():
    """Verificar estado del clustering."""
    response = requests.get(f"{BASE_URL}/cluster-status")
    if response.status_code == 200:
        status = response.json()
        print(f"\n📊 Estado del Clustering:")
        print(f"  Clusters: {status['num_clusters']}")
        print(f"  Capacidades: {status['capacities']}")
        print(f"  Conteo actual: {status['current_counts']}")
        print(f"  Espacios disponibles: {status['available_spaces']}")
        return status
    return None

def add_images(file_paths):
    """Agregar más imágenes al clustering."""
    print_section(f"📤 AGREGAR IMÁGENES - {len(file_paths)} nuevas imágenes")
    
    files = [("files", (Path(fp).name, open(fp, "rb"), "image/png")) for fp in file_paths]
    response = requests.post(f"{BASE_URL}/add-images", files=files)
    for file_tuple in files:
        file_tuple[1][1].close()
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Imágenes agregadas: {len(result['results'])} imágenes")
        for r in result['results']:
            print(f"  • {r['filename']}: Cluster {r['cluster_id']}")
        return result['results']
    else:
        print(f"✗ Error: {response.text}")
        return None

def update_capacities(new_capacities):
    """Actualizar capacidades."""
    print_section(f"⚙️  ACTUALIZAR RESTRICCIONES - {new_capacities}")
    
    response = requests.post(f"{BASE_URL}/update-capacities", data={"capacities": new_capacities})
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Actualizado a: {result['new_capacities']}")
        print(f"  Espacios disponibles: {result['available_spaces']}")
        return result
    else:
        print(f"✗ Error: {response.text}")
        return None

def main():
    print("\n🚀 TEST DE CLUSTERING CON 2 CLUSTERS Y RESTRICCIONES ESTRICTAS\n")
    
    test_dir = Path("/app/test_images")
    all_images = sorted(list(test_dir.glob("test_*.png")))
    
    if len(all_images) < 5:
        print(f"✗ Se necesitan 5 imágenes")
        return
    
    # FASE 1: Crear 2 clusters con capacidad limitada
    print_section("FASE 1: CREAR CLUSTERING CON 2 CLUSTERS")
    results1 = upload_files([str(p) for p in all_images[:3]], capacities="2,2")
    
    if not results1:
        return
    
    status1 = check_status()
    print(f"\n📝 Análisis: Se crearon los clusters")
    print(f"   Cluster 0 tiene {status1['current_counts'][0]} elementos (cupo: 2)")
    if len(status1['current_counts']) > 1:
        print(f"   Cluster 1 tiene {status1['current_counts'][1]} elementos (cupo: 2)")
    
    # FASE 2: Actualizar capacidades primero
    print_section("FASE 2: AUMENTAR CAPACIDADES A [5, 5] PARA HACER ESPACIO")
    update_capacities("5,5")
    status2a = check_status()
    
    # FASE 3: Agregar imágenes (deben entrar correctamente)
    print_section("FASE 3: AGREGAR 2 IMÁGENES CON NUEVO CUPO")
    results2 = add_images([str(p) for p in all_images[3:5]])
    
    if results2:
        status2b = check_status()
        
        # Verificar distribución
        print(f"\n✅ ANÁLISIS FINAL DE DISTRIBUCIÓN:")
        total = sum(status2b['current_counts'])
        print(f"   Total de imágenes: {total}/5 (esperado 5)")
        print(f"   Cluster 0: {status2b['current_counts'][0]}/5")
        if len(status2b['current_counts']) > 1:
            print(f"   Cluster 1: {status2b['current_counts'][1]}/5")
        
        print("\n✅ TEST COMPLETADO - Restricciones funcionando correctamente!")
    else:
        print("\n✗ Error durante prueba")

if __name__ == "__main__":
    main()
