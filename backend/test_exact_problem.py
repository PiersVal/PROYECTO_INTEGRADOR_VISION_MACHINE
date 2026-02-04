#!/usr/bin/env python3
"""Test que replica exactamente el problema: 10 imágenes, 5,5 -> 5,8 -> agregar 3 más"""
import requests
from pathlib import Path

BASE_URL = "http://localhost:8000"

def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def upload_files(file_paths, capacities):
    """Subir archivos."""
    files = [("files", (Path(fp).name, open(fp, "rb"), "image/png")) for fp in file_paths]
    data = {"capacities": capacities}
    response = requests.post(f"{BASE_URL}/analyze", files=files, data=data)
    for file_tuple in files:
        file_tuple[1][1].close()
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ {len(result['results'])} imágenes procesadas")
        for r in result['results']:
            print(f"  • {r['filename']}: Cluster {r['cluster_id']}")
        return result['results']
    else:
        print(f"✗ Error: {response.text}")
        return None

def check_status():
    """Ver estado del clustering."""
    response = requests.get(f"{BASE_URL}/cluster-status")
    if response.status_code == 200:
        status = response.json()
        print(f"\n📊 Estado Actual:")
        for i, (cap, count) in enumerate(zip(status['capacities'], status['current_counts'])):
            avail = cap - count
            print(f"   Cluster {i}: {count}/{cap} (cupo disponible: {avail})")
        return status
    return None

def update_capacities(caps):
    """Actualizar capacidades."""
    response = requests.post(f"{BASE_URL}/update-capacities", data={"capacities": caps})
    if response.status_code == 200:
        print(f"✓ Capacidades actualizadas a: {caps}")
        return response.json()
    else:
        print(f"✗ Error: {response.text}")
        return None

def add_images(file_paths):
    """Agregar imágenes."""
    files = [("files", (Path(fp).name, open(fp, "rb"), "image/png")) for fp in file_paths]
    response = requests.post(f"{BASE_URL}/add-images", files=files)
    for file_tuple in files:
        file_tuple[1][1].close()
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ {len(result['results'])} imágenes agregadas")
        for r in result['results']:
            print(f"  • {r['filename']}: Cluster {r['cluster_id']}")
        return result['results']
    else:
        print(f"✗ Error: {response.text}")
        return None

def main():
    print("\n🚀 TEST: 10 imágenes [5,5] → actualizar a [5,8] → agregar 3 más\n")
    
    test_dir = Path("/app/test_images")
    all_images = sorted(list(test_dir.glob("test_*.png")))
    
    if len(all_images) < 5:
        print("✗ Se necesitan al menos 5 imágenes")
        return
    
    # PASO 1: Subir 10 imágenes (repetir las 5 que tenemos)
    print_section("PASO 1: Subir 10 imágenes con capacidades [5, 5]")
    # Usar las 5 imágenes disponibles y repetirlas
    all_5_times = all_images + all_images  # Ahora son 10
    images_to_upload = all_5_times[:10]
    results1 = upload_files([str(p) for p in images_to_upload], "5,5")
    
    if not results1:
        return
    
    status1 = check_status()
    print(f"\n✅ 10 imágenes subidas:")
    print(f"   Total: {sum(status1['current_counts'])}")
    
    # PASO 2: Actualizar a [5, 8]
    print_section("PASO 2: Actualizar restricciones de [5, 5] a [5, 8]")
    update_capacities("5,8")
    status2 = check_status()
    
    # PASO 3: Agregar 3 imágenes más
    print_section("PASO 3: Agregar 3 imágenes más")
    images_to_add = all_images[:3]
    results3 = add_images([str(p) for p in images_to_add])
    
    if results3:
        status3 = check_status()
        
        # VERIFICACIÓN
        print(f"\n{'='*70}")
        print(f"  ✅ VERIFICACIÓN FINAL")
        print(f"{'='*70}")
        
        # Las 3 nuevas deberían estar en cluster 1
        cluster_1_count = status3['current_counts'][1]
        print(f"\n✓ Cluster 0: {status3['current_counts'][0]}/5 (LLENO)")
        print(f"✓ Cluster 1: {status3['current_counts'][1]}/8")
        
        if status3['current_counts'][1] == 8:
            print(f"\n✅ ¡CORRECTO! Las 3 imágenes se fueron al Cluster 1 que tiene cupo")
        else:
            print(f"\n✗ ¡PROBLEMA! Algo no está bien con la distribución")

if __name__ == "__main__":
    main()
