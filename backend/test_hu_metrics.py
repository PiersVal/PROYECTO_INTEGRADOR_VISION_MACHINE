import numpy as np
from sklearn.preprocessing import normalize
from clustering_online import LinksClusterCapacityOnline

# Crear modelo con capacidades
model = LinksClusterCapacityOnline(capacities=[5, 5])

# Simular vectores de Hu moments (7 dimensiones)
# Cluster 0: vectores similares cerca de [0.1, 0.2, 0.3, 0.1, 0.05, 0.02, 0.01]
# Cluster 1: vectores similares cerca de [0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05]

print("Agregando vectores al Cluster 0...")
for i in range(5):
    base = np.array([0.1, 0.2, 0.3, 0.1, 0.05, 0.02, 0.01])
    noise = np.random.normal(0, 0.02, 7)
    vector = base + noise
    vector_norm = normalize(vector.reshape(1, -1), norm='l2')[0]
    cid, centroid = model.predict_with_centroid(vector_norm)
    print(f"  Imagen {i+1} -> Cluster {cid}")

print("\nAgregando vectores al Cluster 1...")
for i in range(5):
    base = np.array([0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05])
    noise = np.random.normal(0, 0.02, 7)
    vector = base + noise
    vector_norm = normalize(vector.reshape(1, -1), norm='l2')[0]
    cid, centroid = model.predict_with_centroid(vector_norm)
    print(f"  Imagen {i+5+1} -> Cluster {cid}")

print("\n" + "="*50)
print("Estado del clustering:")
print(f"Número de clusters: {len(model.clusters)}")
print(f"Capacidades: {model.capacities}")
print(f"Conteos: {model.cluster_counts}")
print(f"Vectores almacenados: {len(model.all_vectors)}")
print(f"Labels: {model.all_labels}")

print("\n" + "="*50)
print("Calculando métricas...")

dunn = model.calculate_dunn_index()
silhouette = model.calculate_silhouette_coefficient()

print(f"\n📊 Índice de Dunn: {dunn:.4f}")
print(f"📊 Coeficiente de Silueta: {silhouette:.4f}")

print("\n" + "="*50)
print("Verificación:")
if dunn == 0.0 or dunn == 1.0:
    print("❌ PROBLEMA: Dunn devuelve valor por defecto")
else:
    print("✅ Dunn está calculando correctamente")

if silhouette > -0.5 and silhouette < 1:
    print("✅ Silueta está en rango válido")
else:
    print("⚠️ Silueta puede tener problemas")

# Verificar distancias intra-cluster
print("\n" + "="*50)
print("Análisis de distancias:")
X = np.array(model.all_vectors)
labels = np.array(model.all_labels)

for cluster_id in [0, 1]:
    cluster_points = X[labels == cluster_id]
    if len(cluster_points) > 1:
        # Calcular distancias dentro del cluster
        distances = []
        for i in range(len(cluster_points)):
            for j in range(i+1, len(cluster_points)):
                dist = 1 - np.dot(cluster_points[i], cluster_points[j]) / (
                    np.linalg.norm(cluster_points[i]) * np.linalg.norm(cluster_points[j])
                )
                distances.append(dist)
        
        if distances:
            print(f"Cluster {cluster_id}:")
            print(f"  Distancia intra-cluster promedio: {np.mean(distances):.4f}")
            print(f"  Distancia intra-cluster máxima: {np.max(distances):.4f}")

# Calcular distancia entre centroides
if len(model.clusters) == 2:
    centroids = model.get_cluster_centroids()
    dist_between = 1 - np.dot(centroids[0], centroids[1]) / (
        np.linalg.norm(centroids[0]) * np.linalg.norm(centroids[1])
    )
    print(f"\nDistancia entre centroides: {dist_between:.4f}")
    
    if dunn > 0:
        print(f"\n✅ Dunn = {dunn:.4f} es correcto")
        print(f"   (separación entre clusters / dispersión intra-cluster)")
