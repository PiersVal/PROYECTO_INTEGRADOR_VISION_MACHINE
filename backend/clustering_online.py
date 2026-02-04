import numpy as np


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)


class Subcluster:
    def __init__(self, initial_vector: np.ndarray):
        self.centroid = initial_vector.copy()
        self.n_vectors = 1

    def add(self, vector: np.ndarray):
        self.n_vectors += 1
        self.centroid = (self.n_vectors - 1) / self.n_vectors * self.centroid + vector / self.n_vectors


class LinksClusterCapacityOnline:
    def __init__(
        self,
        capacities: list[int],
        cluster_similarity_threshold: float = 0.75,
        subcluster_similarity_threshold: float = 0.85,
        pair_similarity_maximum: float = 0.95,
    ):
        if not capacities or any(int(c) <= 0 for c in capacities):
            raise ValueError("capacities debe ser una lista no vacía de enteros > 0")

        self.capacities = [int(c) for c in capacities]
        self.k = len(self.capacities)

        self.cluster_similarity_threshold = cluster_similarity_threshold
        self.subcluster_similarity_threshold = subcluster_similarity_threshold
        self.pair_similarity_maximum = pair_similarity_maximum

        self.clusters: list[list[Subcluster]] = []
        self.cluster_counts: list[int] = []

        self.last_centroid: np.ndarray | None = None
        self.last_cluster_id: int | None = None
        
        # Almacenar todos los vectores y sus cluster_ids para cálculo de métricas
        self.all_vectors: list[np.ndarray] = []
        self.all_labels: list[int] = []

    def sim_threshold(self, k: int, kp: int) -> float:
        s = (1.0 + 1.0 / k * (1.0 / self.cluster_similarity_threshold**2 - 1.0))
        s *= (1.0 + 1.0 / kp * (1.0 / self.cluster_similarity_threshold**2 - 1.0))
        s = 1.0 / np.sqrt(s)
        s = (
            self.cluster_similarity_threshold**2
            + (self.pair_similarity_maximum - self.cluster_similarity_threshold**2)
            / (1.0 - self.cluster_similarity_threshold**2)
            * (s - self.cluster_similarity_threshold**2)
        )
        return float(s)

    def _has_capacity(self, cid: int) -> bool:
        return self.cluster_counts[cid] < self.capacities[cid]

    def _append_new_cluster(self, x: np.ndarray) -> int:
        if len(self.clusters) >= self.k:
            raise RuntimeError("No se pueden crear más clusters (k alcanzado)")
        self.clusters.append([Subcluster(x)])
        self.cluster_counts.append(1)
        self.last_centroid = x.copy()
        self.last_cluster_id = len(self.clusters) - 1
        return self.last_cluster_id

    def predict_with_centroid(self, x: np.ndarray, allow_new_clusters: bool = True) -> tuple[int, np.ndarray]:
        if len(self.clusters) == 0:
            if allow_new_clusters:
                cid = self._append_new_cluster(x)
                # Almacenar vector y label
                self.all_vectors.append(x.copy())
                self.all_labels.append(cid)
                return cid, self.last_centroid.copy()
            raise RuntimeError("No hay clusters existentes y no se permiten nuevos")

        if all(self.cluster_counts[cid] >= self.capacities[cid] for cid in range(len(self.cluster_counts))):
            if allow_new_clusters and len(self.clusters) < self.k:
                cid = self._append_new_cluster(x)
                # Almacenar vector y label
                self.all_vectors.append(x.copy())
                self.all_labels.append(cid)
                return cid, self.last_centroid.copy()
            raise RuntimeError("Capacidad excedida: todos los clusters están llenos")

        # FASE 1: Buscar mejor cluster CON CUPO disponible
        best_cid, best_sc, best_sim = None, None, -np.inf
        for cid, cl in enumerate(self.clusters):
            if not self._has_capacity(cid):
                continue
            for sc in cl:
                s = cos_sim(x, sc.centroid)
                if s > best_sim:
                    best_sim = s
                    best_cid = cid
                    best_sc = sc

        # Si no hay clusters con cupo, error o crear nuevo
        if best_cid is None or best_sc is None:
            if allow_new_clusters and len(self.clusters) < self.k:
                cid = self._append_new_cluster(x)
                # Almacenar vector y label
                self.all_vectors.append(x.copy())
                self.all_labels.append(cid)
                return cid, self.last_centroid.copy()
            raise RuntimeError("No hay clusters con capacidad disponible")

        # IMPORTANTE: En este punto, best_cid tiene cupo garantizado
        # Verificar también que tiene cupo antes de cualquier asignación
        if not self._has_capacity(best_cid):
            raise RuntimeError("Capacidad excedida: no hay cluster con cupo para asignar el punto")

        # FASE 2: Intentar agregar al subcluster más similar
        if best_sim >= self.subcluster_similarity_threshold:
            best_sc.add(x)
            self.cluster_counts[best_cid] += 1
            self.last_centroid = best_sc.centroid.copy()
            self.last_cluster_id = best_cid
            # Almacenar vector y label
            self.all_vectors.append(x.copy())
            self.all_labels.append(best_cid)
            return best_cid, self.last_centroid.copy()

        # FASE 3: Crear nuevo subcluster si es similar al existente
        new_sc = Subcluster(x)
        s_link = cos_sim(new_sc.centroid, best_sc.centroid)
        if s_link >= self.sim_threshold(best_sc.n_vectors, 1):
            # Verificar cupo antes de agregar subcluster
            if not self._has_capacity(best_cid):
                raise RuntimeError("Capacidad excedida: no hay cupo para nuevo subcluster")
            self.clusters[best_cid].append(new_sc)
            self.cluster_counts[best_cid] += 1
            self.last_centroid = new_sc.centroid.copy()
            self.last_cluster_id = best_cid
            # Almacenar vector y label
            self.all_vectors.append(x.copy())
            self.all_labels.append(best_cid)
            return best_cid, self.last_centroid.copy()

        # FASE 4: No cumple similitud, intentar crear nuevo cluster
        if allow_new_clusters and len(self.clusters) < self.k:
            cid = self._append_new_cluster(x)
            # Almacenar vector y label
            self.all_vectors.append(x.copy())
            self.all_labels.append(cid)
            return cid, self.last_centroid.copy()

        # FASE 5: Última opción - asignar al mejor con cupo (fallback)
        if not self._has_capacity(best_cid):
            raise RuntimeError("Capacidad excedida: no hay cluster con cupo")
        
        best_sc.add(x)
        self.cluster_counts[best_cid] += 1
        self.last_centroid = best_sc.centroid.copy()
        self.last_cluster_id = best_cid
        # Almacenar vector y label
        self.all_vectors.append(x.copy())
        self.all_labels.append(best_cid)
        return best_cid, self.last_centroid.copy()

    def to_dict(self) -> dict:
        """Serializa el estado del modelo para persistencia"""
        return {
            "capacities": self.capacities,
            "cluster_similarity_threshold": self.cluster_similarity_threshold,
            "subcluster_similarity_threshold": self.subcluster_similarity_threshold,
            "pair_similarity_maximum": self.pair_similarity_maximum,
            "cluster_counts": self.cluster_counts,
            "clusters": [
                [{"centroid": sc.centroid.tolist(), "n_vectors": sc.n_vectors} 
                 for sc in cluster] 
                for cluster in self.clusters
            ],
            "last_centroid": self.last_centroid.tolist() if self.last_centroid is not None else None,
            "last_cluster_id": self.last_cluster_id,
            "all_vectors": [v.tolist() for v in self.all_vectors],
            "all_labels": self.all_labels,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LinksClusterCapacityOnline":
        """Reconstruye el modelo desde un diccionario"""
        model = cls(
            capacities=data["capacities"],
            cluster_similarity_threshold=data["cluster_similarity_threshold"],
            subcluster_similarity_threshold=data["subcluster_similarity_threshold"],
            pair_similarity_maximum=data["pair_similarity_maximum"]
        )
        model.cluster_counts = data["cluster_counts"]
        model.clusters = []
        for cluster_data in data["clusters"]:
            subclusters = []
            for sc_data in cluster_data:
                sc = Subcluster(np.array(sc_data["centroid"]))
                sc.n_vectors = sc_data["n_vectors"]
                subclusters.append(sc)
            model.clusters.append(subclusters)
        model.last_centroid = np.array(data["last_centroid"]) if data["last_centroid"] else None
        model.last_cluster_id = data["last_cluster_id"]
        # Restaurar vectores y labels
        model.all_vectors = [np.array(v) for v in data.get("all_vectors", [])]
        model.all_labels = data.get("all_labels", [])
        return model

    def get_cluster_centroids(self) -> np.ndarray:
        """Retorna los centroides de todos los clusters"""
        centroids = []
        for cluster in self.clusters:
            if cluster:
                # Promedio de los centroides de los subclusters ponderado por n_vectors
                centroid = np.zeros_like(cluster[0].centroid)
                total_vectors = 0
                for sc in cluster:
                    centroid += sc.centroid * sc.n_vectors
                    total_vectors += sc.n_vectors
                if total_vectors > 0:
                    centroid /= total_vectors
                centroids.append(centroid)
        return np.array(centroids) if centroids else np.array([])

    def calculate_dunn_index(self) -> float:
        """
        Calcula el Índice de Dunn usando los vectores reales
        DI = min_distance_between_clusters / max_distance_within_cluster
        Valores más altos indican mejor separación entre clusters
        """
        if len(self.all_vectors) == 0 or len(self.clusters) <= 1:
            return 0.0
        
        # Convertir a numpy arrays
        X = np.array(self.all_vectors)
        labels = np.array(self.all_labels)
        unique_labels = np.unique(labels)
        
        if len(unique_labels) <= 1:
            return 0.0
        
        # Distancia mínima entre clusters
        min_inter_distance = np.inf
        for i in range(len(unique_labels)):
            for j in range(i + 1, len(unique_labels)):
                cluster_i = X[labels == unique_labels[i]]
                cluster_j = X[labels == unique_labels[j]]
                
                # Distancia mínima entre puntos de diferentes clusters
                for point_i in cluster_i:
                    for point_j in cluster_j:
                        dist = 1 - cos_sim(point_i, point_j)
                        min_inter_distance = min(min_inter_distance, dist)
        
        # Distancia máxima dentro de clusters
        max_intra_distance = 0
        for label in unique_labels:
            cluster_points = X[labels == label]
            if len(cluster_points) <= 1:
                continue
            
            # Calcular distancia máxima dentro del cluster
            for i in range(len(cluster_points)):
                for j in range(i + 1, len(cluster_points)):
                    dist = 1 - cos_sim(cluster_points[i], cluster_points[j])
                    max_intra_distance = max(max_intra_distance, dist)
        
        if max_intra_distance == 0 or min_inter_distance == np.inf:
            return 0.0
        
        return float(min_inter_distance / max_intra_distance)

    def calculate_silhouette_coefficient(self) -> float:
        """
        Calcula el Coeficiente de Silueta usando los vectores reales
        Rango: [-1, 1], valores más altos indican mejor clustering
        """
        if len(self.all_vectors) == 0 or len(self.clusters) <= 1:
            return 0.0
        
        X = np.array(self.all_vectors)
        labels = np.array(self.all_labels)
        unique_labels = np.unique(labels)
        
        if len(unique_labels) <= 1:
            return 0.0
        
        silhouette_scores = []
        
        for i, point in enumerate(X):
            own_label = labels[i]
            own_cluster = X[labels == own_label]
            
            # a(i) = distancia promedio al resto de puntos en su cluster
            if len(own_cluster) <= 1:
                a_i = 0
            else:
                distances = [1 - cos_sim(point, other) for other in own_cluster if not np.array_equal(point, other)]
                a_i = np.mean(distances) if distances else 0
            
            # b(i) = distancia promedio mínima a puntos de otros clusters
            b_values = []
            for label in unique_labels:
                if label != own_label:
                    other_cluster = X[labels == label]
                    distances = [1 - cos_sim(point, other) for other in other_cluster]
                    b_values.append(np.mean(distances))
            
            if not b_values:
                b_i = a_i
            else:
                b_i = min(b_values)
            
            # Silueta para el punto i
            if max(a_i, b_i) > 0:
                s_i = (b_i - a_i) / max(a_i, b_i)
            else:
                s_i = 0.0
            
            silhouette_scores.append(s_i)
        
        return float(np.mean(silhouette_scores)) if silhouette_scores else 0.0
