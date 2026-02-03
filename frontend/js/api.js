/**
 * api.js
 * Responsabilidad: Todas las llamadas HTTP al backend FastAPI
 * - Upload de imágenes
 * - Análisis con diferentes métodos
 * - Clustering y métricas
 * - Estados de clusters
 */

import { resolveUrl } from './utils.js';

// ==================== GALERÍA ====================

export async function uploadImages(files) {
  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }

  const response = await fetch(resolveUrl("/upload"), {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Error al subir imágenes: ${response.statusText}`);
  }

  return await response.json();
}

export async function getGalleryItems() {
  const response = await fetch(resolveUrl("/images"));
  if (!response.ok) {
    throw new Error(`Error al obtener items: ${response.statusText}`);
  }
  return await response.json();
}

export async function clearAllImages() {
  const response = await fetch(resolveUrl("/images"), {
    method: "DELETE",
  });
  if (!response.ok) {
    throw new Error(`Error al limpiar: ${response.statusText}`);
  }
  return await response.json();
}

// ==================== CLUSTERING SIN ETIQUETAS ====================

export async function analyzeImages(method, files, capacities = null, reset = true) {
  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }
  if (capacities) {
    formData.append("capacities", capacities);
  }
  if (reset) {
    formData.append("reset", "true");
  }

  // Momentos usa /analyze, otros métodos usan /analyze-{method}
  const endpoint = method === 'momentos' ? '/analyze' : `/analyze-${method}`;
  const response = await fetch(resolveUrl(endpoint), {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `Error en análisis ${method}`);
  }

  return await response.json();
}

export async function addImagesToCluster(method, files) {
  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }

  // Momentos usa /add-images, otros métodos usan /add-images-{method}
  const endpoint = method === 'momentos' ? '/add-images' : `/add-images-${method}`;
  const response = await fetch(resolveUrl(endpoint), {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `Error al agregar imágenes`);
  }

  return await response.json();
}

export async function updateCapacities(method, capacities) {
  const formData = new FormData();
  formData.append("capacities", capacities);

  // Momentos usa /update-capacities, otros métodos usan /update-capacities-{method}
  const endpoint = method === 'momentos' ? '/update-capacities' : `/update-capacities-${method}`;
  const response = await fetch(resolveUrl(endpoint), {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `Error al actualizar capacidades`);
  }

  return await response.json();
}

export async function getClusterStatus(method) {
  // Momentos usa /cluster-status, otros métodos usan /cluster-status-{method}
  const endpoint = method === 'momentos' ? '/cluster-status' : `/cluster-status-${method}`;
  const response = await fetch(resolveUrl(endpoint));

  if (!response.ok) {
    throw new Error(`Error al obtener estado del cluster`);
  }

  return await response.json();
}

// ==================== MÉTRICAS EXTERNAS ====================

export async function initializeExternalMetrics(method, numClusters) {
  // Momentos usa /external-metrics, otros usan /external-metrics-{method}
  const endpoint = method === 'momentos' ? '/external-metrics/initialize' : `/external-metrics-${method}/initialize`;
  
  const response = await fetch(resolveUrl(endpoint), {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: `num_clusters=${numClusters}`
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Error al inicializar");
  }

  return await response.json();
}

export async function uploadGroupImages(method, groupId, label, files) {
  const formData = new FormData();
  formData.append("group_id", groupId);
  formData.append("label", label);
  
  for (const file of files) {
    formData.append("files", file);
  }

  // Momentos usa /external-metrics, otros usan /external-metrics-{method}
  const endpoint = method === 'momentos' ? '/external-metrics/upload-group' : `/external-metrics-${method}/upload-group`;
  
  const response = await fetch(resolveUrl(endpoint), {
    method: "POST",
    body: formData
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Error al subir grupo");
  }

  return await response.json();
}

export async function calculateExternalMetrics(method, capacities = null) {
  const formData = new FormData();
  if (capacities) {
    formData.append("capacities", capacities);
  }

  // Momentos usa /external-metrics, otros usan /external-metrics-{method}
  const endpoint = method === 'momentos' ? '/external-metrics/calculate' : `/external-metrics-${method}/calculate`;
  
  const response = await fetch(resolveUrl(endpoint), {
    method: "POST",
    body: formData
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Error al calcular métricas");
  }

  return await response.json();
}

export async function resetExternalMetrics(method) {
  // Momentos usa /external-metrics, otros usan /external-metrics-{method}
  const endpoint = method === 'momentos' ? '/external-metrics/reset' : `/external-metrics-${method}/reset`;
  
  const response = await fetch(resolveUrl(endpoint), {
    method: "DELETE"
  });

  if (!response.ok) {
    throw new Error("Error al resetear");
  }

  return await response.json();
}
