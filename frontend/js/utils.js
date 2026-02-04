/**
 * utils.js
 * Responsabilidad: Funciones auxiliares y helpers comunes
 * - Detección de entorno (local/producción)
 * - Resolución de URLs de API
 * - Helpers de UI (status, formato)
 */

// Detectar si estamos en local o producción
export const isLocal = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";
export const API_BASE = isLocal ? "/api" : "https://remontada-uzn6.onrender.com";

// Resolver URL completa para el API
export function resolveUrl(path) {
  if (!path) return "";
  if (path.startsWith("http")) return path;
  if (path.startsWith("/")) return `${API_BASE}${path}`;
  return `${API_BASE}/${path}`;
}

// Mostrar mensaje de estado
export function setStatus(message) {
  const statusEl = document.getElementById("status");
  if (statusEl) {
    statusEl.textContent = message;
  }
}

// Parsear capacidades desde string
export function parseCapacities(capacitiesStr) {
  if (!capacitiesStr || !capacitiesStr.trim()) {
    return null;
  }
  const parts = capacitiesStr.split(",").map((x) => parseInt(x.trim(), 10));
  return parts.every((x) => !isNaN(x) && x > 0) ? parts : null;
}

// Generar ID único
export function generateId() {
  return Date.now().toString(36) + Math.random().toString(36).substr(2);
}

// Formatear número con decimales
export function formatNumber(num, decimals = 4) {
  return Number(num).toFixed(decimals);
}

// Mapeo de métodos sin etiquetas
export const unsupervisedMethods = {
  "momentos": { value: "momentos", label: "Momentos (24)" },
  "hu": { value: "hu", label: "Momentos de Hu" },
  "zernike": { value: "zernike", label: "Momentos de Zernike" },
  "sift": { value: "sift", label: "SIFT " },
  "hog": { value: "hog", label: "HOG " },
  "cnn": { value: "cnn", label: "CNN/ResNet50 " }
};

// Mapeo de métodos con etiquetas
export const supervisedMethods = {
  "momentos-metrics": { value: "external-metrics", label: " Momentos " },
  "hu-metrics": { value: "external-metrics-hu", label: " Hu " },
  "zernike-metrics": { value: "external-metrics-zernike", label: " Zernike " },
  "sift-metrics": { value: "external-metrics-sift", label: " SIFT " },
  "hog-metrics": { value: "external-metrics-hog", label: " HOG " },
  "cnn-metrics": { value: "external-metrics-cnn", label: " CNN " }
};

//cambio de frontend/js/app.js