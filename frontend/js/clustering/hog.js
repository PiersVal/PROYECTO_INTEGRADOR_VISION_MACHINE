/**
 * clustering/hog.js
 * Responsabilidad: Lógica específica de HOG
 * - Análisis de imágenes con HOG
 * - Agregar imágenes al clustering
 * - Actualizar capacidades
 * - Ver estado del cluster
 */

import { analyzeImages, addImagesToCluster, updateCapacities as apiUpdateCapacities, getClusterStatus } from '../api.js';
import { setStatus, parseCapacities } from '../utils.js';
import { clearResults } from '../state.js';
import { renderResults, clearResultsDisplay } from '../ui/results.js';
import { displayMetrics } from '../ui/metrics.js';

const METHOD = 'hog';

export function initHog() {
  const addImagesBtn = document.getElementById("add-images-btn-hog");
  const updateCapacitiesBtn = document.getElementById("update-capacities-btn-hog");
  const statusBtn = document.getElementById("status-btn-hog");

  if (addImagesBtn) {
    addImagesBtn.addEventListener("click", () => {
      const fileInput = document.getElementById("file-input");
      addImages(fileInput.files);
    });
  }

  if (updateCapacitiesBtn) {
    updateCapacitiesBtn.addEventListener("click", () => {
      const capacitiesInput = document.getElementById("hog-capacities-input");
      updateClusterCapacities(capacitiesInput?.value);
    });
  }

  if (statusBtn) {
    statusBtn.addEventListener("click", showClusterStatus);
  }
}

export async function analyzeHog(files, capacities = null) {
  try {
    setStatus("Analizando con HOG...");
    clearResults();
    clearResultsDisplay();

    const data = await analyzeImages(METHOD, files, capacities);
    
    if (data.results && data.results.length > 0) {
      renderResults(data.results, METHOD);
      
      if (data.metrics) {
        displayMetrics(data.metrics);
      }
      
      setStatus(`✅ ${data.results.length} imágenes analizadas con HOG`);
    }
  } catch (error) {
    alert(`Error: ${error.message}`);
    setStatus("❌ Error en análisis");
  }
}

async function addImages(files) {
  if (!files || files.length === 0) {
    alert("❌ Selecciona imágenes primero");
    return;
  }

  try {
    setStatus(`Agregando ${files.length} imágenes al clustering HOG...`);
    const data = await addImagesToCluster(METHOD, files);
    
    if (data.results && data.results.length > 0) {
      // NO limpiar, solo agregar a los resultados existentes
      renderResults(data.results, METHOD);
      
      if (data.metrics) {
        displayMetrics(data.metrics);
      }
      
      setStatus(`✅ ${data.results.length} imágenes agregadas`);
    }
  } catch (error) {
    alert(`Error: ${error.message}`);
    setStatus("❌ Error al agregar imágenes");
  }
}

async function updateClusterCapacities(capacitiesStr) {
  const caps = parseCapacities(capacitiesStr);
  if (!caps) {
    alert("Formato inválido. Ejemplo: 5,10,15");
    return;
  }

  try {
    setStatus("Actualizando capacidades...");
    const data = await apiUpdateCapacities(METHOD, capacitiesStr);
    
    if (data.results && data.results.length > 0) {
      clearResults();
      clearResultsDisplay();
      renderResults(data.results, METHOD);
      
      if (data.metrics) {
        displayMetrics(data.metrics);
      }
    }
    
    setStatus("✅ Capacidades actualizadas");
  } catch (error) {
    alert(`Error: ${error.message}`);
    setStatus("❌ Error al actualizar");
  }
}

async function showClusterStatus() {
  try {
    const data = await getClusterStatus(METHOD);
    
    const statusDisplay = document.getElementById("cluster-status-display-hog");
    const statusContent = document.getElementById("cluster-status-content-hog");
    
    if (statusDisplay && statusContent) {
      statusContent.innerHTML = `
        <strong>Estado del Clustering (HOG):</strong><br>
        • Total de imágenes: ${data.num_images}<br>
        • Clusters activos: ${data.num_clusters}<br>
        • Capacidades: [${data.capacities.join(", ")}]
      `;
      statusDisplay.style.display = "block";
    }
  } catch (error) {
    alert(`Error: ${error.message}`);
  }
}
