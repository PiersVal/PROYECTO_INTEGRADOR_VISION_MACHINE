/**
 * clustering/zernike.js
 * Responsabilidad: Lógica específica de Momentos de Zernike
 * - Análisis de imágenes con Zernike
 * - Agregar imágenes al clustering
 * - Actualizar capacidades
 * - Ver estado del cluster
 */

import { analyzeImages, addImagesToCluster, updateCapacities as apiUpdateCapacities, getClusterStatus } from '../api.js';
import { setStatus, parseCapacities } from '../utils.js';
import { clearResults } from '../state.js';
import { renderResults, clearResultsDisplay } from '../ui/results.js';
import { displayMetrics } from '../ui/metrics.js';

const METHOD = 'zernike';

export function initZernike() {
  const addImagesBtn = document.getElementById("add-images-btn-zernike");
  const updateCapacitiesBtn = document.getElementById("update-capacities-btn-zernike");
  const statusBtn = document.getElementById("status-btn-zernike");

  if (addImagesBtn) {
    addImagesBtn.addEventListener("click", () => {
      const fileInput = document.getElementById("file-input");
      addImages(fileInput.files);
    });
  }

  if (updateCapacitiesBtn) {
    updateCapacitiesBtn.addEventListener("click", () => {
      const capacitiesInput = document.getElementById("zernike-capacities-input");
      updateClusterCapacities(capacitiesInput?.value);
    });
  }

  if (statusBtn) {
    statusBtn.addEventListener("click", showClusterStatus);
  }
}

export async function analyzeZernike(files, capacities = null) {
  try {
    setStatus("Analizando con Momentos de Zernike...");
    clearResults();
    clearResultsDisplay();

    const data = await analyzeImages(METHOD, files, capacities);
    
    if (data.results && data.results.length > 0) {
      renderResults(data.results, METHOD);
      
      if (data.metrics) {
        displayMetrics(data.metrics);
      }
      
      setStatus(`✅ ${data.results.length} imágenes analizadas con Zernike`);
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
    setStatus(`Agregando ${files.length} imágenes al clustering Zernike...`);
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
    
    const statusDisplay = document.getElementById("cluster-status-display-zernike");
    const statusContent = document.getElementById("cluster-status-content-zernike");
    
    if (statusDisplay && statusContent) {
      statusContent.innerHTML = `
        <strong>Estado del Clustering (Zernike):</strong><br>
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
