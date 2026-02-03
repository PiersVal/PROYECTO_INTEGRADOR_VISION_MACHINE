/**
 * clustering/momentos.js
 * Responsabilidad: Lógica específica de Momentos (24)
 * - Análisis de imágenes con Momentos
 * - Agregar imágenes al clustering
 * - Actualizar capacidades
 * - Ver estado del cluster
 */

import { analyzeImages, addImagesToCluster, updateCapacities as apiUpdateCapacities, getClusterStatus } from '../api.js';
import { setStatus, parseCapacities } from '../utils.js';
import { clearResults } from '../state.js';
import { renderResults, clearResultsDisplay } from '../ui/results.js';
import { displayMetrics } from '../ui/metrics.js';

const METHOD = 'momentos';

export function initMomentos() {
  const addImagesBtn = document.getElementById("add-images-btn");
  const updateCapacitiesBtn = document.getElementById("update-capacities-btn");
  const statusBtn = document.getElementById("status-btn");

  if (addImagesBtn) {
    addImagesBtn.addEventListener("click", () => {
      const fileInput = document.getElementById("file-input");
      addImages(fileInput.files);
    });
  }

  if (updateCapacitiesBtn) {
    updateCapacitiesBtn.addEventListener("click", () => {
      const capacitiesInput = document.getElementById("capacities-input");
      updateClusterCapacities(capacitiesInput.value);
    });
  }

  if (statusBtn) {
    statusBtn.addEventListener("click", showClusterStatus);
  }
}

export async function analyzeMomentos(files, capacities = null) {
  try {
    setStatus("Analizando con Momentos (24)...");
    clearResults();
    clearResultsDisplay();

    const data = await analyzeImages(METHOD, files, capacities);
    
    if (data.results && data.results.length > 0) {
      renderResults(data.results, METHOD);
      
      if (data.metrics) {
        displayMetrics(data.metrics);
      }
      
      setStatus(`✅ ${data.results.length} imágenes analizadas con Momentos`);
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
    setStatus(`Agregando ${files.length} imágenes al clustering Momentos...`);
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
    
    const statusDisplay = document.getElementById("cluster-status-display");
    const statusContent = document.getElementById("cluster-status-content");
    
    if (statusDisplay && statusContent) {
      statusContent.innerHTML = `
        <strong>Estado del Clustering (Momentos):</strong><br>
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
