/**
 * supervised/external-metrics.js
 * Responsabilidad: Clustering de Momentos CON ETIQUETAS (métricas externas)
 * - Inicializar sesión con número de grupos
 * - Cargar imágenes por grupo/carpeta
 * - Calcular ARI, AMI, NMI
 * - Mostrar métricas internas y externas
 */

import { initializeExternalMetrics, uploadGroupImages, calculateExternalMetrics, resetExternalMetrics } from '../api.js';
import { externalMetricsState } from '../state.js';
import { setStatus, resolveUrl } from '../utils.js';
import { renderClusterVisualization } from '../ui/results.js';

const METHOD = 'momentos'; // Momentos usa "momentos" para generar endpoint sin sufijo en api.js

export function initExternalMetrics() {
  const initBtn = document.getElementById("init-external-btn");
  const calculateBtn = document.getElementById("calculate-metrics-btn");
  const resetBtn = document.getElementById("reset-external-btn");
  
  if (initBtn) {
    initBtn.addEventListener("click", initializeSession);
  }
  
  if (calculateBtn) {
    calculateBtn.addEventListener("click", calculate);
  }
  
  if (resetBtn) {
    resetBtn.addEventListener("click", reset);
  }
}

async function initializeSession() {
  const numGroupsInput = document.getElementById("external-num-groups");
  const numGroups = parseInt(numGroupsInput.value);
  
  if (!numGroups || numGroups < 2 || numGroups > 10) {
    alert("Por favor indica un número válido de grupos (entre 2 y 10)");
    return;
  }
  
  try {
    setStatus("Inicializando sesión con etiquetas...");
    
    await initializeExternalMetrics(METHOD, numGroups);
    
    externalMetricsState.numGroups = numGroups;
    externalMetricsState.groupsData = {};
    
    // Crear inputs para cada grupo
    const container = document.getElementById("external-groups-inputs");
    container.innerHTML = "";
    
    for (let i = 0; i < numGroups; i++) {
      const groupDiv = document.createElement("div");
      groupDiv.className = "group-upload-section";
      groupDiv.style.cssText = `
        border: 2px dashed #2196F3;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        background: #f5f5f5;
      `;
      
      groupDiv.innerHTML = `
        <h4>📁 Grupo ${i} (Etiqueta ${i})</h4>
        <div style="margin-bottom: 10px;">
          <label>Nombre de la clase/categoría:</label>
          <input 
            type="text" 
            id="group-label-${i}" 
            placeholder="Ej: Gato, Perro, Auto..." 
            style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px;"
          />
        </div>
        <div style="margin-bottom: 10px;">
          <label>Selecciona imágenes de este grupo:</label>
          <input 
            type="file" 
            id="group-files-${i}" 
            multiple 
            accept="image/png,image/jpeg"
            style="display: block; margin-top: 5px;"
          />
        </div>
        <div id="group-status-${i}" style="font-size: 12px; color: #666; margin-top: 5px;"></div>
      `;
      
      container.appendChild(groupDiv);
      externalMetricsState.groupsData[i] = { label: "", files: 0 };
    }
    
    document.getElementById("external-groups-container").style.display = "block";
    setStatus(`✅ Sesión inicializada para ${numGroups} grupos`);
    
  } catch (error) {
    alert(`Error: ${error.message}`);
    setStatus("❌ Error en inicialización");
  }
}

async function uploadGroup(groupId) {
  const labelInput = document.getElementById(`group-label-${groupId}`);
  const filesInput = document.getElementById(`group-files-${groupId}`);
  const statusDiv = document.getElementById(`group-status-${groupId}`);
  
  if (!labelInput || !labelInput.value.trim()) {
    alert(`Por favor ingresa una etiqueta para el grupo ${groupId}`);
    return false;
  }
  
  if (!filesInput || !filesInput.files || filesInput.files.length === 0) {
    alert(`Por favor selecciona imágenes para el grupo ${groupId}`);
    return false;
  }
  
  const label = labelInput.value.trim();
  const files = Array.from(filesInput.files);
  
  try {
    setStatus(`Cargando grupo ${groupId}...`);
    statusDiv.textContent = "⏳ Cargando...";
    statusDiv.style.color = "#ff9800";
    
    const data = await uploadGroupImages(METHOD, groupId, label, files);
    
    externalMetricsState.groupsData[groupId] = {
      label: label,
      files: files.length,
      uploaded: data.num_images_uploaded
    };
    
    statusDiv.textContent = `✅ ${data.num_images_uploaded} imágenes de "${label}" cargadas`;
    statusDiv.style.color = "#4caf50";
    
    return true;
    
  } catch (error) {
    statusDiv.textContent = `❌ Error: ${error.message}`;
    statusDiv.style.color = "#f44336";
    return false;
  }
}

async function calculate() {
  // Verificar que todos los grupos tengan datos
  for (let i = 0; i < externalMetricsState.numGroups; i++) {
    const filesInput = document.getElementById(`group-files-${i}`);
    const labelInput = document.getElementById(`group-label-${i}`);
    
    if (!filesInput || !filesInput.files || filesInput.files.length === 0) {
      alert(`El grupo ${i} no tiene imágenes. Por favor carga imágenes en todos los grupos.`);
      return;
    }
    
    if (!labelInput || !labelInput.value.trim()) {
      alert(`El grupo ${i} no tiene etiqueta. Por favor completa todos los campos.`);
      return;
    }
  }
  
  try {
    // Subir todos los grupos primero
    setStatus("Cargando todas las imágenes...");
    
    for (let i = 0; i < externalMetricsState.numGroups; i++) {
      const success = await uploadGroup(i);
      if (!success) {
        alert(`Error cargando el grupo ${i}. Por favor intenta de nuevo.`);
        return;
      }
    }
    
    // Calcular métricas
    setStatus("Calculando ARI/AMI/NMI...");
    
    const capacitiesInput = document.getElementById("external-capacities");
    const capacities = capacitiesInput?.value?.trim() || null;
    
    const data = await calculateExternalMetrics(METHOD, capacities);
    
    // Mostrar métricas externas
    document.getElementById("result-ari").textContent = data.external_metrics.ARI.toFixed(4);
    document.getElementById("result-ami").textContent = data.external_metrics.AMI.toFixed(4);
    document.getElementById("result-nmi").textContent = data.external_metrics.NMI.toFixed(4);
    
    // Mostrar métricas internas
    document.getElementById("result-dunn").textContent = data.internal_metrics.dunn_index.toFixed(4);
    document.getElementById("result-silhouette").textContent = data.internal_metrics.silhouette_coefficient.toFixed(4);
    
    // Mostrar resumen
    const summary = data.summary;
    const summaryDiv = document.getElementById("metrics-summary");
    if (summaryDiv) {
      summaryDiv.innerHTML = `
        <strong>📊 Resumen:</strong><br>
        • Total de imágenes: ${summary.num_images}<br>
        • Número de clusters: ${summary.num_clusters}<br>
        • Grupos etiquetados: ${summary.true_groups}<br>
        • Clusters predichos: ${summary.predicted_clusters}
      `;
    }
    
    // Renderizar visualización de clusters
    if (data.clusters) {
      const resultsDiv = document.getElementById("clusters-visualization");
      if (resultsDiv) {
        renderClusterVisualization(data.clusters, "clusters-visualization");
      }
    }
    
    // Mostrar sección de resultados
    document.getElementById("external-metrics-results").style.display = "block";
    document.getElementById("results-section").style.display = "block";
    
    setStatus("✅ Métricas calculadas exitosamente");
    
  } catch (error) {
    alert(`Error: ${error.message}`);
    setStatus("❌ Error al calcular métricas");
  }
}

async function reset() {
  if (!confirm("¿Borrar todos los resultados y reiniciar la sesión?")) {
    return;
  }
  
  try {
    setStatus("Limpiando sesión...");
    
    await resetExternalMetrics(METHOD);
    
    // Limpiar estado
    externalMetricsState.numGroups = 0;
    externalMetricsState.groupsData = {};
    
    // Limpiar UI
    document.getElementById("external-groups-container").style.display = "none";
    document.getElementById("external-groups-inputs").innerHTML = "";
    document.getElementById("external-metrics-results").style.display = "none";
    document.getElementById("external-num-groups").value = "";
    
    // Limpiar resultados
    document.getElementById("result-ari").textContent = "-";
    document.getElementById("result-ami").textContent = "-";
    document.getElementById("result-nmi").textContent = "-";
    document.getElementById("result-dunn").textContent = "-";
    document.getElementById("result-silhouette").textContent = "-";
    
    const summaryDiv = document.getElementById("metrics-summary");
    if (summaryDiv) {
      summaryDiv.innerHTML = "";
    }
    
    const visualizationDiv = document.getElementById("clusters-visualization");
    if (visualizationDiv) {
      visualizationDiv.innerHTML = "";
    }
    
    setStatus("✅ Sesión reiniciada");
    
  } catch (error) {
    alert(`Error: ${error.message}`);
    setStatus("❌ Error al resetear");
  }
}
