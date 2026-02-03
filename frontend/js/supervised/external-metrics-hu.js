/**
 * supervised/external-metrics-hu.js
 * Responsabilidad: Clustering de Hu CON ETIQUETAS (métricas externas)
 */

import { initializeExternalMetrics, uploadGroupImages, calculateExternalMetrics, resetExternalMetrics } from '../api.js';
import { externalMetricsHuState } from '../state.js';
import { setStatus } from '../utils.js';
import { renderClusterVisualization } from '../ui/results.js';

const METHOD = 'hu';
const PREFIX = 'external-hu';

export function initExternalMetricsHu() {
  const initBtn = document.getElementById(`init-${PREFIX}-btn`);
  const calculateBtn = document.getElementById(`calculate-metrics-hu-btn`);
  const resetBtn = document.getElementById(`reset-${PREFIX}-btn`);
  
  if (initBtn) initBtn.addEventListener("click", initializeSession);
  if (calculateBtn) calculateBtn.addEventListener("click", calculate);
  if (resetBtn) resetBtn.addEventListener("click", reset);
}

async function initializeSession() {
  const numGroupsInput = document.getElementById(`${PREFIX}-num-groups`);
  const numGroups = parseInt(numGroupsInput.value);
  
  if (!numGroups || numGroups < 2 || numGroups > 10) {
    alert("Por favor indica un número válido de grupos (entre 2 y 10)");
    return;
  }
  
  try {
    setStatus("Inicializando sesión Hu con etiquetas...");
    await initializeExternalMetrics(METHOD, numGroups);
    
    externalMetricsHuState.numGroups = numGroups;
    externalMetricsHuState.groupsData = {};
    
    const container = document.getElementById(`${PREFIX}-groups-inputs`);
    container.innerHTML = "";
    
    for (let i = 0; i < numGroups; i++) {
      const groupDiv = document.createElement("div");
      groupDiv.className = "group-upload-section";
      groupDiv.style.cssText = "border: 2px dashed #2196F3; border-radius: 8px; padding: 15px; margin-bottom: 15px; background: #f5f5f5;";
      
      groupDiv.innerHTML = `
        <h4>📁 Grupo ${i}</h4>
        <label>Nombre de la clase/categoría:</label>
        <input type="text" id="group-label-hu-${i}" placeholder="Ej: Gato, Perro..." style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; margin-bottom: 10px;" />
        <label>Selecciona imágenes:</label>
        <input type="file" id="group-files-hu-${i}" multiple accept="image/png,image/jpeg" style="display: block; margin-top: 5px;" />
        <div id="group-status-hu-${i}" style="font-size: 12px; color: #666; margin-top: 5px;"></div>
      `;
      
      container.appendChild(groupDiv);
      externalMetricsHuState.groupsData[i] = { label: "", files: 0 };
    }
    
    document.getElementById(`${PREFIX}-groups-container`).style.display = "block";
    setStatus(`✅ Sesión Hu inicializada para ${numGroups} grupos`);
  } catch (error) {
    alert(`Error: ${error.message}`);
    setStatus("❌ Error en inicialización");
  }
}

async function uploadGroup(groupId) {
  const labelInput = document.getElementById(`group-label-hu-${groupId}`);
  const filesInput = document.getElementById(`group-files-hu-${groupId}`);
  const statusDiv = document.getElementById(`group-status-hu-${groupId}`);
  
  if (!labelInput?.value.trim()) {
    alert(`Ingresa una etiqueta para el grupo ${groupId}`);
    return false;
  }
  
  if (!filesInput?.files || filesInput.files.length === 0) {
    alert(`Selecciona imágenes para el grupo ${groupId}`);
    return false;
  }
  
  try {
    statusDiv.textContent = "⏳ Cargando...";
    statusDiv.style.color = "#ff9800";
    
    const data = await uploadGroupImages(METHOD, groupId, labelInput.value.trim(), Array.from(filesInput.files));
    
    externalMetricsHuState.groupsData[groupId] = {
      label: labelInput.value.trim(),
      files: filesInput.files.length,
      uploaded: data.num_images_uploaded
    };
    
    statusDiv.textContent = `✅ ${data.num_images_uploaded} imágenes cargadas`;
    statusDiv.style.color = "#4caf50";
    return true;
  } catch (error) {
    statusDiv.textContent = `❌ ${error.message}`;
    statusDiv.style.color = "#f44336";
    return false;
  }
}

async function calculate() {
  for (let i = 0; i < externalMetricsHuState.numGroups; i++) {
    const filesInput = document.getElementById(`group-files-hu-${i}`);
    const labelInput = document.getElementById(`group-label-hu-${i}`);
    
    if (!filesInput?.files || filesInput.files.length === 0) {
      alert(`El grupo ${i} no tiene imágenes`);
      return;
    }
    if (!labelInput?.value.trim()) {
      alert(`El grupo ${i} no tiene etiqueta`);
      return;
    }
  }
  
  try {
    setStatus("Cargando imágenes...");
    for (let i = 0; i < externalMetricsHuState.numGroups; i++) {
      if (!await uploadGroup(i)) return;
    }
    
    setStatus("Calculando métricas Hu...");
    const capacitiesInput = document.getElementById(`${PREFIX}-capacities`);
    const data = await calculateExternalMetrics(METHOD, capacitiesInput?.value?.trim() || null);
    
    document.getElementById("result-hu-ari").textContent = data.external_metrics.ARI.toFixed(4);
    document.getElementById("result-hu-ami").textContent = data.external_metrics.AMI.toFixed(4);
    document.getElementById("result-hu-nmi").textContent = data.external_metrics.NMI.toFixed(4);
    document.getElementById("result-hu-dunn").textContent = data.internal_metrics.dunn_index.toFixed(4);
    document.getElementById("result-hu-silhouette").textContent = data.internal_metrics.silhouette_coefficient.toFixed(4);
    
    const summaryDiv = document.getElementById("metrics-hu-summary");
    if (summaryDiv) {
      summaryDiv.innerHTML = `<strong>📊 Resumen:</strong><br>• Total: ${data.summary.num_images}<br>• Clusters: ${data.summary.num_clusters}<br>• Grupos: ${data.summary.true_groups}`;
    }
    
    if (data.clusters) {
      renderClusterVisualization(data.clusters, "clusters-hu-visualization");
    }
    
    document.getElementById("external-metrics-hu-results").style.display = "block";
    document.getElementById("results-section").style.display = "block";
    setStatus("✅ Métricas Hu calculadas");
  } catch (error) {
    alert(`Error: ${error.message}`);
    setStatus("❌ Error al calcular");
  }
}

async function reset() {
  if (!confirm("¿Borrar resultados y reiniciar?")) return;
  
  try {
    await resetExternalMetrics(METHOD);
    externalMetricsHuState.numGroups = 0;
    externalMetricsHuState.groupsData = {};
    
    document.getElementById(`${PREFIX}-groups-container`).style.display = "none";
    document.getElementById(`${PREFIX}-groups-inputs`).innerHTML = "";
    document.getElementById("external-metrics-hu-results").style.display = "none";
    document.getElementById(`${PREFIX}-num-groups`).value = "";
    
    ["ari", "ami", "nmi", "dunn", "silhouette"].forEach(m => {
      document.getElementById(`result-hu-${m}`).textContent = "-";
    });
    
    setStatus("✅ Sesión Hu reiniciada");
  } catch (error) {
    alert(`Error: ${error.message}`);
  }
}
