/**
 * supervised/external-metrics-sift.js
 */

import { initializeExternalMetrics, uploadGroupImages, calculateExternalMetrics, resetExternalMetrics } from '../api.js';
import { externalMetricsSiftState } from '../state.js';
import { setStatus } from '../utils.js';
import { renderClusterVisualization } from '../ui/results.js';

const METHOD = 'sift';
const PREFIX = 'external-sift';

export function initExternalMetricsSift() {
  const initBtn = document.getElementById(`init-${PREFIX}-btn`);
  const calculateBtn = document.getElementById(`calculate-metrics-sift-btn`);
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
    setStatus("Inicializando sesión SIFT con etiquetas...");
    await initializeExternalMetrics(METHOD, numGroups);
    
    externalMetricsSiftState.numGroups = numGroups;
    externalMetricsSiftState.groupsData = {};
    
    const container = document.getElementById(`${PREFIX}-groups-inputs`);
    container.innerHTML = "";
    
    for (let i = 0; i < numGroups; i++) {
      const groupDiv = document.createElement("div");
      groupDiv.className = "group-upload-section";
      groupDiv.style.cssText = "border: 2px dashed #2196F3; border-radius: 8px; padding: 15px; margin-bottom: 15px; background: #f5f5f5;";
      
      groupDiv.innerHTML = `
        <h4>📁 Grupo ${i}</h4>
        <label>Nombre de la clase/categoría:</label>
        <input type="text" id="group-label-sift-${i}" placeholder="Ej: Gato, Perro..." style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; margin-bottom: 10px;" />
        <label>Selecciona imágenes:</label>
        <input type="file" id="group-files-sift-${i}" multiple accept="image/png,image/jpeg" style="display: block; margin-top: 5px;" />
        <div id="group-status-sift-${i}" style="font-size: 12px; color: #666; margin-top: 5px;"></div>
      `;
      
      container.appendChild(groupDiv);
      externalMetricsSiftState.groupsData[i] = { label: "", files: 0 };
    }
    
    document.getElementById(`${PREFIX}-groups-container`).style.display = "block";
    setStatus(`✅ Sesión SIFT inicializada para ${numGroups} grupos`);
  } catch (error) {
    alert(`Error: ${error.message}`);
    setStatus("❌ Error en inicialización");
  }
}

async function uploadGroup(groupId) {
  const labelInput = document.getElementById(`group-label-sift-${groupId}`);
  const filesInput = document.getElementById(`group-files-sift-${groupId}`);
  const statusDiv = document.getElementById(`group-status-sift-${groupId}`);
  
  if (!labelInput?.value.trim() || !filesInput?.files || filesInput.files.length === 0) {
    alert(`Completa el grupo ${groupId}`);
    return false;
  }
  
  try {
    statusDiv.textContent = "⏳ Cargando...";
    statusDiv.style.color = "#ff9800";
    
    const data = await uploadGroupImages(METHOD, groupId, labelInput.value.trim(), Array.from(filesInput.files));
    
    externalMetricsSiftState.groupsData[groupId] = {
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
  for (let i = 0; i < externalMetricsSiftState.numGroups; i++) {
    const filesInput = document.getElementById(`group-files-sift-${i}`);
    const labelInput = document.getElementById(`group-label-sift-${i}`);
    
    if (!filesInput?.files || filesInput.files.length === 0 || !labelInput?.value.trim()) {
      alert(`Completa todos los campos del grupo ${i}`);
      return;
    }
  }
  
  try {
    setStatus("Cargando imágenes...");
    for (let i = 0; i < externalMetricsSiftState.numGroups; i++) {
      if (!await uploadGroup(i)) return;
    }
    
    setStatus("Calculando métricas SIFT...");
    const capacitiesInput = document.getElementById(`${PREFIX}-capacities`);
    const data = await calculateExternalMetrics(METHOD, capacitiesInput?.value?.trim() || null);
    
    document.getElementById("result-sift-ari").textContent = data.external_metrics.ARI.toFixed(4);
    document.getElementById("result-sift-ami").textContent = data.external_metrics.AMI.toFixed(4);
    document.getElementById("result-sift-nmi").textContent = data.external_metrics.NMI.toFixed(4);
    document.getElementById("result-sift-dunn").textContent = data.internal_metrics.dunn_index.toFixed(4);
    document.getElementById("result-sift-silhouette").textContent = data.internal_metrics.silhouette_coefficient.toFixed(4);
    
    const summaryDiv = document.getElementById("metrics-sift-summary");
    if (summaryDiv) {
      summaryDiv.innerHTML = `<strong>📊 Resumen:</strong><br>• Total: ${data.summary.num_images}<br>• Clusters: ${data.summary.num_clusters}<br>• Grupos: ${data.summary.true_groups}`;
    }
    
    if (data.clusters) {
      renderClusterVisualization(data.clusters, "clusters-sift-visualization");
    }
    
    document.getElementById("external-metrics-sift-results").style.display = "block";
    document.getElementById("results-section").style.display = "block";
    setStatus("✅ Métricas SIFT calculadas");
  } catch (error) {
    alert(`Error: ${error.message}`);
    setStatus("❌ Error al calcular");
  }
}

async function reset() {
  if (!confirm("¿Borrar resultados y reiniciar?")) return;
  
  try {
    await resetExternalMetrics(METHOD);
    externalMetricsSiftState.numGroups = 0;
    externalMetricsSiftState.groupsData = {};
    
    document.getElementById(`${PREFIX}-groups-container`).style.display = "none";
    document.getElementById(`${PREFIX}-groups-inputs`).innerHTML = "";
    document.getElementById("external-metrics-sift-results").style.display = "none";
    document.getElementById(`${PREFIX}-num-groups`).value = "";
    
    ["ari", "ami", "nmi", "dunn", "silhouette"].forEach(m => {
      document.getElementById(`result-sift-${m}`).textContent = "-";
    });
    
    setStatus("✅ Sesión SIFT reiniciada");
  } catch (error) {
    alert(`Error: ${error.message}`);
  }
}
