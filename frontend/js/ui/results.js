/**
 * ui/results.js
 * Responsabilidad: Renderizado de resultados de clustering
 * - Mostrar clusters con imágenes agrupadas
 * - Visualización de centroides
 * - Organización por cluster_id
 */

import { state } from '../state.js';
import { resolveUrl } from '../utils.js';

export function renderResults(resultList, type) {
  const resultsSection = document.getElementById("results-section");
  const results = document.getElementById("results");
  const gallerySection = document.getElementById("gallery-section");
  
  resultsSection.style.display = "block";
  gallerySection.style.display = "none";

  if (!results) return;

  // Si es clustering con agrupación
  if (type === "momentos" || type === "hu" || type === "zernike" || 
      type === "sift" || type === "hog" || type === "cnn") {
    
    resultList.forEach((item) => {
      state.results.push(item);
      const cid = typeof item.cluster_id === "number" ? item.cluster_id : "sin-cluster";
      
      let group = results.querySelector(`[data-cluster-id="${cid}"]`);
      if (!group) {
        group = document.createElement("div");
        group.className = "cluster-group";
        group.dataset.clusterId = cid;

        const title = document.createElement("h3");
        title.textContent = `Cluster ${cid}`;

        const centroid = document.createElement("div");
        centroid.className = "cluster-centroid";
        centroid.textContent = "Centroide: []";

        const grid = document.createElement("div");
        grid.className = "cluster-grid";

        group.appendChild(title);
        group.appendChild(centroid);
        group.appendChild(grid);
        results.appendChild(group);
      }

      // Actualizar centroide con estadísticas y botón expandible
      const centroidEl = group.querySelector(".cluster-centroid");
      if (centroidEl && Array.isArray(item.ultimo_centroide)) {
        const values = item.ultimo_centroide;
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const min = Math.min(...values);
        const max = Math.max(...values);
        const std = Math.sqrt(values.reduce((sq, n) => sq + Math.pow(n - mean, 2), 0) / values.length);
        
        const allValues = values.map(v => v.toFixed(4)).join(", ");
        const uniqueId = `centroid-${cid}-${Date.now()}`;
        const canvasId = `canvas-${uniqueId}`;
        
        centroidEl.innerHTML = `
          <div style="font-size: 11px; display: flex; gap: 15px; color: #555; flex-wrap: wrap; align-items: center;">
            <span>📈 <strong>Media:</strong> ${mean.toFixed(4)}</span>
            <span>📉 <strong>Min:</strong> ${min.toFixed(4)}</span>
            <span>📈 <strong>Max:</strong> ${max.toFixed(4)}</span>
            <span>📏 <strong>σ:</strong> ${std.toFixed(4)}</span>
            <button id="btn-${uniqueId}" onclick="
              const full = document.getElementById('full-${uniqueId}');
              const isHidden = full.style.display === 'none';
              full.style.display = isHidden ? 'block' : 'none';
              this.textContent = isHidden ? '📋 Ver menos' : '📋 Ver más';
            " style="
              font-size: 10px; 
              padding: 3px 8px; 
              background: #2196F3; 
              color: white; 
              border: none; 
              border-radius: 3px; 
              cursor: pointer;
              transition: background 0.2s;
            " onmouseover="this.style.background='#1976d2'" onmouseout="this.style.background='#2196F3'">
              📋 Ver más
            </button>
          </div>
          <canvas id="${canvasId}" width="400" height="60" style="
            margin-top: 8px;
            border: 1px solid #ddd; 
            border-radius: 4px; 
            background: #fafafa;
            cursor: crosshair;
          "></canvas>
          <div id="full-${uniqueId}" style="
            display: none; 
            margin-top: 8px; 
            padding: 8px; 
            background: #f5f5f5; 
            border-radius: 4px; 
            font-size: 10px; 
            max-height: 120px; 
            overflow-y: auto;
            color: #333;
            font-family: monospace;
            word-break: break-all;
          ">
            [${allValues}]
          </div>
        `;
        
        // Dibujar el gráfico canvas
        setTimeout(() => {
          const canvas = document.getElementById(canvasId);
          if (!canvas) return;
          
          const ctx = canvas.getContext('2d');
          const displayValues = values.slice(0, 200);
          const maxAbs = Math.max(...values.map(Math.abs));
          
          ctx.clearRect(0, 0, 400, 60);
          
          // Línea central (eje Y=0)
          ctx.strokeStyle = '#ddd';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(0, 30);
          ctx.lineTo(400, 30);
          ctx.stroke();
          
          // Señal del centroide
          ctx.strokeStyle = '#2196F3';
          ctx.lineWidth = 2;
          ctx.beginPath();
          
          displayValues.forEach((v, i) => {
            const x = (i / displayValues.length) * 400;
            const y = 30 - (v / maxAbs) * 25;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
          });
          
          ctx.stroke();
          
          // Puntos en la señal
          ctx.fillStyle = '#1976d2';
          displayValues.forEach((v, i) => {
            if (i % Math.max(1, Math.floor(displayValues.length / 30)) === 0) {
              const x = (i / displayValues.length) * 400;
              const y = 30 - (v / maxAbs) * 25;
              ctx.beginPath();
              ctx.arc(x, y, 2, 0, Math.PI * 2);
              ctx.fill();
            }
          });
          
          // Tooltip interactivo
          canvas.addEventListener('mousemove', (e) => {
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const index = Math.floor((x / 400) * displayValues.length);
            if (index >= 0 && index < displayValues.length) {
              canvas.title = `Índice: ${index} | Valor: ${displayValues[index].toFixed(6)}`;
            }
          });
        }, 50);
      }

      // Agregar imagen
      const grid = group.querySelector(".cluster-grid");
      const tile = document.createElement("div");
      tile.className = "cluster-item";

      const img = document.createElement("img");
      // Usar imágenes procesadas con visualización (para SIFT, HOG, etc.)
      const imageUrl = item.binarized_url || item.processed_url || item.original_url;
      img.src = resolveUrl(imageUrl);
      img.alt = item.filename || "imagen";

      const caption = document.createElement("div");
      caption.className = "cluster-caption";
      caption.textContent = item.filename || "imagen";

      tile.appendChild(img);
      tile.appendChild(caption);
      grid.appendChild(tile);
    });
  }
}

export function clearResultsDisplay() {
  const results = document.getElementById("results");
  const resultsSection = document.getElementById("results-section");
  
  if (results) {
    results.innerHTML = "";
  }
  
  // Limpiar estado de resultados
  state.results = [];
  
  // Ocultar sección de resultados al cambiar de modo
  if (resultsSection) {
    resultsSection.style.display = "none";
  }
}

export function renderClusterVisualization(clustersData, containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;
  
  container.innerHTML = "";
  
  const sortedClusters = Object.keys(clustersData).sort((a, b) => parseInt(a) - parseInt(b));
  
  sortedClusters.forEach(clusterId => {
    const images = clustersData[clusterId];
    
    // Usar el estilo moderno estándar
    const clusterDiv = document.createElement("div");
    clusterDiv.className = "cluster-group";
    clusterDiv.dataset.clusterId = clusterId;
    
    // Título
    const title = document.createElement("h3");
    title.textContent = `Cluster ${clusterId}`;
    clusterDiv.appendChild(title);
    
    // Info
    const info = document.createElement("div");
    info.className = "cluster-centroid";
    info.textContent = `${images.length} imágenes en este cluster`;
    clusterDiv.appendChild(info);
    
    // Grid
    const grid = document.createElement("div");
    grid.className = "cluster-grid";
    
    images.forEach(img => {
      const tile = document.createElement("div");
      tile.className = "cluster-item";
      
      // Imagen
      const imgEl = document.createElement("img");
      const imageUrl = img.binarized_url || img.processed_url || img.original_url;
      imgEl.src = resolveUrl(imageUrl);
      imgEl.alt = img.filename || "imagen";
      tile.appendChild(imgEl);
      
      // Badge de etiqueta real (si existe)
      if (img.true_label) {
        const labelBadge = document.createElement("div");
        labelBadge.className = "cluster-caption";
        labelBadge.innerHTML = `<span style="display: inline-block; background: #667eea; color: white; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; margin-bottom: 4px;">🏷️ ${img.true_label}</span>`;
        tile.appendChild(labelBadge);
      }
      
      // Nombre del archivo
      const caption = document.createElement("div");
      caption.className = "cluster-caption";
      caption.textContent = img.filename || "imagen";
      tile.appendChild(caption);
      
      grid.appendChild(tile);
    });
    
    clusterDiv.appendChild(grid);
    container.appendChild(clusterDiv);
  });
}
