/**
 * ui/metrics.js
 * Responsabilidad: Visualización de métricas
 * - Mostrar métricas internas (Dunn, Silhouette)
 * - Mostrar métricas externas (ARI, AMI, NMI)
 * - Resumen de clustering
 */

import { state } from '../state.js';

export function displayMetrics(metrics) {
  const resultsSection = document.getElementById("results-section");
  const results = document.getElementById("results");
  
  let metricsContainer = document.getElementById("metrics-display");
  if (!metricsContainer) {
    metricsContainer = document.createElement("div");
    metricsContainer.id = "metrics-display";
    metricsContainer.className = "metrics-container";
    resultsSection.insertBefore(metricsContainer, results);
  }

  const dunn = metrics.dunn_index !== undefined ? Number(metrics.dunn_index).toFixed(4) : "N/A";
  const silhouette = metrics.silhouette_coefficient !== undefined 
    ? Number(metrics.silhouette_coefficient).toFixed(4) : "N/A";

  const totalImages = state.results.length;
  const uniqueClusters = new Set(state.results.map(r => r.cluster_id)).size;

  metricsContainer.innerHTML = `
    <div style="margin-top: 20px; padding: 15px; background: #f3e5f5; border: 2px solid #9c27b0; border-radius: 6px;">
      <h4>📊 Resultados de Métricas Internas:</h4>
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 10px 0;">
        <div style="background: white; padding: 10px; border-radius: 4px; text-align: center;">
          <strong>Dunn Index</strong><br>
          <span style="font-size: 20px; color: #f57c00;">${dunn}</span>
        </div>
        <div style="background: white; padding: 10px; border-radius: 4px; text-align: center;">
          <strong>Silhouette</strong><br>
          <span style="font-size: 20px; color: #0097a7;">${silhouette}</span>
        </div>
      </div>
      <div style="margin-top: 10px; font-size: 12px; color: #555;">
        <strong>Resumen:</strong>
        <ul style="margin: 5px 0; padding-left: 20px;">
          <li>Total de imágenes: ${totalImages}</li>
          <li>Número de clusters: ${uniqueClusters}</li>
          <li>Clusters predichos: ${uniqueClusters}</li>
        </ul>
      </div>
      <hr style="margin: 20px 0; border: none; border-top: 2px solid #9c27b0;">
      <h4 style="margin-top: 20px;">🎯 Clusters Predichos (con Imágenes):</h4>
    </div>
  `;
}

export function displayMetricsWithExternal(metrics, summary) {
  const resultsSection = document.getElementById("results-section");
  const results = document.getElementById("results");
  
  let metricsContainer = document.getElementById("metrics-display");
  if (!metricsContainer) {
    metricsContainer = document.createElement("div");
    metricsContainer.id = "metrics-display";
    metricsContainer.className = "metrics-container";
    resultsSection.insertBefore(metricsContainer, results);
  }

  const dunn = metrics.dunn_index !== undefined ? Number(metrics.dunn_index).toFixed(4) : "N/A";
  const silhouette = metrics.silhouette_coefficient !== undefined 
    ? Number(metrics.silhouette_coefficient).toFixed(4) : "N/A";

  let externalMetricsHTML = "";
  if (metrics.external_metrics) {
    const ari = Number(metrics.external_metrics.ARI).toFixed(4);
    const ami = Number(metrics.external_metrics.AMI).toFixed(4);
    const nmi = Number(metrics.external_metrics.NMI).toFixed(4);
    
    externalMetricsHTML = `
      <h4>📊 Resultados de Métricas Externas:</h4>
      <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin: 10px 0;">
        <div style="background: white; padding: 10px; border-radius: 4px; text-align: center;">
          <strong>ARI</strong><br>
          <span style="font-size: 20px; color: #d32f2f;">${ari}</span>
        </div>
        <div style="background: white; padding: 10px; border-radius: 4px; text-align: center;">
          <strong>AMI</strong><br>
          <span style="font-size: 20px; color: #1976d2;">${ami}</span>
        </div>
        <div style="background: white; padding: 10px; border-radius: 4px; text-align: center;">
          <strong>NMI</strong><br>
          <span style="font-size: 20px; color: #388e3c;">${nmi}</span>
        </div>
      </div>
      <hr style="margin: 20px 0; border: none; border-top: 1px solid #ddd;">
    `;
  }

  const totalImages = summary ? summary.num_images : 0;
  const numClusters = summary ? summary.num_clusters : 0;
  const trueGroups = summary ? summary.true_groups : 0;
  const predictedClusters = summary ? summary.predicted_clusters : 0;

  metricsContainer.innerHTML = `
    <div style="margin-top: 20px; padding: 15px; background: #f3e5f5; border: 2px solid #9c27b0; border-radius: 6px;">
      ${externalMetricsHTML}
      <h4>📊 Resultados de Métricas Internas:</h4>
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 10px 0;">
        <div style="background: white; padding: 10px; border-radius: 4px; text-align: center;">
          <strong>Dunn Index</strong><br>
          <span style="font-size: 20px; color: #f57c00;">${dunn}</span>
        </div>
        <div style="background: white; padding: 10px; border-radius: 4px; text-align: center;">
          <strong>Silhouette</strong><br>
          <span style="font-size: 20px; color: #0097a7;">${silhouette}</span>
        </div>
      </div>
      <div style="margin-top: 10px; font-size: 12px; color: #555;">
        <strong>Resumen:</strong>
        <ul style="margin: 5px 0; padding-left: 20px;">
          <li>Total de imágenes: ${totalImages}</li>
          <li>Número de clusters: ${numClusters}</li>
          ${trueGroups ? `<li>Grupos etiquetados: ${trueGroups}</li>` : ''}
          <li>Clusters predichos: ${predictedClusters}</li>
        </ul>
      </div>
      <hr style="margin: 20px 0; border: none; border-top: 2px solid #9c27b0;">
      <h4 style="margin-top: 20px;">🎯 Clusters Predichos (con Imágenes):</h4>
    </div>
  `;
}
