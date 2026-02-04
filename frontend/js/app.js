

/**
 * app.js
 * Responsabilidad: Punto de entrada principal
 * - Inicializar todos los módulos
 * - Coordinar la aplicación
 * - Separar modos: galería, sin etiquetas, con etiquetas
 */

import { initSidebar } from './ui/sidebar.js';
import { initGallery } from './ui/gallery.js';
import { clearResultsDisplay } from './ui/results.js';
import { initMomentos, analyzeMomentos } from './clustering/momentos.js';
import { initHu, analyzeHu } from './clustering/hu.js';
import { initZernike, analyzeZernike } from './clustering/zernike.js';
import { initSift, analyzeSift } from './clustering/sift.js';
import { initHog, analyzeHog } from './clustering/hog.js';
import { initCnn, analyzeCnn } from './clustering/cnn.js';
import { initExternalMetrics } from './supervised/external-metrics.js';
import { initExternalMetricsHu } from './supervised/external-metrics-hu.js';
import { initExternalMetricsZernike } from './supervised/external-metrics-zernike.js';
import { initExternalMetricsSift } from './supervised/external-metrics-sift.js';
import { initExternalMetricsHog } from './supervised/external-metrics-hog.js';
import { initExternalMetricsCnn } from './supervised/external-metrics-cnn.js';
import { setStatus } from './utils.js';

// Inicializar aplicación
document.addEventListener('DOMContentLoaded', () => {
  console.log('🚀 Iniciando aplicación...');
  
  // Inicializar UI
  initSidebar();
  initGallery();
  
  // Inicializar módulos de clustering (sin etiquetas)
  initMomentos();
  initHu();
  initZernike();
  initSift();
  initHog();
  initCnn();
  
  // Inicializar módulos supervisados (con etiquetas)
  initExternalMetrics();
  initExternalMetricsHu();
  initExternalMetricsZernike();
  initExternalMetricsSift();
  initExternalMetricsHog();
  initExternalMetricsCnn();
  
  // Event listeners para cambio de modo
  initModeListeners();
  
  console.log('✅ Aplicación iniciada');
});

function initModeListeners() {
  const modeRadios = document.querySelectorAll('input[name="mode"]');
  const fileInput = document.getElementById("file-input");
  
  modeRadios.forEach((radio) => {
    radio.addEventListener("change", () => {
      const mode = radio.value;
      
      // Limpiar resultados anteriores al cambiar de modo
      clearResultsDisplay();
      
      // Ocultar todas las configuraciones
      document.querySelectorAll('.mode-config').forEach(config => {
        config.style.display = 'none';
      });
      
      // Limpiar el input de archivos
      const fileInput = document.getElementById("file-input");
      if (fileInput) {
        fileInput.value = '';
      }
      
      // Determinar si es modo galería, sin etiquetas o con etiquetas
      const isGallery = mode === 'gallery';
      const isSupervised = mode.startsWith('external-metrics');
      const isUnsupervised = !isGallery && !isSupervised;
      
      // Mostrar/ocultar secciones según el modo
      const gallerySection = document.getElementById("gallery-section");
      const resultsSection = document.getElementById("results-section");
      
      if (isGallery) {
        // Modo galería: solo mostrar galería
        gallerySection.style.display = "block";
        resultsSection.style.display = "none";
      } else {
        // Modos clustering (con o sin etiquetas): ocultar galería
        gallerySection.style.display = "none";
        resultsSection.style.display = "none"; // Se mostrará después del análisis
        
        // Mostrar configuración correspondiente
        const configMap = {
          'momentos': 'momentos-config',
          'hu': 'hu-config',
          'zernike': 'zernike-config',
          'sift': 'sift-config',
          'hog': 'hog-config',
          'cnn': 'cnn-config',
          'external-metrics': 'external-metrics-config',
          'external-metrics-hu': 'external-hu-config',
          'external-metrics-zernike': 'external-zernike-config',
          'external-metrics-sift': 'external-sift-config',
          'external-metrics-hog': 'external-hog-config',
          'external-metrics-cnn': 'external-cnn-config',
        };
        
        const configId = configMap[mode];
        const configEl = document.getElementById(configId);
        if (configEl) {
          configEl.style.display = 'flex';
        }
      }
    });
  });
  
  // Listener para botón subir (solo para modos SIN ETIQUETAS)
  const uploadBtn = document.getElementById("upload-btn");
  if (uploadBtn) {
    uploadBtn.addEventListener("click", async () => {
      const files = fileInput.files;
      if (files.length === 0) {
        alert("Selecciona al menos una imagen");
        return;
      }
      
      const mode = document.querySelector('input[name="mode"]:checked')?.value;
      
      // Solo para modos sin etiquetas
      if (mode && !mode.startsWith('external-metrics')) {
        // Obtener capacidades según el método
        let capacitiesInput;
        if (mode === 'momentos') {
          capacitiesInput = document.getElementById("capacities-input");
        } else {
          capacitiesInput = document.getElementById(`${mode}-capacities-input`);
        }
        
        const capacities = capacitiesInput?.value?.trim() || null;
        
        // Rutear al método correcto
        switch (mode) {
          case 'momentos':
            await analyzeMomentos(files, capacities);
            break;
          case 'hu':
            await analyzeHu(files, capacities);
            break;
          case 'zernike':
            await analyzeZernike(files, capacities);
            break;
          case 'sift':
            await analyzeSift(files, capacities);
            break;
          case 'hog':
            await analyzeHog(files, capacities);
            break;
          case 'cnn':
            await analyzeCnn(files, capacities);
            break;
          default:
            setStatus("Método no implementado");
        }
        
        // Mostrar sección de resultados después del análisis
        document.getElementById("results-section").style.display = "block";
      }
    });
  }
}
