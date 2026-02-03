/**
 * ui/sidebar.js
 * Responsabilidad: Lógica del menú lateral
 * - Cambio entre modos (galería, sin etiquetas, con etiquetas)
 * - Populación dinámica de métodos
 * - Activación de radio buttons
 */

import { unsupervisedMethods, supervisedMethods } from '../utils.js';

export function initSidebar() {
  const mainModeSelect = document.getElementById("main-mode-select");
  const methodSelect = document.getElementById("method-select");
  const methodSection = document.getElementById("method-section");
  const gallerySection = document.getElementById("gallery-section");
  const resultsSection = document.getElementById("results-section");

  // Listener para el combo principal
  mainModeSelect.addEventListener("change", function() {
    const mainMode = this.value;
    methodSelect.innerHTML = '<option value="">-- Seleccionar método --</option>';
    
    if (mainMode === "gallery") {
      // Modo galería
      methodSection.style.display = "none";
      gallerySection.style.display = "block";
      resultsSection.style.display = "none";
      
      const galleryRadio = document.querySelector('input[name="mode"][value="gallery"]');
      if (galleryRadio) {
        galleryRadio.checked = true;
        galleryRadio.dispatchEvent(new Event("change"));
      }
    } else if (mainMode === "unsupervised") {
      // Poblar con métodos sin etiquetas
      gallerySection.style.display = "none";
      Object.entries(unsupervisedMethods).forEach(([key, method]) => {
        const option = document.createElement("option");
        option.value = method.value;
        option.textContent = method.label;
        methodSelect.appendChild(option);
      });
      methodSection.style.display = "block";
    } else if (mainMode === "supervised") {
      // Poblar con métodos con etiquetas
      gallerySection.style.display = "none";
      Object.entries(supervisedMethods).forEach(([key, method]) => {
        const option = document.createElement("option");
        option.value = method.value;
        option.textContent = method.label;
        methodSelect.appendChild(option);
      });
      methodSection.style.display = "block";
    } else {
      methodSection.style.display = "none";
    }
  });

  // Listener para el combo de métodos
  methodSelect.addEventListener("change", function() {
    const selectedMode = this.value;
    if (selectedMode) {
      const radioToActivate = document.querySelector(`input[name="mode"][value="${selectedMode}"]`);
      if (radioToActivate) {
        radioToActivate.checked = true;
        radioToActivate.dispatchEvent(new Event("change"));
      }
    }
  });

  // Inicializar modo galería por defecto
  setTimeout(() => {
    mainModeSelect.dispatchEvent(new Event("change"));
  }, 100);
}
