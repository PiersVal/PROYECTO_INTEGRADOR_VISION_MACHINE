/**
 * ui/gallery.js
 * Responsabilidad: Gestión de la galería de imágenes
 * - Renderizado de galería
 * - Upload de imágenes
 * - Drag & drop
 * - Limpieza
 */

import { state, addItem, clearAll } from '../state.js';
import { uploadImages, getGalleryItems, clearAllImages } from '../api.js';
import { setStatus, resolveUrl } from '../utils.js';

export function initGallery() {
  const fileInput = document.getElementById("file-input");
  const uploadBtn = document.getElementById("upload-btn");
  const clearBtn = document.getElementById("clear-btn");
  const dropZone = document.getElementById("drop-zone");

  // Upload
  if (uploadBtn) {
    uploadBtn.addEventListener("click", async () => {
      const files = fileInput.files;
      if (files.length === 0) {
        alert("Selecciona al menos una imagen");
        return;
      }
      await handleUpload(files);
    });
  }

  // Clear
  if (clearBtn) {
    clearBtn.addEventListener("click", async () => {
      if (!confirm("¿Eliminar todas las imágenes?")) return;
      
      try {
        setStatus("Limpiando...");
        await clearAllImages();
        clearAll();
        renderGallery();
        setStatus("✅ Limpieza completa");
      } catch (error) {
        alert(`Error: ${error.message}`);
        setStatus("❌ Error al limpiar");
      }
    });
  }

  // Drag & drop
  if (dropZone) {
    dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
      dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", async (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        fileInput.files = files;
        await handleUpload(files);
      }
    });
  }

  // Cargar items existentes
  loadExistingItems();
}

async function handleUpload(files) {
  try {
    setStatus("Subiendo imágenes...");
    const data = await uploadImages(files);
    
    data.items.forEach(item => addItem(item));
    renderGallery();
    setStatus(`✅ ${data.items.length} imagen(es) subida(s)`);
  } catch (error) {
    alert(`Error: ${error.message}`);
    setStatus("❌ Error al subir");
  }
}

async function loadExistingItems() {
  try {
    const data = await getGalleryItems();
    state.items = data.items || [];
    renderGallery();
  } catch (error) {
    console.error("Error al cargar items:", error);
  }
}

export function renderGallery() {
  const gallery = document.getElementById("gallery");
  if (!gallery) return;

  gallery.innerHTML = "";

  if (state.items.length === 0) {
    gallery.innerHTML = '<p style="color: #999;">No hay imágenes en la galería</p>';
    return;
  }

  state.items.forEach((item) => {
    const card = document.createElement("div");
    card.className = "card";

    const title = document.createElement("h3");
    title.textContent = item.filename || "imagen";

    const imagesDiv = document.createElement("div");
    imagesDiv.className = "images";

    // Original
    const originalDiv = document.createElement("div");
    const originalImg = document.createElement("img");
    originalImg.src = resolveUrl(item.original_url);
    originalImg.alt = "Original";
    const originalLabel = document.createElement("div");
    originalLabel.className = "image-label";
    originalLabel.textContent = "Original";
    originalDiv.appendChild(originalImg);
    originalDiv.appendChild(originalLabel);

    // Procesada
    const processedDiv = document.createElement("div");
    const processedImg = document.createElement("img");
    processedImg.src = resolveUrl(item.processed_url);
    processedImg.alt = "Procesada";
    const processedLabel = document.createElement("div");
    processedLabel.className = "image-label";
    processedLabel.textContent = "Procesada";
    processedDiv.appendChild(processedImg);
    processedDiv.appendChild(processedLabel);

    // Binarizada
    const binarizedDiv = document.createElement("div");
    const binarizedImg = document.createElement("img");
    binarizedImg.src = resolveUrl(item.binarized_url);
    binarizedImg.alt = "Binarizada";
    const binarizedLabel = document.createElement("div");
    binarizedLabel.className = "image-label";
    binarizedLabel.textContent = "Binarizada";
    binarizedDiv.appendChild(binarizedImg);
    binarizedDiv.appendChild(binarizedLabel);

    imagesDiv.appendChild(originalDiv);
    imagesDiv.appendChild(processedDiv);
    imagesDiv.appendChild(binarizedDiv);

    card.appendChild(title);
    card.appendChild(imagesDiv);
    gallery.appendChild(card);
  });
}
