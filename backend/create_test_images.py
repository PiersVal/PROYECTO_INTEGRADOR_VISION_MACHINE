#!/usr/bin/env python3
"""Crear imágenes de prueba para test del clustering."""
import os
from PIL import Image, ImageDraw

# Crear directorio si no existe
os.makedirs('test_images', exist_ok=True)

# Crear 5 imágenes de prueba con patrones diferentes
for i in range(5):
    # Crear imagen blanca
    img = Image.new('RGB', (256, 256), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Dibujar círculo con color único
    color = (255, 100 + i*30, 50)
    radius = 50 + i*10
    draw.ellipse([100, 100, 100+radius*2, 100+radius*2], fill=color)
    
    # Agregar algunos rectángulos adicionales
    draw.rectangle([20 + i*10, 20 + i*10, 50 + i*20, 50 + i*20], outline=color, width=3)
    
    # Guardar imagen
    filename = f'test_images/test_{i+1}.png'
    img.save(filename)
    print(f'✓ {filename} creada')

print('\n5 imágenes de prueba creadas en test_images/')
