# Proyecto Integrador: Evaluación del rendimiento de algoritmos de clustering online de imágenes con restricciones de tamaño

##  Información del Proyecto
- **Carrera:** Ingeniería en Ciencias de la Computación  
- **Nivel:** Séptimo  
- **Materias:** Visión por Computador/ Aprendizaje Automático
-


## Objetivo
Crear un sistema inteligente para agrupamiento online de imágenes con restricciones de tamaño, evaluando el rendimiento de algoritmos de clustering online aplicados a conjuntos de datos con al menos tres clases y 100 instancias.

## Descripción del Proyecto
El proyecto consiste en:
1. Implementar un algoritmo de **clustering online con restricciones de tamaño**.
2. Aplicar tres métodos de extracción de características:
   - Momentos (Momentos, HU, Zernike)
   - SIFT o SURF
   - HOG
3. Usar al menos dos conjuntos de datos con ground truth (mínimo 3 clases, 100 instancias).
4. Calcular métricas de rendimiento internas y externas.
5. Modificar un algoritmo de clustering online para cumplir con restricciones de tamaño.

##  Resultados de Aprendizaje
### Visión por Computador:
- Desarrolla algoritmos para extracción de características globales y locales.
- Implementa preprocesamiento y manipulación de imágenes.
- Extrae descriptores para reconocimiento de patrones.
- Identifica características, espacios de color y elementos de formación de imágenes.

### Aprendizaje Automático:
- Selecciona e implementa algoritmos de búsqueda y clustering.
- Aplica Máquinas de Soporte Vectorial (SVM) para clasificación y regresión.
- Desarrolla algoritmos basados en árboles de decisión.
###

## Instrucciones de ejecución

### Opción recomendada: Docker Compose
**Requisitos:** Docker Desktop instalado.

1. En la raíz del proyecto, ejecutar:
   ```bash
   docker compose up --build
   ```
   Para ejecutar en segundo plano:
   ```bash
   docker compose up -d --build
   ```
2. Abrir la interfaz en el navegador:
   - Frontend: http://localhost:8080
   - Backend (API): http://localhost:8000

Para detener:
```bash
docker compose down
```

Ver logs en tiempo real:
```bash
docker compose logs -f
```

---

### Opción local (sin Docker)
**Requisitos:** Python 3.11+, pip.

#### 1) Backend (FastAPI)
1. Crear y activar un entorno virtual:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. Instalar dependencias:
   ```bash
   pip install -r backend/requirements.txt
   ```
3. Ejecutar el servidor:
   ```bash
   cd backend
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

#### 2) Frontend (HTML/JS)
El frontend espera que el backend esté disponible vía `/api`. Para modo local sin Docker, tienes dos alternativas:

**A. Usar un servidor con proxy `/api` (recomendado).**
Puedes levantar Nginx con una configuración similar a `frontend/nginx.conf`, o usar Docker solo para el frontend.

**B. Cambiar temporalmente el endpoint local.**
Editar `frontend/js/utils.js` y reemplazar:
```js
export const API_BASE = isLocal ? "/api" : "https://remontada-uzn6.onrender.com";
```
por:
```js
export const API_BASE = isLocal ? "http://localhost:8000" : "https://remontada-uzn6.onrender.com";
```

Luego, servir los archivos estáticos:
```bash
cd frontend
python -m http.server 8080
```
Abrir: http://localhost:8080