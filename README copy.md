# Proyecto Full Stack - Procesamiento de Imágenes

## Ejecutar

1) En la carpeta app:

```
docker compose up --build
```

2) Abrir en el navegador:

```
http://localhost:8080
```

## Endpoints

- POST /upload (multipart/form-data, campo `files` con múltiples imágenes)
- GET /images (listado acumulado)
- GET /files/originals/{filename}
- GET /files/processed/{filename}
- GET /files/binarized/{filename}

## Notas

- El frontend usa Nginx y hace proxy a `/api` hacia el backend.
- Las URLs retornadas por el backend son relativas y el frontend las consume con `/api`.
- El volumen `data` persiste en `/data` (originales, procesadas, binarizadas, index.json).

## Estructura

```
/app
  /backend
    main.py
    PROCESAMIENTO_IMG.py
    requirements.txt
    Dockerfile
  /frontend
    index.html
    styles.css
    app.js
    Dockerfile
    nginx.conf
  docker-compose.yml
  README.md
```
