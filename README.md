# Job Recommendation API

API REST avanzada con FastAPI para recomendación de empleos usando NLP y similitud semántica.

## 🚀 Características

- **API REST** con FastAPI y documentación automática
- **Múltiples métodos de búsqueda**: title_only, combined, hybrid
- **Modelo de ML** usando TF-IDF y similitud coseno
- **Compatibilidad backward** con modelos básicos
- **Dockerizado** para fácil despliegue
- **Nginx** como reverse proxy
- **Health checks** y logging
- **Paginación** para listas grandes

## 📋 Endpoints Disponibles

### Información de la API
- `GET /` - Información general de la API
- `GET /health` - Verificación de salud
- `GET /docs` - Documentación Swagger

### Empleos
- `POST /api/similar-jobs` - Buscar empleos similares
- `GET /api/jobs` - Obtener todos los empleos disponibles
- `GET /api/model-info` - Información del modelo ML

## 🛠️ Instalación Local

### 1. Clonar y preparar entorno
```bash
cd JobSimilarityEngineBackEnd
python -m venv venv

# Windows
venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Ejecutar la API
```bash
# API simple (recomendado)
python simple_api.py

# O con uvicorn
uvicorn simple_api:app --reload --host 0.0.0.0 --port 8000
```

La API estará disponible en: http://localhost:8000

## � Despliegue con Docker

### Construcción y ejecución
```bash
# Construir imagen
docker build -t job-similarity-api .

# Ejecutar contenedor
docker run -p 8000:8000 job-similarity-api
```

### Con Docker Compose
```bash
docker-compose up --build
```

## � Uso de la API

### Buscar empleos similares
```http
POST /api/similar-jobs
Content-Type: application/json

{
    "title": "Data Scientist",
    "limit": 10,
    "min_similarity": 0.1
}
```

**Respuesta:**
```json
{
    "query": "Data Scientist",
    "results": [
        {
            "job_id": 123,
            "title": "Senior Data Scientist",
            "company": "Tech Corp",
            "location": "New York",
            "similarity_score": 0.85
        }
    ],
    "total_results": 5
}
```

### Obtener todos los empleos
```http
GET /api/jobs
```

### Información del modelo
```http
GET /api/model-info
```

## 🧪 Testing

```bash
# Probar la API (después de que esté corriendo)
python test_simple_api.py
```

### Ejemplo con curl
```bash
# Buscar empleos similares
curl -X POST "http://localhost:8000/api/similar-jobs" \
     -H "Content-Type: application/json" \
     -d '{
       "title": "Software Engineer",
       "limit": 5
     }'

# Obtener todos los empleos
curl "http://localhost:8000/api/jobs"

# Información del modelo
curl "http://localhost:8000/api/model-info"
```

## 📁 Estructura del Proyecto

```
JobSimilarityEngineBackEnd/
├── job_similarity_model/      # Modelo ML preentrenado
│   ├── job_similarity_utils.py
│   ├── tfidf_vectorizer.pkl
│   ├── tfidf_matrix.pkl
│   ├── jobs_data.csv
│   └── ...
├── simple_api.py             # API principal (sin autenticación)
├── main.py                   # API completa (con autenticación)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── test_simple_api.py
└── README.md
```

## � Características del Modelo

- **Empleos indexados**: 976
- **Algoritmo**: TF-IDF + Similitud Coseno
- **Vocabulario**: 380 términos únicos
- **Formato de entrada**: Títulos de trabajo en texto
- **Formato de salida**: Lista de empleos similares con scores

## 🚀 Despliegue en Producción

### Variables de entorno (opcionales)
```bash
# Puerto (default: 8000)
export PORT=8000

# Host (default: 0.0.0.0)
export HOST=0.0.0.0
```

### Con Gunicorn
```bash
pip install gunicorn
gunicorn simple_api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: job-similarity-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: job-similarity-api
  template:
    metadata:
      labels:
        app: job-similarity-api
    spec:
      containers:
      - name: api
        image: job-similarity-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
```

## 🔍 Troubleshooting

### Problema: Modelo no carga
- Verificar que la carpeta `job_similarity_model` existe
- Verificar que todos los archivos `.pkl` están presentes
- Verificar versiones de scikit-learn y pandas

### Problema: Error de dependencias
```bash
pip install --upgrade scikit-learn==1.6.1
```

### Problema: Docker build falla
- Verificar que Docker tiene suficiente memoria
- Verificar que todos los archivos están en el contexto

## 📄 Licencia

Este proyecto está bajo la Licencia MIT.
