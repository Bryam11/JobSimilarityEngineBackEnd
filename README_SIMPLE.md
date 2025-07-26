# Job Recommendation Microservice

Microservicio simple con FastAPI para recomendación de empleos. Diseñado para ser consumido por un gateway que maneja la seguridad.

## 🚀 Características

- **API REST** con FastAPI
- **Sin autenticación** (manejada por el gateway)
- **Múltiples métodos de búsqueda**
- **Dockerizado** para fácil despliegue
- **Health checks** integrados

## 📋 Endpoints

- `POST /recommend` - Obtener recomendaciones
- `GET /jobs` - Listar empleos (paginado)
- `GET /jobs/{id}` - Detalles de empleo
- `GET /health` - Health check
- `GET /docs` - Documentación

## 🐳 Despliegue con Docker (Recomendado)

```bash
# Script automatizado
.\deploy.bat

# O manual
docker build -t job-recommendation-api .
docker run -p 8000:8000 job-recommendation-api
```

## 💻 Ejecución Local

```bash
# Script automatizado
.\start_local.bat

# O manual
venv\Scripts\activate
pip install -r requirements.txt
python app_fastapi.py
```

## 📚 Uso

### Recomendaciones
```bash
curl -X POST "http://localhost:8000/recommend" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Data Scientist",
       "top_n": 5,
       "method": "hybrid"
     }'
```

### Health Check
```bash
curl http://localhost:8000/health
```

## 🔧 Métodos de Búsqueda

- `title_only` - Solo títulos (rápido)
- `combined` - Títulos + descripciones (completo)
- `hybrid` - Combinado ponderado (recomendado)

## 📖 Documentación

- **Swagger**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

El microservicio está listo para ser consumido por tu gateway!
