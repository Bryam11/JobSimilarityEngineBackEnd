#!/bin/bash

echo "🚀 Desplegando Job Recommendation API con Docker..."

# Parar contenedores existentes
echo "📦 Parando contenedores existentes..."
docker-compose down

# Construir imagen
echo "🔨 Construyendo imagen Docker..."
docker-compose build --no-cache

# Iniciar servicios
echo "🌟 Iniciando servicios..."
docker-compose up -d

# Esperar a que los servicios estén listos
echo "⏳ Esperando a que los servicios estén listos..."
sleep 10

# Mostrar logs
echo "📋 Mostrando logs..."
docker-compose logs job-recommendation-api

echo "✅ Despliegue completado!"
echo "📖 API disponible en: http://localhost:8000/docs"
echo "🏥 Health check: http://localhost:8000/health"

# Probar la API
echo "🧪 Probando la API..."
curl -f http://localhost:8000/health || echo "❌ API no disponible aún"
