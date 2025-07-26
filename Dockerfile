FROM python:3.11-slim

# Establecer directorio de trabajo
WORKDIR /app

# Variables de entorno para optimización
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8080

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copiar requirements e instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación y modelo
COPY app_fastapi.py .

# Crear directorio y copiar archivos del modelo de forma explícita
RUN mkdir -p job_recommendation_model
COPY job_recommendation_model/*.csv ./job_recommendation_model/
COPY job_recommendation_model/*.pkl ./job_recommendation_model/
COPY job_recommendation_model/*.joblib ./job_recommendation_model/
COPY job_recommendation_model/ ./job_recommendation_model/

# Verificar que los archivos se copiaron correctamente
RUN echo "=== VERIFICACIÓN DE ARCHIVOS ===" && \
    ls -la job_recommendation_model/ && \
    echo "=== ARCHIVOS CSV ===" && \
    find job_recommendation_model/ -name "*.csv" -ls && \
    echo "=== ARCHIVOS PKL ===" && \
    find job_recommendation_model/ -name "*.pkl" -ls && \
    echo "=== TAMAÑO DE ARCHIVOS ===" && \
    du -sh job_recommendation_model/* && \
    echo "=== VERIFICACIÓN COMPLETA ==="

# Exponer puerto
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:$PORT/api/model/health || exit 1

# Comando para ejecutar la aplicación con timeout aumentado
CMD exec uvicorn app_fastapi:app --host 0.0.0.0 --port $PORT --timeout-keep-alive 60
