@echo off
echo 🚀 Desplegando Job Recommendation API...

REM Parar contenedor existente
echo 📦 Parando contenedor existente...
docker stop job-recommendation-api 2>nul
docker rm job-recommendation-api 2>nul

REM Construir imagen
echo 🔨 Construyendo imagen...
docker build -t job-recommendation-api .

REM Ejecutar contenedor
echo 🌟 Iniciando API...
docker run -d ^
  --name job-recommendation-api ^
  -p 8000:8000 ^
  -v "%cd%\job_similarity_model:/app/job_similarity_model:ro" ^
  --restart unless-stopped ^
  job-recommendation-api

REM Esperar un momento
timeout /t 10 /nobreak > nul

REM Verificar estado
echo 📋 Verificando estado...
docker ps | findstr job-recommendation-api

echo ✅ Despliegue completado!
echo 📖 API disponible en: http://localhost:8000/docs
echo 🏥 Health check: http://localhost:8000/health

pause
