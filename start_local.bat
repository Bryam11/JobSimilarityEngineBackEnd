@echo off
echo 🚀 Iniciando Job Recommendation API localmente...

REM Verificar si el entorno virtual existe
if not exist "venv" (
    echo 📦 Creando entorno virtual...
    python -m venv venv
)

REM Activar entorno virtual
echo 🔧 Activando entorno virtual...
call venv\Scripts\activate.bat

REM Instalar dependencias
echo 📚 Instalando dependencias...
pip install -r requirements.txtt

REM Ejecutar API
echo 🌟 Iniciando API en http://localhost:8000...
echo 📖 Documentación: http://localhost:8000/docs
echo 🏥 Health check: http://localhost:8000/health
echo.
echo Presiona Ctrl+C para detener la API
echo.

python app_fastapi.py
