from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import pickle
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import logging
import uvicorn
import os
from pathlib import Path
import re

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Definir TextPreprocessor para compatibilidad con pickle
class TextPreprocessor:
    def __init__(self):
        self.stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
            'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with',
            'i', 'ii', 'iii', 'iv', 'v', 'sr', 'jr', '&', 'amp'
        }

    def preprocess_text(self, text):
        """Función básica de preprocesamiento de texto"""
        if pd.isna(text):
            return ""

        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text)

        words = text.split()
        words = [word for word in words if word not in self.stopwords and len(word) > 1]

        return ' '.join(words)

# Modelos Pydantic para validación de datos
class JobRecommendationRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Consulta de búsqueda de empleo")
    top_n: int = Field(default=10, ge=1, le=100, description="Número de recomendaciones a devolver")
    method: Literal['title_only', 'combined', 'hybrid'] = Field(default='hybrid', description="Método de búsqueda")

class JobRecommendation(BaseModel):
    rank: int
    id: int
    title: str
    company: str
    location: str
    description: Optional[str] = None
    work_type: Optional[str] = None
    employment_type: Optional[str] = None
    similarity_score: float

class RecommendationResponse(BaseModel):
    success: bool
    query: str
    method: str
    total_results: int
    recommendations: List[JobRecommendation]
    metadata: dict

class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    status: str
    model_loaded: bool
    total_jobs: int
    version: str = "1.0"

class BatchPredictionRequest(BaseModel):
    titles: List[str] = Field(..., min_items=1, max_items=50, description="Lista de títulos de empleo a buscar")

class BatchPredictionResponse(BaseModel):
    success: bool
    total_queries: int
    results: List[dict]

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await recommendation_service.load_model()
    yield
    # Shutdown (if needed)
    pass

# Crear aplicación FastAPI
app = FastAPI(
    title="Job Recommendation API",
    description="API de recomendación de empleos usando NLP y similitud semántica",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica dominios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class JobRecommendationService:
    def __init__(self, model_dir: str = "job_recommendation_model"):
        self.model_dir = model_dir
        self.df = None
        self.tfidf_title = None
        self.tfidf_combined = None
        self.title_matrix = None
        self.combined_matrix = None
        self.preprocessor = None
        self.is_loaded = False
        
        # Adaptación para el modelo actual
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.jobs_data = None
        
        # Para el modelo completo pkl
        self.model = None
        self.model_type = None  # 'complete' o 'components'
    
    async def load_model(self):
        """Cargar modelo de forma asíncrona"""
        try:
            logger.info("🔄 Cargando modelo...")
            
            # Intentar cargar el modelo completo primero (job_recommendation_system.pkl)
            try:
                with open(f"{self.model_dir}/job_recommendation_system.pkl", "rb") as f:
                    self.model = pickle.load(f)
                
                with open(f"{self.model_dir}/text_preprocessor.pkl", "rb") as f:
                    self.preprocessor = pickle.load(f)
                
                # Cargar datos para los otros endpoints
                self.df = pd.read_csv(f"{self.model_dir}/processed_jobs_data.csv")
                
                self.model_type = 'complete'
                self.is_loaded = True
                logger.info("✅ Modelo completo cargado exitosamente")
                return
                
            except FileNotFoundError:
                logger.info("Modelo completo no encontrado, intentando cargar componentes individuales...")
            
            # Intentar cargar el nuevo modelo por componentes
            try:
                self.df = pd.read_csv(f"{self.model_dir}/processed_jobs_data.csv")
                self.tfidf_title = joblib.load(f"{self.model_dir}/tfidf_title_vectorizer.pkl")
                self.tfidf_combined = joblib.load(f"{self.model_dir}/tfidf_combined_vectorizer.pkl")
                self.title_matrix = joblib.load(f"{self.model_dir}/title_tfidf_matrix.pkl")
                self.combined_matrix = joblib.load(f"{self.model_dir}/combined_tfidf_matrix.pkl")
                
                # Intentar cargar preprocessor, si falla crear uno nuevo
                try:
                    with open(f"{self.model_dir}/text_preprocessor.pkl", 'rb') as f:
                        self.preprocessor = pickle.load(f)
                except (AttributeError, ModuleNotFoundError, pickle.PickleError) as e:
                    logger.warning(f"No se pudo cargar preprocessor desde pickle: {e}")
                    logger.info("Creando nuevo preprocessor...")
                    self.preprocessor = TextPreprocessor()

                self.model_type = 'components'
                logger.info("✅ Modelo avanzado cargado")
                
            except (FileNotFoundError, Exception) as e:
                # Fallback al modelo básico existente
                logger.info(f"🔄 Cargando modelo básico como fallback... Error: {e}")
                try:
                    self.jobs_data = pd.read_csv(f"{self.model_dir}/processed_jobs_data.csv")
                    # Buscar archivos de vectorizador disponibles
                    if os.path.exists(f"{self.model_dir}/tfidf_vectorizer.pkl"):
                        self.tfidf_vectorizer = joblib.load(f"{self.model_dir}/tfidf_vectorizer.pkl")
                        self.tfidf_matrix = joblib.load(f"{self.model_dir}/tfidf_matrix.pkl")
                    else:
                        # Si no existe el vectorizador básico, usar el combinado como fallback
                        logger.info("Usando vectorizador combinado como fallback...")
                        self.tfidf_vectorizer = joblib.load(f"{self.model_dir}/tfidf_combined_vectorizer.pkl")
                        self.tfidf_matrix = joblib.load(f"{self.model_dir}/combined_tfidf_matrix.pkl")

                    self.df = self.jobs_data  # Para compatibilidad
                    self.preprocessor = TextPreprocessor()
                    self.model_type = 'basic'
                    logger.info("✅ Modelo básico cargado")
                except Exception as fallback_error:
                    logger.error(f"Error en fallback: {fallback_error}")
                    raise fallback_error

            self.is_loaded = True
            logger.info(f"✅ Modelo cargado exitosamente. {len(self.df)} empleos disponibles")
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            # En lugar de lanzar HTTPException, crear un modelo mínimo funcional
            try:
                logger.info("🔄 Intentando crear modelo mínimo funcional...")
                self.df = pd.read_csv(f"{self.model_dir}/processed_jobs_data.csv")
                self.preprocessor = TextPreprocessor()

                # Crear un vectorizador básico si no existe
                from sklearn.feature_extraction.text import TfidfVectorizer
                self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

                # Crear matriz básica usando descripciones o títulos disponibles
                if 'combined_processed' in self.df.columns:
                    text_data = self.df['combined_processed'].fillna('')
                elif 'description' in self.df.columns:
                    text_data = self.df['description'].fillna('').apply(self.clean_text_basic)
                else:
                    text_data = self.df['title'].fillna('').apply(self.clean_text_basic)

                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_data)
                self.model_type = 'minimal'
                self.is_loaded = True
                logger.info("✅ Modelo mínimo funcional creado")

            except Exception as final_error:
                logger.error(f"❌ Error fatal: {final_error}")
                raise HTTPException(status_code=500, detail=f"Error loading model: {str(final_error)}")

    def clean_text_basic(self, text):
        """Función básica de limpieza de texto"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Stopwords básicas
        stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 
            'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with',
            'i', 'ii', 'iii', 'iv', 'v', 'sr', 'jr', '&', 'amp'
        }
        
        words = text.split()
        words = [word for word in words if word not in stopwords and len(word) > 1]
        
        return ' '.join(words)
    
    async def get_recommendations(self, request: JobRecommendationRequest) -> List[JobRecommendation]:
        """Obtener recomendaciones de empleos"""
        if not self.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        try:
            # Si tenemos el modelo completo, usar esa lógica
            if self.model_type == 'complete' and self.model is not None:
                return await self._get_complete_model_recommendations(request)
            # Si tenemos el modelo avanzado, usar esa lógica
            elif hasattr(self, 'tfidf_title') and self.tfidf_title is not None:
                return await self._get_advanced_recommendations(request)
            else:
                # Usar el modelo básico
                return await self._get_basic_recommendations(request)
            
        except Exception as e:
            logger.error(f"Error en recomendación: {e}")
            raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")
    
    async def _get_complete_model_recommendations(self, request: JobRecommendationRequest) -> List[JobRecommendation]:
        """Lógica para el modelo completo pkl"""
        try:
            # Preprocesar la consulta usando el método transform del preprocessor
            processed_query = self.preprocessor.transform([request.query])
            
            # Obtener recomendaciones del modelo
            results = self.model.recommend(processed_query, top_n=request.top_n)
            
            # Convertir resultados al formato esperado
            recommendations = []
            for i, result in enumerate(results[:request.top_n]):
                # Ajustar según la estructura exacta de tu resultado
                if isinstance(result, dict):
                    job_data = JobRecommendation(
                        rank=i + 1,
                        id=int(result.get('id', i)),
                        title=result.get('title', ''),
                        company=result.get('company', ''),
                        location=result.get('location', ''),
                        description=result.get('description', ''),
                        work_type=result.get('work_type', ''),
                        employment_type=result.get('employment_type', ''),
                        similarity_score=float(result.get('similarity_score', 0.0))
                    )
                elif isinstance(result, (list, tuple)) and len(result) >= 4:
                    # Si el resultado es una tupla o lista
                    job_data = JobRecommendation(
                        rank=i + 1,
                        id=int(result[0]) if len(result) > 0 else i,
                        title=str(result[1]) if len(result) > 1 else '',
                        company=str(result[2]) if len(result) > 2 else '',
                        location=str(result[3]) if len(result) > 3 else '',
                        description=str(result[4]) if len(result) > 4 else '',
                        work_type=str(result[5]) if len(result) > 5 else '',
                        employment_type=str(result[6]) if len(result) > 6 else '',
                        similarity_score=float(result[-1]) if len(result) > 0 else 0.0
                    )
                else:
                    # Fallback básico
                    job_data = JobRecommendation(
                        rank=i + 1,
                        id=i,
                        title=str(result),
                        company="",
                        location="",
                        similarity_score=0.0
                    )
                
                recommendations.append(job_data)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error en modelo completo: {e}")
            # Fallback a método básico si falla
            return await self._get_basic_recommendations(request)
    
    async def _get_advanced_recommendations(self, request: JobRecommendationRequest) -> List[JobRecommendation]:
        """Lógica para el modelo avanzado"""
        processed_query = self.preprocessor.preprocess_text(request.query)
        
        if request.method == 'title_only':
            query_vector = self.tfidf_title.transform([processed_query])
            similarity_scores = cosine_similarity(query_vector, self.title_matrix).flatten()
        elif request.method == 'combined':
            query_vector = self.tfidf_combined.transform([processed_query])
            similarity_scores = cosine_similarity(query_vector, self.combined_matrix).flatten()
        else:  # hybrid
            title_vector = self.tfidf_title.transform([processed_query])
            combined_vector = self.tfidf_combined.transform([processed_query])
            
            title_scores = cosine_similarity(title_vector, self.title_matrix).flatten()
            combined_scores = cosine_similarity(combined_vector, self.combined_matrix).flatten()
            
            similarity_scores = 0.3 * title_scores + 0.7 * combined_scores
        
        # Obtener top N recomendaciones
        top_indices = similarity_scores.argsort()[-request.top_n:][::-1]
        
        recommendations = []
        for i, idx in enumerate(top_indices):
            job_data = JobRecommendation(
                rank=i + 1,
                id=int(self.df.iloc[idx]['id']),
                title=self.df.iloc[idx]['title'],
                company=self.df.iloc[idx]['company'],
                location=self.df.iloc[idx].get('location', ''),
                description=self.df.iloc[idx].get('description', ''),
                work_type=self.df.iloc[idx].get('work_type', ''),
                employment_type=self.df.iloc[idx].get('employment_type', ''),
                similarity_score=float(similarity_scores[idx])
            )
            recommendations.append(job_data)
        
        return recommendations
    
    async def _get_basic_recommendations(self, request: JobRecommendationRequest) -> List[JobRecommendation]:
        """Lógica para el modelo básico"""
        # Limpiar consulta
        cleaned_query = self.clean_text_basic(request.query)
        
        if not cleaned_query.strip():
            return []
        
        # Vectorizar consulta
        query_vector = self.tfidf_vectorizer.transform([cleaned_query])
        
        # Calcular similitudes
        similarity_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Obtener resultados ordenados
        top_indices = similarity_scores.argsort()[-request.top_n:][::-1]
        
        recommendations = []
        for i, idx in enumerate(top_indices):
            if similarity_scores[idx] >= 0.1:  # Umbral mínimo
                job_data = JobRecommendation(
                    rank=i + 1,
                    id=int(self.df.iloc[idx]['id']),
                    title=self.df.iloc[idx]['title'],
                    company=self.df.iloc[idx]['company'],
                    location=self.df.iloc[idx].get('location', ''),
                    description=self.df.iloc[idx].get('description', ''),
                    work_type=self.df.iloc[idx].get('work_type', ''),
                    employment_type=self.df.iloc[idx].get('employment_type', ''),
                    similarity_score=float(similarity_scores[idx])
                )
                recommendations.append(job_data)
        
        return recommendations

# Inicializar servicio
recommendation_service = JobRecommendationService()

@app.get("/api/model/health", response_model=HealthResponse)
async def health_check():
    """Endpoint de salud"""
    return HealthResponse(
        status="healthy",
        model_loaded=recommendation_service.is_loaded,
        total_jobs=len(recommendation_service.df) if recommendation_service.df is not None else 0
    )

@app.post("/api/model/recommend", response_model=RecommendationResponse)
async def recommend_jobs(request: JobRecommendationRequest):
    """Endpoint principal para obtener recomendaciones"""
    try:
        recommendations = await recommendation_service.get_recommendations(request)
        
        return RecommendationResponse(
            success=True,
            query=request.query,
            method=request.method,
            total_results=len(recommendations),
            recommendations=recommendations,
            metadata={
                "total_jobs_in_db": len(recommendation_service.df),
                "search_method": request.method,
                "top_n_requested": request.top_n
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en /recommend: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model/search-methods")
async def get_search_methods():
    """Obtener información sobre los métodos de búsqueda disponibles"""
    return {
        "methods": {
            "title_only": {
                "name": "Búsqueda por Título",
                "description": "Busca similitud solo en títulos de puestos",
                "use_case": "Cuando buscas puestos muy específicos"
            },
            "combined": {
                "name": "Búsqueda Combinada", 
                "description": "Busca en títulos y descripciones completas",
                "use_case": "Para búsquedas más amplias y exploratorias"
            },
            "hybrid": {
                "name": "Búsqueda Híbrida",
                "description": "Combinación ponderada de título (30%) y descripción (70%)",
                "use_case": "Recomendado para uso general, balance entre precisión y diversidad"
            }
        },
        "recommended": "hybrid"
    }

@app.get("/api/model/jobs/{job_id}")
async def get_job_details(job_id: int):
    """Obtener detalles de un empleo específico"""
    if not recommendation_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    job = recommendation_service.df[recommendation_service.df['id'] == job_id]
    if job.empty:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = job.iloc[0]
    return {
        "id": int(job_data['id']),
        "title": job_data['title'],
        "company": job_data['company'],
        "location": job_data.get('location', ''),
        "description": job_data.get('description', ''),
        "work_type": job_data.get('work_type', ''),
        "employment_type": job_data.get('employment_type', '')
    }

@app.get("/api/model/jobs")
async def get_all_jobs(skip: int = 0, limit: int = 100):
    """Obtener lista de todos los empleos con paginación"""
    if not recommendation_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    total_jobs = len(recommendation_service.df)
    jobs_subset = recommendation_service.df.iloc[skip:skip + limit]
    
    jobs_list = []
    for _, job in jobs_subset.iterrows():
        jobs_list.append({
            "id": int(job['id']),
            "title": job['title'],
            "company": job['company'],
            "location": job.get('location', ''),
            "description": job.get('description', '')[:200] + "..." if len(str(job.get('description', ''))) > 200 else job.get('description', ''),
            "work_type": job.get('work_type', ''),
            "employment_type": job.get('employment_type', '')
        })
    
    return {
        "total": total_jobs,
        "skip": skip,
        "limit": limit,
        "jobs": jobs_list
    }

# =================== CONTROLADORES MÉTODO SIMPLIFICADO ===================

@app.get("/api/model/predict")
def predict(title: str = Query(..., description="Título de empleo a buscar")):
    """Endpoint compatible con método simplificado - Solo para modelo completo"""
    if not recommendation_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if recommendation_service.model_type != 'complete':
        raise HTTPException(status_code=400, detail="Este endpoint requiere el modelo completo (job_recommendation_system.pkl)")
    
    try:
        # Preprocesa el título
        processed_title = recommendation_service.preprocessor.transform([title])
        # Obtiene recomendaciones
        results = recommendation_service.model.recommend(processed_title)
        return {"results": results}
    except Exception as e:
        logger.error(f"Error en /predict: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/model/predict/batch")
def predict_batch(request: BatchPredictionRequest):
    """Predicción en lote para múltiples títulos"""
    if not recommendation_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if recommendation_service.model_type != 'complete':
        raise HTTPException(status_code=400, detail="Este endpoint requiere el modelo completo")
    
    try:
        results = []
        for title in request.titles:
            processed_title = recommendation_service.preprocessor.transform([title])
            recommendations = recommendation_service.model.recommend(processed_title)
            results.append({
                "query": title,
                "recommendations": recommendations
            })
        
        return BatchPredictionResponse(
            success=True,
            total_queries=len(request.titles),
            results=results
        )
    except Exception as e:
        logger.error(f"Error en /predict/batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model/predict/similar/{job_id}")
def get_similar_jobs(job_id: int, top_n: int = Query(default=10, ge=1, le=50)):
    """Encontrar empleos similares a un empleo específico"""
    if not recommendation_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Buscar el empleo específico
        job = recommendation_service.df[recommendation_service.df['id'] == job_id]
        if job.empty:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job_data = job.iloc[0]
        job_title = job_data['title']
        
        if recommendation_service.model_type == 'complete':
            # Usar modelo completo
            processed_title = recommendation_service.preprocessor.transform([job_title])
            results = recommendation_service.model.recommend(processed_title, top_n=top_n + 1)  # +1 para excluir el mismo trabajo
            
            # Filtrar el trabajo original
            filtered_results = [r for r in results if r.get('id', 0) != job_id][:top_n]
            
            return {
                "original_job": {
                    "id": int(job_data['id']),
                    "title": job_data['title'],
                    "company": job_data['company']
                },
                "similar_jobs": filtered_results
            }
        else:
            # Fallback usando método básico
            cleaned_query = recommendation_service.clean_text_basic(job_title)
            query_vector = recommendation_service.tfidf_vectorizer.transform([cleaned_query])
            similarity_scores = cosine_similarity(query_vector, recommendation_service.tfidf_matrix).flatten()
            
            # Obtener top N+1 para excluir el original
            top_indices = similarity_scores.argsort()[-(top_n + 5):][::-1]
            
            similar_jobs = []
            for idx in top_indices:
                if int(recommendation_service.df.iloc[idx]['id']) != job_id and len(similar_jobs) < top_n:
                    similar_jobs.append({
                        "id": int(recommendation_service.df.iloc[idx]['id']),
                        "title": recommendation_service.df.iloc[idx]['title'],
                        "company": recommendation_service.df.iloc[idx]['company'],
                        "similarity_score": float(similarity_scores[idx])
                    })
            
            return {
                "original_job": {
                    "id": int(job_data['id']),
                    "title": job_data['title'],
                    "company": job_data['company']
                },
                "similar_jobs": similar_jobs
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en /predict/similar/{job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model/info")
async def get_model_info():
    """Información detallada sobre el modelo cargado"""
    if not recommendation_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = {
        "model_type": recommendation_service.model_type,
        "total_jobs": len(recommendation_service.df),
        "is_loaded": recommendation_service.is_loaded,
        "available_endpoints": [],
        "model_capabilities": {}
    }
    
    # Endpoints disponibles según el tipo de modelo
    if recommendation_service.model_type == 'complete':
        info["available_endpoints"] = [
            "/api/model/predict", "/api/model/predict/batch", "/api/model/predict/similar/{job_id}",
            "/api/model/recommend", "/api/model/health", "/api/model/jobs", "/api/model/jobs/{job_id}", "/api/model/search-methods"
        ]
        info["model_capabilities"] = {
            "simple_prediction": True,
            "batch_prediction": True,
            "similarity_search": True,
            "advanced_search": True,
            "hybrid_methods": False
        }
    elif recommendation_service.model_type == 'components':
        info["available_endpoints"] = [
            "/api/model/recommend", "/api/model/health", "/api/model/jobs", "/api/model/jobs/{job_id}", "/api/model/search-methods", "/api/model/predict/similar/{job_id}"
        ]
        info["model_capabilities"] = {
            "simple_prediction": False,
            "batch_prediction": False,
            "similarity_search": True,
            "advanced_search": True,
            "hybrid_methods": True
        }
    else:
        info["available_endpoints"] = [
            "/api/model/recommend", "/api/model/health", "/api/model/jobs", "/api/model/jobs/{job_id}", "/api/model/predict/similar/{job_id}"
        ]
        info["model_capabilities"] = {
            "simple_prediction": False,
            "batch_prediction": False,
            "similarity_search": True,
            "advanced_search": False,
            "hybrid_methods": False
        }
    
    return info

@app.get("/api/model/predict/company/{company_name}")
async def get_jobs_by_company(company_name: str, limit: int = Query(default=20, ge=1, le=100)):
    """Obtener empleos de una empresa específica"""
    if not recommendation_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Filtrar empleos por empresa (búsqueda parcial, case-insensitive)
        company_jobs = recommendation_service.df[
            recommendation_service.df['company'].str.contains(company_name, case=False, na=False)
        ].head(limit)
        
        if company_jobs.empty:
            return {
                "company": company_name,
                "total_jobs": 0,
                "jobs": []
            }
        
        jobs_list = []
        for _, job in company_jobs.iterrows():
            jobs_list.append({
                "id": int(job['id']),
                "title": job['title'],
                "company": job['company'],
                "location": job.get('location', ''),
                "description": job.get('description', '')[:200] + "..." if len(str(job.get('description', ''))) > 200 else job.get('description', ''),
                "work_type": job.get('work_type', ''),
                "employment_type": job.get('employment_type', '')
            })
        
        return {
            "company": company_name,
            "total_jobs": len(company_jobs),
            "jobs": jobs_list
        }
        
    except Exception as e:
        logger.error(f"Error en /predict/company/{company_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model/stats")
async def get_statistics():
    """Estadísticas generales del dataset de empleos"""
    if not recommendation_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        df = recommendation_service.df
        
        stats = {
            "total_jobs": len(df),
            "unique_companies": df['company'].nunique(),
            "unique_locations": df['location'].nunique() if 'location' in df.columns else 0,
            "model_type": recommendation_service.model_type
        }
        
        # Top empresas
        if 'company' in df.columns:
            top_companies = df['company'].value_counts().head(10).to_dict()
            stats["top_companies"] = top_companies
        
        # Top ubicaciones
        if 'location' in df.columns:
            top_locations = df['location'].value_counts().head(10).to_dict()
            stats["top_locations"] = top_locations
        
        # Tipos de trabajo
        if 'work_type' in df.columns:
            work_types = df['work_type'].value_counts().to_dict()
            stats["work_types"] = work_types
        
        # Tipos de empleo
        if 'employment_type' in df.columns:
            employment_types = df['employment_type'].value_counts().to_dict()
            stats["employment_types"] = employment_types
        
        return stats
        
    except Exception as e:
        logger.error(f"Error en /stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=8080, reload=True)
