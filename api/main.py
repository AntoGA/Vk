"""
FastAPI приложение для VK Interest Predictor.

Предоставляет REST API для:
- Предсказания интересов пользователей
- Получения информации о сегментах
- Управления моделью
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from pathlib import Path

from api.routes import predictions, segments, model_management
from api.dependencies import get_predictor_service, get_online_learner
from ml.inference.predictor import InterestPredictorService
from ml.training.online_learner import OnlineLearner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("Starting VK Interest Predictor API...")
    
    # Initialize predictor service
    try:
        predictor = get_predictor_service()
        app.state.predictor = predictor
        logger.info("Predictor service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
        raise
    
    # Initialize online learner (optional)
    try:
        learner = get_online_learner()
        app.state.learner = learner
        logger.info("Online learner initialized successfully")
    except Exception as e:
        logger.warning(f"Online learner not available: {e}")
        app.state.learner = None
    
    yield
    
    # Cleanup
    logger.info("Shutting down VK Interest Predictor API...")


app = FastAPI(
    title="VK Interest Predictor API",
    description="API для предсказания интересов пользователей ВКонтакте",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predictions.router, prefix="/api/v1", tags=["predictions"])
app.include_router(segments.router, prefix="/api/v1", tags=["segments"])
app.include_router(model_management.router, prefix="/api/v1", tags=["model"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "VK Interest Predictor API",
        "version": "0.1.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "predictor_loaded": app.state.predictor is not None,
        "online_learner_active": app.state.learner is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
