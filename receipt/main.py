from contextlib import asynccontextmanager
import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from routes import receipts, insights, health
from config.settings import get_settings
from config.database import init_database
from config.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    setup_logging()
    logger = structlog.get_logger()
    logger.info("Starting Receipt Management Platform")

    try:
        await init_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize database", error=str(e))
        raise

    yield

    # Shutdown
    logger.info("Shutting down Receipt Management Platform")


def create_application() -> FastAPI:
    """Application factory"""
    settings = get_settings()

    app = FastAPI(
        title="Food Receipt Management Platform",
        description="AI-powered platform for receipt processing and insights",
        version="1.0.0",
        lifespan=lifespan
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static files
    app.mount("/static", StaticFiles(directory="static"), name="static")

    # Include routers
    app.include_router(health.router, prefix="/api/v1")
    app.include_router(receipts.router, prefix="/api/v1")
    app.include_router(insights.router, prefix="/api/v1")

    return app


app = create_application()

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_config=None  # We use structlog
    )