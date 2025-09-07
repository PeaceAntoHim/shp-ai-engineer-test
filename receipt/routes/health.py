from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from config.database import get_db_session

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check(db: AsyncSession = Depends(get_db_session)):
    """Health check endpoint"""
    try:
        # Test database connection
        await db.execute(text("SELECT 1"))
        
        return {
            "status": "healthy",
            "service": "Food Receipt Management Platform",
            "version": "1.0.0",
            "database": "connected"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "Food Receipt Management Platform",
            "version": "1.0.0",
            "database": "disconnected",
            "error": str(e)
        }
