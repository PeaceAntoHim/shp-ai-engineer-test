from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from config.database import get_db_session
from schemas.insights_schema import InsightQuery, InsightResponse, SpendingAnalytics, AIMode
from services.ai_insights_service import AIService

router = APIRouter(prefix="/insights", tags=["Insights"])
logger = structlog.get_logger()

# Initialize AI service
ai_service = AIService()


@router.get("/ai-availability")
async def get_ai_availability():
    """Get available AI modes"""
    return {
        "generative_available": ai_service.client is not None,
        "rule_based_available": True,
        "default_mode": "generative" if ai_service.client is not None else "rule_based"
    }


@router.post("/query", response_model=InsightResponse)
async def ask_question(
    query: InsightQuery,
    db: AsyncSession = Depends(get_db_session)
):
    """Ask a natural language question about your food purchases"""
    try:
        logger.info("Processing insight query", question=query.question, ai_mode=query.ai_mode)

        response = await ai_service.process_insight_query(
            question=query.question,
            db=db,
            preferred_mode=query.ai_mode
        )

        return response

    except Exception as e:
        logger.error("Failed to process insight query", error=str(e), question=query.question)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process your question"
        )


@router.get("/analytics", response_model=SpendingAnalytics)
async def get_spending_analytics(
    days: int = 30,
    db: AsyncSession = Depends(get_db_session)
):
    """Get spending analytics for the specified period"""
    try:
        if days <= 0 or days > 365:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Days must be between 1 and 365"
            )

        analytics = await ai_service.get_spending_analytics(db, days)

        return analytics

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to generate analytics", error=str(e), days=days)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate spending analytics"
        )