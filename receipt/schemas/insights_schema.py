from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class AIMode(str, Enum):
    """AI processing modes"""
    GENERATIVE = "generative"
    RULE_BASED = "rule_based"


class InsightQuery(BaseModel):
    """Schema for insight query requests"""
    question: str = Field(..., min_length=1, max_length=1000)
    ai_mode: Optional[AIMode] = Field(default=None, description="AI processing mode preference")


class InsightResponse(BaseModel):
    """Schema for AI insight responses"""
    query: str
    answer: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    ai_mode_used: AIMode = Field(description="AI mode that was actually used")
    relevant_data: List[Dict[str, Any]] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SpendingAnalytics(BaseModel):
    """Schema for spending analytics responses"""
    total_spent: float = Field(default=0.0, ge=0)
    transaction_count: int = Field(default=0, ge=0)
    average_transaction: float = Field(default=0.0, ge=0)
    top_categories: List[Dict[str, Any]] = Field(default_factory=list)
    daily_spending: List[Dict[str, Any]] = Field(default_factory=list)
    period_days: int = Field(default=30, gt=0)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }