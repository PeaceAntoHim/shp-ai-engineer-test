from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class ReceiptItemBase(BaseModel):
    """Base schema for receipt items"""
    item_name: str = Field(..., min_length=1, max_length=255)
    category: Optional[str] = Field(None, max_length=100)
    quantity: float = Field(..., gt=0)
    unit_price: float = Field(..., ge=0)
    total_price: float = Field(..., ge=0)
    description: Optional[str] = None


class ReceiptItemCreate(ReceiptItemBase):
    """Schema for creating receipt items"""
    pass


class ReceiptItem(ReceiptItemBase):
    """Schema for receipt item responses"""
    id: int
    receipt_id: int
    
    class Config:
        orm_mode = True


class ReceiptBase(BaseModel):
    """Base schema for receipts"""
    store_name: str = Field(..., min_length=1, max_length=255)
    store_address: Optional[str] = None
    receipt_date: datetime
    total_amount: float = Field(..., ge=0)
    tax_amount: Optional[float] = Field(0.0, ge=0)
    discount_amount: Optional[float] = Field(0.0, ge=0)


class ReceiptCreate(ReceiptBase):
    """Schema for creating receipts"""
    original_filename: str
    file_path: str
    file_size: int
    raw_text: str
    items: List[ReceiptItemCreate] = []


class Receipt(ReceiptBase):
    """Schema for receipt responses"""
    id: int
    store_id: Optional[int]
    original_filename: str
    file_path: str
    file_size: int
    raw_text: str
    processed_at: datetime
    is_processed: bool
    items: List[ReceiptItem] = []
    
    class Config:
        orm_mode = True


class ReceiptUploadResponse(BaseModel):
    """Schema for receipt upload responses"""
    message: str
    receipt_id: int  # <- This field is expected
    total_amount: float
    items_count: int
    store_name: str
