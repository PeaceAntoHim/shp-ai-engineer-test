from sqlalchemy import Column, Integer, String, DateTime, Float, Text, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from config.database import Base


class Receipt(Base):
    """Receipt model representing a processed receipt"""
    
    __tablename__ = "receipts"
    
    id = Column(Integer, primary_key=True, index=True)
    store_id = Column(Integer, index=True, nullable=True)
    store_name = Column(String(255), nullable=False)
    store_address = Column(Text, nullable=True)
    
    # Receipt details
    receipt_date = Column(DateTime, nullable=False)
    total_amount = Column(Float, nullable=False)
    tax_amount = Column(Float, nullable=True, default=0.0)
    discount_amount = Column(Float, nullable=True, default=0.0)
    
    # File information
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    
    # Processing information
    raw_text = Column(Text, nullable=False)
    processed_at = Column(DateTime, default=func.now(), nullable=False)
    is_processed = Column(Boolean, default=False, nullable=False)
    
    # Relationships
    items = relationship("ReceiptItem", back_populates="receipt", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<Receipt(id={self.id}, store='{self.store_name}', date={self.receipt_date}, total={self.total_amount})>"
