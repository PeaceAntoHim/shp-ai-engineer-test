from sqlalchemy import Column, Integer, String, Float, ForeignKey, Text
from sqlalchemy.orm import relationship

from config.database import Base


class ReceiptItem(Base):
    """Receipt item model representing individual items on a receipt"""
    
    __tablename__ = "receipt_items"
    
    id = Column(Integer, primary_key=True, index=True)
    receipt_id = Column(Integer, ForeignKey("receipts.id"), nullable=False)
    
    # Item details
    item_name = Column(String(255), nullable=False, index=True)
    category = Column(String(100), nullable=True, index=True)
    quantity = Column(Float, nullable=False, default=1.0)
    unit_price = Column(Float, nullable=False)
    total_price = Column(Float, nullable=False)
    
    # Additional information
    description = Column(Text, nullable=True)
    
    # Relationships
    receipt = relationship("Receipt", back_populates="items")
    
    def __repr__(self) -> str:
        return f"<ReceiptItem(id={self.id}, name='{self.item_name}', price={self.total_price})>"
