from sqlalchemy import Column, Integer, String, Text
from config.database import Base


class Store(Base):
    """Store model representing retail stores"""

    __tablename__ = "stores"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    address = Column(Text, nullable=True)
    phone = Column(String(50), nullable=True)
    category = Column(String(100), nullable=True, index=True)

    def __repr__(self) -> str:
        return f"<Store(id={self.id}, name='{self.name}')>"
