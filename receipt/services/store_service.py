from typing import Optional
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from models.store_model import Store

logger = structlog.get_logger()


class StoreService:
    """Service for managing store records"""

    @staticmethod
    async def find_or_create_store(
        db: AsyncSession, store_name: str, store_address: Optional[str] = None
    ) -> Store:
        """Find existing store or create new one"""
        try:
            # Normalize store name for consistent matching
            normalized_name = StoreService._normalize_store_name(store_name)

            # First, try to find existing store by name
            query = select(Store).where(Store.name.ilike(f"%{normalized_name}%"))
            result = await db.execute(query)
            existing_store = result.scalars().first()

            if existing_store:
                logger.info(
                    "Found existing store",
                    store_id=existing_store.id,
                    store_name=existing_store.name,
                )

                # Update address if it's more complete
                if store_address and not existing_store.address:
                    existing_store.address = store_address
                    await db.commit()
                    logger.info("Updated store address", store_id=existing_store.id)

                return existing_store

            # Create new store
            new_store = Store(
                name=store_name,
                address=store_address,
                category=StoreService._categorize_store(store_name),
            )

            db.add(new_store)
            await db.flush()  # Get the ID

            logger.info(
                "Created new store",
                store_id=new_store.id,
                store_name=new_store.name,
                category=new_store.category,
            )

            return new_store

        except IntegrityError as e:
            await db.rollback()
            logger.error(
                "Store creation failed due to integrity constraint", error=str(e)
            )
            # Try to find the store that might have been created by another process
            query = select(Store).where(Store.name == store_name)
            result = await db.execute(query)
            return result.scalars().first()

        except Exception as e:
            await db.rollback()
            logger.error(
                "Failed to create or find store", error=str(e), store_name=store_name
            )
            raise

    @staticmethod
    def _normalize_store_name(store_name: str) -> str:
        """Normalize store name for consistent matching"""
        if not store_name:
            return ""

        # Remove common business suffixes for matching
        normalized = store_name.lower().strip()
        suffixes_to_remove = [
            "sdn bhd",
            "sdn.bhd",
            "ltd",
            "limited",
            "inc",
            "incorporated",
            "corp",
            "corporation",
            "company",
            "co.",
            "pte ltd",
        ]

        for suffix in suffixes_to_remove:
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)].strip()
                break

        # Remove extra whitespace and special characters
        normalized = " ".join(normalized.split())
        return normalized

    @staticmethod
    def _categorize_store(store_name: str) -> Optional[str]:
        """Categorize store based on name"""
        if not store_name:
            return "Other"

        name_lower = store_name.lower()

        categories = {
            "Stationery": ["stationery", "book", "office", "supplies"],
            "Grocery": ["supermarket", "market", "grocery", "mart", "hypermarket"],
            "Restaurant": ["restaurant", "cafe", "coffee", "food", "kitchen", "dining"],
            "Pharmacy": ["pharmacy", "medical", "health", "clinic"],
            "Department Store": ["department", "shopping", "mall", "center"],
            "Convenience Store": ["convenience", "7-eleven", "mini mart", "express"],
            "Electronics": ["electronic", "tech", "computer", "mobile"],
            "Clothing": ["fashion", "apparel", "clothing", "boutique"],
        }

        for category, keywords in categories.items():
            if any(keyword in name_lower for keyword in keywords):
                return category

        return "Other"

    @staticmethod
    async def get_store_statistics(db: AsyncSession) -> dict:
        """Get statistics about stores in the database"""
        try:
            query = select(Store)
            result = await db.execute(query)
            stores = result.scalars().all()

            total_stores = len(stores)
            categories = {}

            for store in stores:
                category = store.category or "Other"
                categories[category] = categories.get(category, 0) + 1

            return {
                "total_stores": total_stores,
                "categories": categories,
                "stores": [
                    {
                        "id": store.id,
                        "name": store.name,
                        "category": store.category,
                        "has_address": bool(store.address),
                    }
                    for store in stores
                ],
            }

        except Exception as e:
            logger.error("Failed to get store statistics", error=str(e))
            return {"total_stores": 0, "categories": {}, "stores": []}
