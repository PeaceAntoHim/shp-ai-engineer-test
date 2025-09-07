import os
import uuid
from typing import List

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from config.settings import get_settings
from config.database import get_db_session
from models.receipt_model import Receipt
from models.receipt_item_model import ReceiptItem
from models.store_model import Store
from schemas.receipt_schema import Receipt as ReceiptSchema, ReceiptUploadResponse
from services.ocr_service import OCRService
from services.receipt_parser_service import ReceiptParser
from services.store_service import StoreService

router = APIRouter(prefix="/receipts", tags=["Receipts"])
logger = structlog.get_logger()
settings = get_settings()


@router.get("/image/{receipt_id}")
async def get_receipt_image(
    receipt_id: int,
    db: AsyncSession = Depends(get_db_session)
):
    """Get the receipt image file"""
    try:
        # Get receipt from database
        query = select(Receipt).where(Receipt.id == receipt_id)
        result = await db.execute(query)
        receipt = result.scalars().first()
        
        if not receipt:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Receipt not found"
            )
        
        # Check if file exists
        if not os.path.exists(receipt.file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Receipt image file not found"
            )
        
        # Return the image file
        return FileResponse(
            path=receipt.file_path,
            media_type="image/jpeg",
            filename=receipt.original_filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to serve receipt image", error=str(e), receipt_id=receipt_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load receipt image"
        )


@router.post("/upload", response_model=ReceiptUploadResponse)
async def upload_receipt(
        file: UploadFile = File(...),
        db: AsyncSession = Depends(get_db_session)
):
    """Upload and process a receipt image"""

    # Validate file
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only image files are allowed"
        )

    if file.size and file.size > settings.max_file_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum allowed size of {settings.max_file_size} bytes"
        )

    try:
        # Create uploads directory if it doesn't exist
        os.makedirs(settings.upload_directory, exist_ok=True)

        # Generate unique filename
        file_extension = file.filename.split('.')[-1].lower() if file.filename else 'jpg'
        unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
        file_path = os.path.join(settings.upload_directory, unique_filename)

        # Save uploaded file
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)

        logger.info("File uploaded successfully",
                    filename=file.filename,
                    saved_as=unique_filename,
                    size=len(content))

        # Extract text using OCR
        extracted_text = OCRService.extract_text_from_image(file_path)
        if not extracted_text:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Could not extract text from the image"
            )

        # Parse receipt data
        receipt_data = ReceiptParser.parse_receipt(
            extracted_text,
            file.filename or unique_filename,
            file_path,
            len(content)
        )

        # Find or create store record
        store = await StoreService.find_or_create_store(
            db=db,
            store_name=receipt_data.store_name,
            store_address=receipt_data.store_address
        )

        # Save to database
        db_receipt = Receipt(
            store_id=store.id,  # Now properly linked to Store table
            store_name=receipt_data.store_name,
            store_address=receipt_data.store_address,
            receipt_date=receipt_data.receipt_date,
            total_amount=receipt_data.total_amount,
            tax_amount=receipt_data.tax_amount,
            discount_amount=receipt_data.discount_amount,
            original_filename=receipt_data.original_filename,
            file_path=receipt_data.file_path,
            file_size=receipt_data.file_size,
            raw_text=receipt_data.raw_text,
            is_processed=True
        )

        db.add(db_receipt)
        await db.flush()  # Get the ID

        # Save receipt items
        for item_data in receipt_data.items:
            db_item = ReceiptItem(
                receipt_id=db_receipt.id,
                item_name=item_data.item_name,
                category=item_data.category,
                quantity=item_data.quantity,
                unit_price=item_data.unit_price,
                total_price=item_data.total_price,
                description=item_data.description
            )
            db.add(db_item)

        await db.commit()

        logger.info("Receipt processed and saved successfully",
                    receipt_id=db_receipt.id,
                    store_id=store.id,
                    store_name=store.name,
                    items_count=len(receipt_data.items))

        return ReceiptUploadResponse(
            receipt_id=db_receipt.id,
            message="Receipt uploaded and processed successfully",
            store_name=receipt_data.store_name,
            total_amount=receipt_data.total_amount,
            items_count=len(receipt_data.items)
        )

    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error("Receipt processing failed", error=str(e), filename=file.filename)
        # Clean up uploaded file on error
        try:
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
        except Exception as cleanup_error:
            logger.warning("Failed to cleanup uploaded file", error=str(cleanup_error))

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process receipt"
        )


@router.get("/", response_model=List[ReceiptSchema])
async def get_receipts(
        skip: int = 0,
        limit: int = 100,
        db: AsyncSession = Depends(get_db_session)
):
    """Get list of processed receipts"""
    try:
        # Simple query without eager loading for now to avoid issues
        query = (
            select(Receipt)
            .offset(skip)
            .limit(limit)
            .order_by(Receipt.receipt_date.desc())
        )
        result = await db.execute(query)
        receipts = result.scalars().all()

        # Manually convert to dict and ensure proper serialization
        receipt_list = []
        for receipt in receipts:
            try:
                # Load items separately to avoid lazy loading issues
                items_query = select(ReceiptItem).where(ReceiptItem.receipt_id == receipt.id)
                items_result = await db.execute(items_query)
                items = items_result.scalars().all()
                
                receipt_dict = {
                    "id": receipt.id,
                    "store_id": receipt.store_id,
                    "store_name": receipt.store_name,
                    "store_address": receipt.store_address,
                    "receipt_date": receipt.receipt_date,
                    "total_amount": float(receipt.total_amount),
                    "tax_amount": float(receipt.tax_amount) if receipt.tax_amount else 0.0,
                    "discount_amount": float(receipt.discount_amount) if receipt.discount_amount else 0.0,
                    "original_filename": receipt.original_filename,
                    "file_path": receipt.file_path,
                    "file_size": receipt.file_size,
                    "raw_text": receipt.raw_text,
                    "processed_at": receipt.processed_at,
                    "is_processed": receipt.is_processed,
                    "items": [
                        {
                            "id": item.id,
                            "receipt_id": item.receipt_id,
                            "item_name": item.item_name,
                            "category": item.category,
                            "quantity": float(item.quantity),
                            "unit_price": float(item.unit_price),
                            "total_price": float(item.total_price),
                            "description": item.description
                        }
                        for item in items
                    ]
                }
                receipt_list.append(receipt_dict)
            except Exception as item_error:
                logger.error("Error processing individual receipt", 
                           receipt_id=receipt.id, error=str(item_error))
                continue

        return receipt_list

    except Exception as e:
        logger.error("Failed to fetch receipts", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch receipts"
        )


@router.get("/{receipt_id}", response_model=ReceiptSchema)
async def get_receipt(
        receipt_id: int,
        db: AsyncSession = Depends(get_db_session)
):
    """Get specific receipt by ID"""
    try:
        # Get receipt
        query = select(Receipt).where(Receipt.id == receipt_id)
        result = await db.execute(query)
        receipt = result.scalars().first()

        if not receipt:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Receipt not found"
            )

        # Get items separately
        items_query = select(ReceiptItem).where(ReceiptItem.receipt_id == receipt.id)
        items_result = await db.execute(items_query)
        items = items_result.scalars().all()

        # Return as dict to ensure proper serialization
        return {
            "id": receipt.id,
            "store_id": receipt.store_id,
            "store_name": receipt.store_name,
            "store_address": receipt.store_address,
            "receipt_date": receipt.receipt_date,
            "total_amount": float(receipt.total_amount),
            "tax_amount": float(receipt.tax_amount) if receipt.tax_amount else 0.0,
            "discount_amount": float(receipt.discount_amount) if receipt.discount_amount else 0.0,
            "original_filename": receipt.original_filename,
            "file_path": receipt.file_path,
            "file_size": receipt.file_size,
            "raw_text": receipt.raw_text,
            "processed_at": receipt.processed_at,
            "is_processed": receipt.is_processed,
            "items": [
                {
                    "id": item.id,
                    "receipt_id": item.receipt_id,
                    "item_name": item.item_name,
                    "category": item.category,
                    "quantity": float(item.quantity),
                    "unit_price": float(item.unit_price),
                    "total_price": float(item.total_price),
                    "description": item.description
                }
                for item in items
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to fetch receipt", error=str(e), receipt_id=receipt_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch receipt"
        )
