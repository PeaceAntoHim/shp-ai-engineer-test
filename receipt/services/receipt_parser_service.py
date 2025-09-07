import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import structlog

from schemas.receipt_schema import ReceiptCreate, ReceiptItemCreate

logger = structlog.get_logger()


class ReceiptParser:
    """Service for parsing receipt text into structured data"""

    # Enhanced patterns for better receipt parsing
    PRICE_PATTERN = r'\$?\s*(\d+\.?\d*)\s*'
    ITEM_PRICE_PATTERN = r'(\d+\.\d{2})'  # More specific for item prices
    DATE_PATTERNS = [
        r'(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})',
        r'(\d{4}[-/.]\d{1,2}[-/.]\d{1,2})',
        r'(\d{1,2}\s+\d{1,2}\s+\d{2,4})',
    ]

    TOTAL_KEYWORDS = ['total amt', 'total amount', 'total', 'grand total', 'amount due', 'balance due',
                      'total amt payable']
    TAX_KEYWORDS = ['tax', 'hst', 'gst', 'pst', 'sales tax', 'sr @', 'sr@']
    STORE_IDENTIFIER_KEYWORDS = ['sdn bhd', 'sdn.bhd', 'ltd', 'inc', 'corp', 'company', 'store', 'mart']

    # Items that should be ignored (not actual products)
    IGNORE_PATTERNS = [
        r'gst\s*id',
        r'tax\s*invoice',
        r'receipt',
        r'thank\s*you',
        r'total\s*amt',
        r'rounding\s*adjustment',
        r'paid\s*amount',
        r'change',
        r'total\s*qty',
        r'cashier',
        r'transaction',
        r'\d+\s*seri\s*kembangan',  # Address parts
        r'selangor',
        r'darul\s*ehsan',
        r'jalan\s*sr',
        r'seksyen',
        r'taman\s*serdang',
        r'no\.\s*\d+',  # Street numbers
    ]

    @classmethod
    def parse_receipt(cls, raw_text: str, filename: str, file_path: str, file_size: int) -> ReceiptCreate:
        """Parse raw OCR text into structured receipt data with improved accuracy"""
        logger.info("Starting enhanced receipt parsing", filename=filename)

        try:
            # Clean and normalize the text first
            cleaned_text = cls._clean_raw_text(raw_text)

            # Extract basic information
            store_info = cls._extract_store_info(cleaned_text)
            receipt_date = cls._extract_date(cleaned_text)
            amounts = cls._extract_amounts(cleaned_text)
            items = cls._extract_items(cleaned_text)

            receipt_data = ReceiptCreate(
                store_name=store_info.get('name', 'Unknown Store'),
                store_address=store_info.get('address'),
                receipt_date=receipt_date,
                total_amount=amounts.get('total', 0.0),
                tax_amount=amounts.get('tax', 0.0),
                discount_amount=amounts.get('discount', 0.0),
                original_filename=filename,
                file_path=file_path,
                file_size=file_size,
                raw_text=raw_text,
                items=items
            )

            logger.info("Enhanced receipt parsing completed",
                        store_name=receipt_data.store_name,
                        total_amount=receipt_data.total_amount,
                        items_count=len(items))

            return receipt_data

        except Exception as e:
            logger.error("Receipt parsing failed", error=str(e), filename=filename)
            # Return minimal valid data
            return ReceiptCreate(
                store_name="Unknown Store",
                receipt_date=datetime.now(),
                total_amount=0.0,
                original_filename=filename,
                file_path=file_path,
                file_size=file_size,
                raw_text=raw_text,
                items=[]
            )

    @classmethod
    def _clean_raw_text(cls, text: str) -> str:
        """Clean and normalize raw OCR text"""
        if not text:
            return ""

        # Split into lines and clean each line
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if line:
                # Remove excessive spaces
                line = ' '.join(line.split())
                lines.append(line)

        return '\n'.join(lines)

    @classmethod
    def _extract_store_info(cls, text: str) -> Dict[str, Optional[str]]:
        """Extract store name and address with improved accuracy"""
        lines = text.split('\n')

        store_name = None
        address_lines = []
        found_store_line = False

        for i, line in enumerate(lines[:8]):  # Check first 8 lines
            line = line.strip()
            if not line or len(line) < 3:
                continue

            line_lower = line.lower()

            # Skip receipt numbers/IDs at the top
            if re.match(r'^\d+$', line):
                continue

            # Look for store identifier keywords
            if any(keyword in line_lower for keyword in cls.STORE_IDENTIFIER_KEYWORDS):
                # This line likely contains the store name
                store_name = line
                found_store_line = True
                continue

            # If we haven't found a store name yet and this looks like a business name
            if not store_name and not found_store_line:
                # Check if it looks like a business name (has multiple words, not all numbers)
                if len(line.split()) >= 2 and not re.match(r'^[\d\s\-\.]+$', line):
                    # Skip obvious non-store lines
                    if not any(skip in line_lower for skip in ['receipt', 'invoice', 'gst id', 'tax invoice']):
                        store_name = line
                        found_store_line = True
                        continue

            # Collect address lines after store name is found
            if found_store_line and i < 7:
                # Look for address patterns
                if (re.search(r'\d+.*\w+', line) or
                        any(word in line_lower for word in
                            ['jalan', 'street', 'st', 'ave', 'road', 'rd', 'taman', 'seksyen']) or
                        re.search(r'\d{5}', line)):  # Postal code pattern
                    address_lines.append(line)

        return {
            'name': store_name or 'Unknown Store',
            'address': ', '.join(address_lines) if address_lines else None
        }

    @classmethod
    def _extract_date(cls, text: str) -> datetime:
        """Extract receipt date with improved pattern matching"""
        # Look for date patterns
        for pattern in cls.DATE_PATTERNS:
            matches = re.findall(pattern, text)
            if matches:
                for date_str in matches:
                    try:
                        # Clean the date string
                        date_str = re.sub(r'[^\d\-/.]', ' ', date_str).strip()
                        date_str = re.sub(r'\s+', '-', date_str)

                        # Try different date formats
                        date_formats = [
                            '%d-%m-%Y', '%m-%d-%Y', '%Y-%m-%d',
                            '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d',
                            '%d.%m.%Y', '%m.%d.%Y', '%Y.%m.%d',
                            '%d-%m-%y', '%m-%d-%y', '%y-%m-%d',
                            '%d/%m/%y', '%m/%d/%y', '%y/%m/%d'
                        ]

                        for fmt in date_formats:
                            try:
                                return datetime.strptime(date_str, fmt)
                            except ValueError:
                                continue
                    except Exception:
                        continue

        # Default to current date if no date found
        logger.warning("Could not extract date from receipt, using current date")
        return datetime.now()

    @classmethod
    def _extract_amounts(cls, text: str) -> Dict[str, float]:
        """Extract total, tax, and discount amounts with improved accuracy"""
        amounts = {'total': 0.0, 'tax': 0.0, 'discount': 0.0}
        lines = text.lower().split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Extract total amount - look for specific patterns
            for keyword in cls.TOTAL_KEYWORDS:
                if keyword in line:
                    # Look for price pattern after the keyword
                    price_matches = re.findall(cls.ITEM_PRICE_PATTERN, line)
                    if price_matches:
                        try:
                            # Take the last (rightmost) price as the total
                            amounts['total'] = float(price_matches[-1])
                            logger.info(f"Found total amount: {amounts['total']} in line: {line}")
                        except ValueError:
                            pass
                        break

            # Extract tax amount
            for keyword in cls.TAX_KEYWORDS:
                if keyword in line and 'total' not in line:
                    price_matches = re.findall(cls.ITEM_PRICE_PATTERN, line)
                    if price_matches:
                        try:
                            amounts['tax'] = float(price_matches[-1])
                            logger.info(f"Found tax amount: {amounts['tax']} in line: {line}")
                        except ValueError:
                            pass
                        break

        return amounts

    @classmethod
    def _extract_items(cls, text: str) -> List[ReceiptItemCreate]:
        """Extract individual items with improved filtering"""
        items = []
        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            if not line or len(line) < 3:
                continue

            line_lower = line.lower()

            # Skip lines that are definitely not items
            if cls._should_ignore_line(line_lower):
                continue

            # Look for lines with prices that could be items
            price_matches = re.findall(cls.ITEM_PRICE_PATTERN, line)
            if price_matches and len(price_matches) <= 3:  # Reasonable number of prices per line
                item_info = cls._parse_item_line(line, price_matches)
                if item_info:
                    items.append(item_info)

        # Filter out items that are likely parsing errors
        items = cls._filter_valid_items(items)

        logger.info("Extracted items from receipt", count=len(items))
        return items

    @classmethod
    def _should_ignore_line(cls, line_lower: str) -> bool:
        """Check if a line should be ignored (not an item)"""
        # Check against ignore patterns
        for pattern in cls.IGNORE_PATTERNS:
            if re.search(pattern, line_lower):
                return True

        # Skip lines that are clearly not items
        ignore_keywords = [
            'total', 'subtotal', 'tax', 'change', 'cash', 'credit', 'debit',
            'receipt', 'thank you', 'cashier', 'transaction', 'invoice',
            'gst', 'amount', 'paid', 'rounding', 'balance'
        ]

        if any(keyword in line_lower for keyword in ignore_keywords):
            return True

        # Skip lines that are mostly numbers (likely not item names)
        if re.match(r'^[\d\s\-\.]+$', line_lower):
            return True

        return False

    @classmethod
    def _parse_item_line(cls, line: str, price_matches: List[str]) -> Optional[ReceiptItemCreate]:
        """Parse individual item line with better name extraction"""
        try:
            # Take the rightmost price as the total price
            total_price = float(price_matches[-1])

            # Remove all prices from line to get item name
            item_name = line
            for price in price_matches:
                # Remove price patterns more carefully
                item_name = re.sub(rf'\b{re.escape(price)}\b', '', item_name)
                item_name = re.sub(rf'\${re.escape(price)}', '', item_name)

            # Clean up the item name
            item_name = re.sub(r'\s+', ' ', item_name).strip()
            item_name = re.sub(r'[^\w\s\-]', ' ', item_name).strip()

            if not item_name or len(item_name) < 2:
                return None

            # Try to extract quantity
            quantity = 1.0
            qty_patterns = [
                r'(\d+)\s*x\s*',
                r'(\d+)\s*@\s*',
                r'(\d+)\s+qty',
                r'qty\s*(\d+)',
            ]

            for pattern in qty_patterns:
                qty_match = re.search(pattern, item_name.lower())
                if qty_match:
                    try:
                        quantity = float(qty_match.group(1))
                        item_name = re.sub(pattern, '', item_name, flags=re.IGNORECASE).strip()
                        break
                    except ValueError:
                        continue

            unit_price = total_price / quantity if quantity > 0 else total_price

            return ReceiptItemCreate(
                item_name=item_name[:255],  # Truncate to fit database
                quantity=quantity,
                unit_price=unit_price,
                total_price=total_price,
                category=cls._categorize_item(item_name)
            )

        except (ValueError, IndexError) as e:
            logger.debug(f"Failed to parse item line: {line}, error: {e}")
            return None

    @classmethod
    def _filter_valid_items(cls, items: List[ReceiptItemCreate]) -> List[ReceiptItemCreate]:
        """Filter out items that are likely parsing errors"""
        valid_items = []

        for item in items:
            # Skip items with unreasonably high prices (likely parsing errors)
            if item.total_price > 10000:  # Adjust threshold as needed
                logger.debug(f"Filtering out item with high price: {item.item_name} - ${item.total_price}")
                continue

            # Skip items with very short names that are likely parsing artifacts
            if len(item.item_name.strip()) < 3:
                continue

            # Skip items that are mostly numbers or symbols
            if re.match(r'^[\d\s\-\.\$]+$', item.item_name):
                continue

            valid_items.append(item)

        return valid_items

    @classmethod
    def _categorize_item(cls, item_name: str) -> Optional[str]:
        """Categorize items with expanded patterns"""
        if not item_name:
            return 'Other'

        item_lower = item_name.lower()

        categories = {
            'Stationery': ['pencil', 'pen', 'paper', 'notebook', 'eraser', 'ruler', 'stapler', 'faber', 'castell',
                           'pilot', 'stabilo'],
            'Food & Beverages': ['coffee', 'tea', 'soda', 'water', 'juice', 'beer', 'wine', 'drink', 'snack', 'chip',
                                 'candy'],
            'Dairy': ['milk', 'cheese', 'butter', 'yogurt', 'cream'],
            'Meat & Seafood': ['chicken', 'beef', 'pork', 'fish', 'turkey', 'ham', 'bacon', 'seafood'],
            'Produce': ['apple', 'banana', 'orange', 'lettuce', 'tomato', 'potato', 'onion', 'fruit', 'vegetable'],
            'Bakery': ['bread', 'bagel', 'muffin', 'cake', 'cookies', 'pastry'],
            'Personal Care': ['shampoo', 'soap', 'toothpaste', 'deodorant', 'lotion'],
            'Household': ['detergent', 'tissue', 'paper towel', 'cleaner', 'toilet paper'],
            'Electronics': ['battery', 'charger', 'cable', 'phone', 'computer'],
        }

        for category, keywords in categories.items():
            if any(keyword in item_lower for keyword in keywords):
                return category

        return 'Other'
