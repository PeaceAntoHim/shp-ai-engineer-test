import cv2
import numpy as np
import pytesseract
from PIL import Image
from typing import Optional
import structlog

logger = structlog.get_logger()


class OCRService:
    """Service for extracting text from receipt images using OCR"""
    
    @staticmethod
    def preprocess_image(image_path: str) -> Optional[np.ndarray]:
        """Preprocess image for better OCR accuracy"""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                logger.error("Failed to load image", path=image_path)
                return None
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Resize image for better OCR (receipts are often small)
            height, width = gray.shape
            if height < 800:
                scale_factor = 800 / height
                new_width = int(width * scale_factor)
                gray = cv2.resize(gray, (new_width, 800), interpolation=cv2.INTER_CUBIC)
            
            # Apply bilateral filter to reduce noise while preserving edges
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Apply adaptive threshold for better text extraction
            thresh = cv2.adaptiveThreshold(
                filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Apply morphological operations to improve text clarity
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Apply dilation to make text bolder
            kernel_dilate = np.ones((1, 1), np.uint8)
            processed = cv2.dilate(processed, kernel_dilate, iterations=1)
            
            return processed
            
        except Exception as e:
            logger.error("Image preprocessing failed", error=str(e), path=image_path)
            return None
    
    @staticmethod
    def extract_text_from_image(image_path: str) -> Optional[str]:
        """Extract text from receipt image using Tesseract OCR"""
        try:
            # Try multiple approaches for better accuracy
            texts = []
            
            # Approach 1: Preprocessed image with custom config
            processed_img = OCRService.preprocess_image(image_path)
            if processed_img is not None:
                pil_image = Image.fromarray(processed_img)
                custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,$/:()\-@& '
                text1 = pytesseract.image_to_string(pil_image, config=custom_config)
                texts.append(text1)
            
            # Approach 2: Original image with different PSM
            original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if original_img is not None:
                pil_original = Image.fromarray(original_img)
                config2 = r'--oem 3 --psm 4'
                text2 = pytesseract.image_to_string(pil_original, config=config2)
                texts.append(text2)
            
            # Approach 3: Different preprocessing
            enhanced_img = OCRService.enhance_image_alternative(image_path)
            if enhanced_img is not None:
                pil_enhanced = Image.fromarray(enhanced_img)
                config3 = r'--oem 3 --psm 8'
                text3 = pytesseract.image_to_string(pil_enhanced, config=config3)
                texts.append(text3)
            
            # Choose the best result (longest meaningful text)
            best_text = max(texts, key=lambda t: len(t.strip())) if texts else ""
            
            # Clean and normalize text
            cleaned_text = OCRService.clean_extracted_text(best_text)
            
            logger.info("Text extraction successful", 
                       characters_extracted=len(cleaned_text),
                       approaches_tried=len(texts),
                       path=image_path)
            
            return cleaned_text
            
        except Exception as e:
            logger.error("OCR extraction failed", error=str(e), path=image_path)
            return None
    
    @staticmethod
    def enhance_image_alternative(image_path: str) -> Optional[np.ndarray]:
        """Alternative image enhancement approach"""
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            
            # Resize if too small
            height, width = img.shape
            if height < 800:
                scale_factor = 800 / height
                new_width = int(width * scale_factor)
                img = cv2.resize(img, (new_width, 800), interpolation=cv2.INTER_CUBIC)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(img)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            # Apply threshold
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            return thresh
            
        except Exception as e:
            logger.error("Alternative image enhancement failed", error=str(e))
            return None
    
    @staticmethod
    def clean_extracted_text(text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace and normalize line breaks
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Clean individual lines
        cleaned_lines = []
        for line in lines:
            # Remove extra spaces
            line = ' '.join(line.split())
            
            # Fix common OCR errors
            line = OCRService.fix_common_ocr_errors(line)
            
            if line:  # Only add non-empty lines
                cleaned_lines.append(line)
        
        cleaned_text = '\n'.join(cleaned_lines)
        
        return cleaned_text
    
    @staticmethod
    def fix_common_ocr_errors(text: str) -> str:
        """Fix common OCR misreading errors"""
        # Common character substitutions
        replacements = {
            # Letters often confused with numbers
            'O': '0',  # Only in numeric contexts
            'o': '0',  # Only in numeric contexts  
            'l': '1',  # Only in numeric contexts
            'I': '1',  # Only in numeric contexts
            'S': '5',  # Only in numeric contexts
            's': '5',  # Only in numeric contexts
            'B': '8',  # Only in numeric contexts
            
            # Common word corrections
            'STATIONERY': 'STATIONERY',
            'SDN': 'SDN',
            'BHD': 'BHD',
            'GST': 'GST',
            'TAX': 'TAX',
            'TOTAL': 'TOTAL',
            'AMOUNT': 'AMOUNT',
            'RECEIPT': 'RECEIPT',
        }
        
        # Apply intelligent replacements
        result = text
        
        # Fix numeric patterns (prices, dates, etc.)
        import re
        
        # Fix price patterns - replace letters that look like numbers in price contexts
        price_pattern = r'\$?[\dOolIS]+\.[\dOolIS]+'
        def fix_price(match):
            price = match.group(0)
            fixed = price.replace('O', '0').replace('o', '0').replace('l', '1').replace('I', '1').replace('S', '5').replace('s', '5')
            return fixed
        
        result = re.sub(price_pattern, fix_price, result)
        
        # Fix decimal number patterns
        decimal_pattern = r'\b[\dOolIS]+\.[\dOolIS]+\b'
        def fix_decimal(match):
            num = match.group(0)
            fixed = num.replace('O', '0').replace('o', '0').replace('l', '1').replace('I', '1').replace('S', '5').replace('s', '5')
            return fixed
        
        result = re.sub(decimal_pattern, fix_decimal, result)
        
        return result
