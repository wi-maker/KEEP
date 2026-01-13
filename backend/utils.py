"""Utility functions for KEEP backend
- Text extraction (PDF, images)
- Text chunking
- File handling
"""

import os
from typing import List
import PyPDF2
from PIL import Image, ImageEnhance
import pytesseract

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file with OCR fallback for image-based PDFs"""
    text = ""
    try:
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                
                # If text extraction yields little or no text, try OCR
                if page_text and len(page_text.strip()) > 50:
                    text += page_text + "\n\n"
                else:
                    # Try OCR on this page (image-based PDF)
                    try:
                        # Convert PDF page to image and OCR it
                        import pdf2image
                        images = pdf2image.convert_from_path(
                            file_path,
                            first_page=page_num + 1,
                            last_page=page_num + 1
                        )
                        if images:
                            ocr_text = pytesseract.image_to_string(images[0])
                            if ocr_text.strip():
                                text += ocr_text + "\n\n"
                            elif page_text:  # Use whatever text we got
                                text += page_text + "\n\n"
                    except ImportError:
                        # pdf2image not installed, use whatever text we got
                        if page_text:
                            text += page_text + "\n\n"
                    except Exception as ocr_e:
                        # OCR failed, use whatever text extraction gave us
                        if page_text:
                            text += page_text + "\n\n"
                        
    except Exception as e:
        raise ValueError(f"Error extracting text from PDF: {str(e)}")
    
    return text.strip()

def extract_text_from_image(file_path: str) -> str:
    """Extract text from image using OCR with enhanced configuration"""
    try:
        # Try to auto-detect Tesseract on Windows if not in PATH
        if os.name == 'nt':  # Windows
            # Common Tesseract installation paths on Windows
            possible_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME', '')),
            ]
            
            # Check if pytesseract can find Tesseract
            try:
                pytesseract.get_tesseract_version()
            except:
                # Try to find Tesseract manually
                for path in possible_paths:
                    if os.path.exists(path):
                        pytesseract.pytesseract.tesseract_cmd = path
                        break
        
        # Open and preprocess image for better OCR results
        image = Image.open(file_path)
        
        # Convert to RGB if needed (handles RGBA, grayscale, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Enhance image for better OCR (optional preprocessing)
        # Convert to grayscale for better text detection
        from PIL import ImageEnhance
        image = image.convert('L')  # Convert to grayscale
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        # Extract text with custom config for better accuracy
        # --oem 3: Use default OCR Engine Mode (LSTM + Legacy)
        # --psm 3: Fully automatic page segmentation (default)
        custom_config = r'--oem 3 --psm 3'
        text = pytesseract.image_to_string(image, config=custom_config)
        
        # Clean up extracted text
        text = text.strip()
        
        # If we got very little text, try again with different preprocessing
        if len(text) < 50:
            # Try without preprocessing
            original_image = Image.open(file_path)
            text_alt = pytesseract.image_to_string(original_image)
            if len(text_alt) > len(text):
                text = text_alt.strip()
        
        if not text:
            return "No text could be extracted from this image. The image may be blank, too low quality, or contain only graphics/photos without text."
        
        return text
        
    except Exception as e:
        error_msg = str(e).lower()
        
        # Provide helpful error messages
        if "tesseract" in error_msg and ("not found" in error_msg or "not installed" in error_msg):
            raise ValueError(
                "Tesseract OCR is not installed or not found in PATH. "
                "Please install Tesseract OCR:\n"
                "- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki\n"
                "- Mac: brew install tesseract\n"
                "- Linux: sudo apt-get install tesseract-ocr"
            )
        elif "permission" in error_msg or "access" in error_msg:
            raise ValueError(f"Permission denied accessing image file: {file_path}")
        elif "file" in error_msg or "image" in error_msg:
            raise ValueError(f"Could not open or read image file. The file may be corrupted or in an unsupported format.")
        else:
            raise ValueError(f"Error extracting text from image: {str(e)}")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks with safeguards."""
    if not text:
        return []
    max_text_length = 5_000_000  # limit to ~5MB characters
    if len(text) > max_text_length:
        text = text[:max_text_length]
    chunks: List[str] = []
    start = 0
    text_length = len(text)
    max_chunks = 1000
    while start < text_length and len(chunks) < max_chunks:
        end = start + chunk_size
        if end < text_length:
            for punct in [". ", ".\n", "! ", "?\n"]:
                last_punct = text[start:end].rfind(punct)
                if last_punct != -1:
                    end = start + last_punct + len(punct)
                    break
        else:
            # Ensure we capture the remaining text
            end = text_length
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # If we've reached the end, break the loop
        if end >= text_length:
            break
            
        # Move start position forward with overlap
        start = end - overlap
    return chunks

def get_file_extension(filename: str) -> str:
    """Get file extension from filename"""
    return os.path.splitext(filename)[1].lower().lstrip(".")

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    unsafe_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in unsafe_chars:
        filename = filename.replace(char, "_")
    return filename

def get_file_size(file_path: str) -> int:
    """Get file size in bytes"""
    return os.path.getsize(file_path)
