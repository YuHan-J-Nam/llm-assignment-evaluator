import os
import base64
import logging
from PyPDF2 import PdfReader
import io

def read_pdf_content(file_path):
    """Extract text content from a PDF file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    
    try:
        text_content = ""
        with open(file_path, 'rb') as file:
            pdf = PdfReader(file)
            for page in pdf.pages:
                text_content += page.extract_text() + "\n"
        return text_content
    except Exception as e:
        logging.error(f"Error reading PDF file {file_path}: {str(e)}")
        raise

def encode_pdf_base64(file_path):
    """Encode PDF file as base64 string for API uploads"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    
    try:
        with open(file_path, 'rb') as file:
            pdf_bytes = file.read()
            base64_encoded = base64.b64encode(pdf_bytes).decode('utf-8')
        return base64_encoded
    except Exception as e:
        logging.error(f"Error encoding PDF file {file_path}: {str(e)}")
        raise

def validate_pdf(file_path):
    """Validate that the file is a valid PDF"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not file_path.lower().endswith('.pdf'):
        raise ValueError(f"File {file_path} is not a PDF")
    
    try:
        # Try to read the PDF to validate it
        with open(file_path, 'rb') as file:
            PdfReader(file)
        return True
    except Exception as e:
        logging.error(f"Invalid PDF file {file_path}: {str(e)}")
        raise ValueError(f"Invalid PDF file: {str(e)}")