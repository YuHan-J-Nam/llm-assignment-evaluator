import os
import base64
import logging
from PyPDF2 import PdfReader
import io

def read_pdf_content(file_path):
    """PDF 파일에서 텍스트 내용을 추출합니다."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {file_path}")
    
    try:
        text_content = ""
        with open(file_path, 'rb') as file:
            pdf = PdfReader(file)
            for page in pdf.pages:
                text_content += page.extract_text() + "\n"
        return text_content
    except Exception as e:
        logging.error(f"PDF 파일을 읽는 중 오류 발생 {file_path}: {str(e)}")
        raise

def encode_pdf_base64(file_path):
    """API 업로드를 위해 PDF 파일을 base64 문자열로 인코딩합니다."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {file_path}")
    
    try:
        with open(file_path, 'rb') as file:
            pdf_bytes = file.read()
            base64_encoded = base64.b64encode(pdf_bytes).decode('utf-8')
        return base64_encoded
    except Exception as e:
        logging.error(f"PDF 파일 인코딩 중 오류 발생 {file_path}: {str(e)}")
        raise

def validate_pdf(file_path):
    """파일이 올바른 PDF인지 검증합니다."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
    
    if not file_path.lower().endswith('.pdf'):
        raise ValueError(f"파일 {file_path}은(는) PDF가 아닙니다.")
    
    try:
        # PDF를 읽어서 유효성 검증
        with open(file_path, 'rb') as file:
            PdfReader(file)
        return True
    except Exception as e:
        logging.error(f"유효하지 않은 PDF 파일 {file_path}: {str(e)}")
        raise ValueError(f"유효하지 않은 PDF 파일: {str(e)}")