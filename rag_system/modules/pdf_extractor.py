"""
Module for extracting text from PDF files.
"""
import os
import traceback
from typing import List, Dict, Optional, Tuple

import PyPDF2
from pdfminer.high_level import extract_text as pdfminer_extract_text

from rag_system.config import PDF_DATA_FOLDER, CHUNK_SIZE, CHUNK_OVERLAP
from rag_system.utils.logger import setup_logger
from rag_system.utils.exceptions import PDFExtractionError


logger = setup_logger(__name__)


class PDFExtractor:
    """
    Class for extracting text from PDF files.
    """
    
    def __init__(self, pdf_folder: str = PDF_DATA_FOLDER):
        """
        Initialize the PDFExtractor.
        
        Args:
            pdf_folder (str): Path to the folder containing PDF files.
        """
        self.pdf_folder = pdf_folder
        logger.info(f"PDFExtractor initialized with folder: {pdf_folder}")
    
    def get_pdf_files(self) -> List[str]:
        """
        Get a list of PDF files in the specified folder.
        
        Returns:
            List[str]: List of PDF file paths.
        """
        pdf_files = []
        try:
            for filename in os.listdir(self.pdf_folder):
                if filename.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(self.pdf_folder, filename))
            
            logger.info(f"Found {len(pdf_files)} PDF files in {self.pdf_folder}")
            return pdf_files
        
        except Exception as e:
            logger.error(f"Error getting PDF files: {str(e)}")
            raise PDFExtractionError(f"Error getting PDF files: {str(e)}")
    
    def extract_text_with_pypdf2(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file using PyPDF2.
        
        Args:
            pdf_path (str): Path to the PDF file.
            
        Returns:
            str: Extracted text.
        """
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text += page.extract_text() + "\n"
                
                return text
        
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed for {pdf_path}: {str(e)}")
            return ""
    
    def extract_text_with_pdfminer(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file using pdfminer.six.
        
        Args:
            pdf_path (str): Path to the PDF file.
            
        Returns:
            str: Extracted text.
        """
        try:
            text = pdfminer_extract_text(pdf_path)
            return text
        
        except Exception as e:
            logger.warning(f"pdfminer extraction failed for {pdf_path}: {str(e)}")
            return ""
    
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file using multiple methods.
        
        Args:
            pdf_path (str): Path to the PDF file.
            
        Returns:
            str: Extracted text.
            
        Raises:
            PDFExtractionError: If text extraction fails with all methods.
        """
        logger.info(f"Extracting text from {pdf_path}")
        
        # Try PyPDF2 first
        text = self.extract_text_with_pypdf2(pdf_path)
        
        # If PyPDF2 fails, try pdfminer
        if not text.strip():
            logger.info(f"PyPDF2 extraction yielded no text, trying pdfminer for {pdf_path}")
            text = self.extract_text_with_pdfminer(pdf_path)
        
        # If all methods fail, raise an error
        if not text.strip():
            error_msg = f"Failed to extract text from {pdf_path} with all methods"
            logger.error(error_msg)
            raise PDFExtractionError(error_msg)
        
        logger.info(f"Successfully extracted {len(text)} characters from {pdf_path}")
        return text
    
    def extract_from_all_pdfs(self) -> Dict[str, str]:
        """
        Extract text from all PDF files in the folder.
        
        Returns:
            Dict[str, str]: Dictionary mapping PDF file paths to extracted text.
        """
        pdf_files = self.get_pdf_files()
        result = {}
        
        for pdf_path in pdf_files:
            try:
                text = self.extract_text(pdf_path)
                result[pdf_path] = text
            except PDFExtractionError as e:
                logger.error(f"Skipping {pdf_path} due to extraction error: {str(e)}")
                continue
        
        logger.info(f"Extracted text from {len(result)} out of {len(pdf_files)} PDF files")
        return result
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, 
                  chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text (str): Text to split.
            chunk_size (int): Size of each chunk.
            chunk_overlap (int): Overlap between chunks.
            
        Returns:
            List[str]: List of text chunks.
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunks.append(text[start:end])
            start = end - chunk_overlap
            
            # Break if we've reached the end of the text
            if start >= text_length - chunk_overlap:
                break
        
        return chunks
    
    def extract_and_chunk(self, pdf_path: Optional[str] = None) -> List[Tuple[str, List[str]]]:
        """
        Extract text from PDFs and split into chunks.
        
        Args:
            pdf_path (Optional[str]): Path to a specific PDF file. If None, process all PDFs.
            
        Returns:
            List[Tuple[str, List[str]]]: List of tuples containing PDF path and its text chunks.
        """
        result = []
        
        if pdf_path:
            if not os.path.exists(pdf_path):
                raise PDFExtractionError(f"PDF file not found: {pdf_path}")
            
            try:
                text = self.extract_text(pdf_path)
                chunks = self.chunk_text(text)
                result.append((pdf_path, chunks))
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {str(e)}")
                logger.debug(traceback.format_exc())
                raise PDFExtractionError(f"Error processing {pdf_path}: {str(e)}")
        else:
            pdf_texts = self.extract_from_all_pdfs()
            for pdf_path, text in pdf_texts.items():
                chunks = self.chunk_text(text)
                result.append((pdf_path, chunks))
        
        return result 