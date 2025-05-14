"""
Tests for the PDF extractor module.
"""
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from rag_system.modules.pdf_extractor import PDFExtractor
from rag_system.utils.exceptions import PDFExtractionError


class TestPDFExtractor(unittest.TestCase):
    """Test case for the PDFExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.pdf_folder = self.temp_dir.name
        
        # Create a mock PDF file
        self.pdf_path = os.path.join(self.pdf_folder, "test.pdf")
        with open(self.pdf_path, "wb") as f:
            f.write(b"%PDF-1.5\n")  # Minimal PDF signature
        
        self.extractor = PDFExtractor(pdf_folder=self.pdf_folder)
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    def test_get_pdf_files(self):
        """Test get_pdf_files method."""
        pdf_files = self.extractor.get_pdf_files()
        self.assertEqual(len(pdf_files), 1)
        self.assertEqual(pdf_files[0], self.pdf_path)
    
    def test_get_pdf_files_empty_directory(self):
        """Test get_pdf_files with an empty directory."""
        # Remove the PDF file
        os.remove(self.pdf_path)
        
        pdf_files = self.extractor.get_pdf_files()
        self.assertEqual(len(pdf_files), 0)
    
    @patch("rag_system.modules.pdf_extractor.PyPDF2.PdfReader")
    def test_extract_text_with_pypdf2(self, mock_pdf_reader):
        """Test extract_text_with_pypdf2 method."""
        # Set up mock
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Test content"
        mock_pdf_reader.return_value.pages = [mock_page]
        
        # Call method
        text = self.extractor.extract_text_with_pypdf2(self.pdf_path)
        
        # Assertions
        self.assertEqual(text, "Test content\n")
        mock_pdf_reader.assert_called_once()
        mock_page.extract_text.assert_called_once()
    
    @patch("rag_system.modules.pdf_extractor.pdfminer_extract_text")
    def test_extract_text_with_pdfminer(self, mock_extract_text):
        """Test extract_text_with_pdfminer method."""
        # Set up mock
        mock_extract_text.return_value = "Test content from pdfminer"
        
        # Call method
        text = self.extractor.extract_text_with_pdfminer(self.pdf_path)
        
        # Assertions
        self.assertEqual(text, "Test content from pdfminer")
        mock_extract_text.assert_called_once_with(self.pdf_path)
    
    @patch("rag_system.modules.pdf_extractor.PDFExtractor.extract_text_with_pypdf2")
    @patch("rag_system.modules.pdf_extractor.PDFExtractor.extract_text_with_pdfminer")
    def test_extract_text_pypdf2_success(self, mock_pdfminer, mock_pypdf2):
        """Test extract_text when PyPDF2 succeeds."""
        # Set up mocks
        mock_pypdf2.return_value = "Test content from PyPDF2"
        mock_pdfminer.return_value = "Test content from pdfminer"
        
        # Call method
        text = self.extractor.extract_text(self.pdf_path)
        
        # Assertions
        self.assertEqual(text, "Test content from PyPDF2")
        mock_pypdf2.assert_called_once_with(self.pdf_path)
        mock_pdfminer.assert_not_called()
    
    @patch("rag_system.modules.pdf_extractor.PDFExtractor.extract_text_with_pypdf2")
    @patch("rag_system.modules.pdf_extractor.PDFExtractor.extract_text_with_pdfminer")
    def test_extract_text_pypdf2_failure(self, mock_pdfminer, mock_pypdf2):
        """Test extract_text when PyPDF2 fails and pdfminer succeeds."""
        # Set up mocks
        mock_pypdf2.return_value = ""  # PyPDF2 fails
        mock_pdfminer.return_value = "Test content from pdfminer"
        
        # Call method
        text = self.extractor.extract_text(self.pdf_path)
        
        # Assertions
        self.assertEqual(text, "Test content from pdfminer")
        mock_pypdf2.assert_called_once_with(self.pdf_path)
        mock_pdfminer.assert_called_once_with(self.pdf_path)
    
    @patch("rag_system.modules.pdf_extractor.PDFExtractor.extract_text_with_pypdf2")
    @patch("rag_system.modules.pdf_extractor.PDFExtractor.extract_text_with_pdfminer")
    def test_extract_text_all_failures(self, mock_pdfminer, mock_pypdf2):
        """Test extract_text when both extractors fail."""
        # Set up mocks
        mock_pypdf2.return_value = ""  # PyPDF2 fails
        mock_pdfminer.return_value = ""  # pdfminer fails
        
        # Call method and check for exception
        with self.assertRaises(PDFExtractionError):
            self.extractor.extract_text(self.pdf_path)
        
        mock_pypdf2.assert_called_once_with(self.pdf_path)
        mock_pdfminer.assert_called_once_with(self.pdf_path)
    
    def test_chunk_text(self):
        """Test chunk_text method."""
        text = "This is a test text for chunking. " * 20  # Multiple sentences
        chunks = self.extractor.chunk_text(text, chunk_size=100, chunk_overlap=20)
        
        # Assertions
        self.assertGreater(len(chunks), 1)  # Should have multiple chunks
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 100)  # Each chunk within size limit
        
        # Check overlap
        if len(chunks) > 1:
            overlap = chunks[0][-20:]
            self.assertTrue(chunks[1].startswith(overlap))
    
    def test_chunk_text_empty(self):
        """Test chunk_text with empty text."""
        chunks = self.extractor.chunk_text("")
        self.assertEqual(chunks, [])
    
    @patch("rag_system.modules.pdf_extractor.PDFExtractor.extract_text")
    @patch("rag_system.modules.pdf_extractor.PDFExtractor.chunk_text")
    def test_extract_and_chunk_specific_file(self, mock_chunk, mock_extract):
        """Test extract_and_chunk with a specific file."""
        # Set up mocks
        mock_extract.return_value = "Test content"
        mock_chunk.return_value = ["Chunk 1", "Chunk 2"]
        
        # Call method
        result = self.extractor.extract_and_chunk(self.pdf_path)
        
        # Assertions
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], self.pdf_path)
        self.assertEqual(result[0][1], ["Chunk 1", "Chunk 2"])
        mock_extract.assert_called_once_with(self.pdf_path)
        mock_chunk.assert_called_once_with("Test content")
    
    @patch("rag_system.modules.pdf_extractor.PDFExtractor.extract_from_all_pdfs")
    @patch("rag_system.modules.pdf_extractor.PDFExtractor.chunk_text")
    def test_extract_and_chunk_all_files(self, mock_chunk, mock_extract_all):
        """Test extract_and_chunk with all files."""
        # Set up mocks
        mock_extract_all.return_value = {self.pdf_path: "Test content"}
        mock_chunk.return_value = ["Chunk 1", "Chunk 2"]
        
        # Call method
        result = self.extractor.extract_and_chunk()
        
        # Assertions
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], self.pdf_path)
        self.assertEqual(result[0][1], ["Chunk 1", "Chunk 2"])
        mock_extract_all.assert_called_once()
        mock_chunk.assert_called_once_with("Test content")


if __name__ == "__main__":
    unittest.main() 