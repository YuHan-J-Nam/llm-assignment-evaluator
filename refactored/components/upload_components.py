"""
PDF upload components for the educational assessment system.
Handles PDF file uploads with validation and temporary file management.
"""
import os
import io
import re
from datetime import datetime
from typing import Optional
import ipywidgets as widgets
from IPython.display import display

from ..base_classes import BaseComponent
from ..utils import ensure_directories_exist


class UploadPdfWidgetsComponent(BaseComponent):
    """Component for PDF file upload widgets for API requests
    
    Provides file upload functionality with proper validation and temporary file management.
    Integrates with the PDF utilities for validation and encoding.
    """
    
    def __init__(self, manager):
        """Initialize PDF upload widgets component
        
        Args:
            manager: The widget manager instance
        """
        super().__init__(manager)
        self.pdf_file_path = None
        self.original_filename = None
        self.create_widgets()
    
    def create_widgets(self):
        """Create PDF upload widgets"""
        # Ensure temp directory exists
        ensure_directories_exist(['./temp'])
        
        # PDF upload widget for API requests
        self.pdf_upload = widgets.FileUpload(
            description="PDF 업로드:",
            accept=".pdf",
            multiple=False,
            layout=widgets.Layout(width='400px')
        )
        
        # Status display widget
        self.status_output = widgets.Output()
        
        # Clear button to remove uploaded file
        self.clear_button = widgets.Button(
            description="파일 제거",
            button_style='warning',
            disabled=True,
            layout=widgets.Layout(width='100px')
        )
        
        # Add event handlers
        self.pdf_upload.observe(self.process_uploaded_pdf, names='value')
        self.clear_button.on_click(self.clear_uploaded_file)
    
    def get_widgets(self):
        """Return all widgets in this component"""
        return [
            widgets.HTML("<h4>PDF 파일 업로드</h4>"),
            self.pdf_upload,
            self.clear_button,
            self.status_output
        ]
    
    def create_layout(self):
        """Create a layout for the PDF upload widgets"""
        return widgets.VBox(self.get_widgets())
    
    def process_uploaded_pdf(self, change):
        """Process uploaded PDF file for API requests
        
        Args:
            change: The change event from the file upload widget
        """
        if not self.pdf_upload.value:
            return
            
        try:
            with self.status_output:
                self.status_output.clear_output()
                print("PDF 파일 처리 중...")
            
            # Clean up previous temporary file if it exists
            self._cleanup_previous_file()
            
            # Get uploaded file content
            uploaded_file = self.pdf_upload.value[0]
            self.original_filename = uploaded_file['name']
            
            # Validate file extension
            if not self.original_filename.lower().endswith('.pdf'):
                raise ValueError(f"업로드된 파일 '{self.original_filename}'은(는) PDF 파일이 아닙니다.")
            
            # Create a temporary file path with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_filename = self._sanitize_filename(self.original_filename)
            temp_filename = f"temp_{timestamp}_{safe_filename}"
            temp_path = os.path.join("./temp", temp_filename)
            
            # Write the uploaded content to temporary file
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file['content'])
            
            # Validate the PDF file using pdf_utils validation
            try:
                self._validate_pdf_file(temp_path)
            except Exception as validation_error:
                # Remove invalid file and raise error
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise ValueError(f"유효하지 않은 PDF 파일: {str(validation_error)}")
            
            # Store the file path
            self.pdf_file_path = temp_path
            
            # Register this temporary file with the manager for cleanup
            if hasattr(self.manager, 'add_temp_file'):
                self.manager.add_temp_file(temp_path)
            
            # Update UI
            self.clear_button.disabled = False
            
            with self.status_output:
                self.status_output.clear_output()
                print(f"✓ PDF 파일 업로드 완료: {self.original_filename}")
                print(f"  임시 저장 위치: {temp_path}")
                
            # Log success
            if hasattr(self.manager, 'logger'):
                self.manager.logger.info(f"PDF 파일 업로드 및 검증 완료: {self.original_filename} -> {temp_path}")
                
        except Exception as e:
            error_msg = f"PDF 업로드 중 오류: {str(e)}"
            
            with self.status_output:
                self.status_output.clear_output()
                print(f"❌ {error_msg}")
            
            # Log error
            if hasattr(self.manager, 'logger'):
                self.manager.logger.error(error_msg)
                
            # Reset state
            self.pdf_file_path = None
            self.original_filename = None
            self.clear_button.disabled = True
    
    def clear_uploaded_file(self, button=None):
        """Clear the uploaded PDF file
        
        Args:
            button: The button widget that triggered this event (unused)
        """
        try:
            # Clean up the temporary file
            self._cleanup_previous_file()
            
            # Reset widget state
            self.pdf_upload.value = ()
            self.pdf_file_path = None
            self.original_filename = None
            self.clear_button.disabled = True
            
            with self.status_output:
                self.status_output.clear_output()
                print("파일이 제거되었습니다.")
                
            # Log removal
            if hasattr(self.manager, 'logger'):
                self.manager.logger.info("업로드된 PDF 파일이 제거되었습니다.")
                
        except Exception as e:
            error_msg = f"파일 제거 중 오류: {str(e)}"
            
            with self.status_output:
                self.status_output.clear_output()
                print(f"❌ {error_msg}")
                
            if hasattr(self.manager, 'logger'):
                self.manager.logger.error(error_msg)
    
    def get_pdf_path(self) -> Optional[str]:
        """Get the current PDF file path
        
        Returns:
            str or None: The path to the uploaded PDF file, or None if no file is uploaded
        """
        return self.pdf_file_path
    
    def get_original_filename(self) -> Optional[str]:
        """Get the original filename of the uploaded PDF
        
        Returns:
            str or None: The original filename, or None if no file is uploaded
        """
        return self.original_filename
    
    def has_pdf(self) -> bool:
        """Check if a PDF file is currently uploaded
        
        Returns:
            bool: True if a PDF file is uploaded and available, False otherwise
        """
        return self.pdf_file_path is not None and os.path.exists(self.pdf_file_path)
    
    def _cleanup_previous_file(self):
        """Clean up any previously uploaded temporary file"""
        if self.pdf_file_path and os.path.exists(self.pdf_file_path):
            try:
                os.remove(self.pdf_file_path)
                if hasattr(self.manager, 'logger'):
                    self.manager.logger.info(f"이전 임시 PDF 파일 삭제: {self.pdf_file_path}")
            except Exception as e:
                if hasattr(self.manager, 'logger'):
                    self.manager.logger.error(f"이전 임시 PDF 파일 삭제 중 오류: {str(e)}")
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe temporary file creation
        
        Args:
            filename: The original filename
            
        Returns:
            str: A sanitized filename safe for file system use
        """
        # Remove directory paths and keep only the filename
        filename = os.path.basename(filename)
        
        # Replace problematic characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Limit length
        if len(filename) > 100:
            name, ext = os.path.splitext(filename)
            filename = name[:96] + ext
            
        return filename
    
    def _validate_pdf_file(self, file_path: str):
        """Validate that the uploaded file is a proper PDF
        
        This method uses the same validation logic as the API utilities
        to ensure consistency with PDF processing.
        
        Args:
            file_path: Path to the PDF file to validate
            
        Raises:
            ValueError: If the file is not a valid PDF
            FileNotFoundError: If the file doesn't exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        
        if not file_path.lower().endswith('.pdf'):
            raise ValueError(f"파일 {file_path}은(는) PDF가 아닙니다.")
        
        try:
            # Use PyPDF2 to validate the PDF file (same as pdf_utils.validate_pdf)
            from PyPDF2 import PdfReader
            with open(file_path, 'rb') as file:
                PdfReader(file)
        except Exception as e:
            raise ValueError(f"유효하지 않은 PDF 파일: {str(e)}")
    
    def __del__(self):
        """Cleanup when the component is destroyed"""
        self._cleanup_previous_file()


class StudentSubmissionWidgetsComponent(BaseComponent):
    """Component for student submission content widgets
    
    Allows input of student submissions either through text input or file upload.
    """
    
    def __init__(self, manager):
        """Initialize student submission widgets component
        
        Args:
            manager: The widget manager instance
        """
        super().__init__(manager)
        self.file_name = ""
        self.create_widgets()
    
    def create_widgets(self):
        """Create submission widgets"""
        # File upload widget for text files
        self.file_upload = widgets.FileUpload(
            description="제출물 파일:",
            accept=".txt,.docx,.doc",
            multiple=False,
            layout=widgets.Layout(width='400px')
        )
        
        # Submission text area
        self.submission_text = widgets.Textarea(
            description="제출물 내용:",
            placeholder='학생 제출물을 직접 입력하거나, 위에서 파일을 업로드하세요.',
            layout=widgets.Layout(width='90%', height='250px')
        )
        
        # Status output
        self.status_output = widgets.Output()
        
        # Clear button
        self.clear_button = widgets.Button(
            description="내용 지우기",
            button_style='warning',
            layout=widgets.Layout(width='120px')
        )
        
        # Add event handlers
        self.file_upload.observe(self.process_uploaded_file, names='value')
        self.clear_button.on_click(self.clear_submission)
    
    def get_widgets(self):
        """Return all widgets in this component"""
        return [
            widgets.HTML("<h4>학생 제출물</h4>"),
            self.file_upload,
            self.submission_text,
            widgets.HBox([self.clear_button]),
            self.status_output
        ]
    
    def create_layout(self):
        """Create a layout for the submission widgets"""
        return widgets.VBox(self.get_widgets())
    
    def process_uploaded_file(self, change):
        """Process uploaded file and update submission text
        
        Args:
            change: The change event from the file upload widget
        """
        if not self.file_upload.value:
            return
            
        try:
            with self.status_output:
                self.status_output.clear_output()
                print("파일 처리 중...")
            
            # Get uploaded file content
            uploaded_file = self.file_upload.value[0]
            file_name = uploaded_file['name']
            content = uploaded_file['content']
            
            # Try to decode the content as text
            try:
                # Try UTF-8 first
                text_content = content.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    # Try with encoding detection
                    import chardet
                    detected = chardet.detect(content)
                    encoding = detected['encoding'] or 'utf-8'
                    text_content = content.decode(encoding)
                except:
                    # Fallback to latin-1 which accepts any byte
                    text_content = content.decode('latin-1')
            
            # Update the text area
            self.submission_text.value = text_content
            
            # Extract file name without extension for future use
            self.file_name = os.path.splitext(file_name)[0]
            
            with self.status_output:
                self.status_output.clear_output()
                print(f"✓ 파일 업로드 완료: {file_name}")
                print(f"  텍스트 길이: {len(text_content)} 문자")
                
            # Log success
            if hasattr(self.manager, 'logger'):
                self.manager.logger.info(f"학생 제출물 파일 업로드 완료: {file_name}")
                
        except Exception as e:
            error_msg = f"파일 처리 중 오류: {str(e)}"
            
            with self.status_output:
                self.status_output.clear_output()
                print(f"❌ {error_msg}")
                
            if hasattr(self.manager, 'logger'):
                self.manager.logger.error(error_msg)
    
    def clear_submission(self, button=None):
        """Clear the submission text and file upload
        
        Args:
            button: The button widget that triggered this event (unused)
        """
        self.submission_text.value = ""
        self.file_upload.value = ()
        self.file_name = ""
        
        with self.status_output:
            self.status_output.clear_output()
            print("제출물 내용이 지워졌습니다.")
            
        if hasattr(self.manager, 'logger'):
            self.manager.logger.info("학생 제출물 내용이 지워졌습니다.")
    
    def get_submission_text(self) -> str:
        """Get the current submission text
        
        Returns:
            str: The current submission text content
        """
        return self.submission_text.value
    
    def get_file_name(self) -> str:
        """Get the current file name (without extension)
        
        Returns:
            str: The file name without extension, or empty string if no file uploaded
        """
        return self.file_name
    
    def has_content(self) -> bool:
        """Check if there is any submission content
        
        Returns:
            bool: True if there is text content, False otherwise
        """
        return bool(self.submission_text.value.strip())
