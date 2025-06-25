"""
Input components for basic information and submissions.
"""
import os
import io
from datetime import datetime
import ipywidgets as widgets
from ..base_classes import BaseComponent
from ..constants import ASSESSMENT_TYPES, GRADE_OPTIONS


class InputWidgetsComponent(BaseComponent):
    """Component for basic information input widgets"""
    
    def __init__(self, manager, include_grade=True):
        """Initialize input widgets component"""
        super().__init__(manager)
        self.include_grade = include_grade
        self.create_widgets()
    
    def create_widgets(self):
        """Create all input widgets"""
        if self.include_grade:
            self.grade_widget = widgets.Dropdown(
                options=GRADE_OPTIONS,
                value=GRADE_OPTIONS[0],
                description='학년:'
            )
        
        self.subject_widget = widgets.Text(
            description="과목:", 
            placeholder='예: 국어'
        )
        
        self.title_widget = widgets.Text(
            description="제목:", 
            placeholder='예: 비혼주의자에 대한 본인의 의견'
        )
        
        self.assessment_type_widget = widgets.Dropdown(
            options=ASSESSMENT_TYPES,
            value=ASSESSMENT_TYPES[0],
            description='유형:'
        )
        
        self.description_widget = widgets.Textarea(
            description="설명:", 
            placeholder='수행평가에 대한 설명을 입력하세요',
            layout=widgets.Layout(width='60%', height='80px')
        )
    
    def get_widgets(self):
        """Return all widgets in this component"""
        if self.include_grade:
            return [
                self.grade_widget,
                self.subject_widget,
                self.title_widget,
                self.assessment_type_widget,
                self.description_widget
            ]
        else:
            return [
                self.subject_widget,
                self.title_widget,
                self.assessment_type_widget,
                self.description_widget
            ]
    
    def create_layout(self):
        """Create a layout for the input widgets"""
        return widgets.VBox(self.get_widgets())
    
    def get_values(self):
        """Get all input values as a dictionary"""
        values = {
            '과목': self.subject_widget.value,
            '수행평가 제목': self.title_widget.value,
            '수행평가 유형': self.assessment_type_widget.value,
            '수행평가 설명': self.description_widget.value,
        }
        
        if self.include_grade:
            values['학년'] = self.grade_widget.value
            
        return values
    
    def validate_inputs(self):
        """Validate that all required inputs have values"""
        values = self.get_values()
        required = list(values.keys())
        
        # Check that all required fields have values
        missing = [key for key in required if not values.get(key)]
        
        if missing:
            missing_fields = ', '.join(missing)
            self.manager.display_error(f"다음 필드를 입력해주세요: {missing_fields}")
            return False
            
        return True


class StudentSubmissionWidgetsComponent(BaseComponent):
    """Component for student submission content widgets"""
    
    def __init__(self, manager):
        """Initialize student submission widgets component"""
        super().__init__(manager)
        self.file_name = ""
        self.create_widgets()
    
    def create_widgets(self):
        """Create submission widgets"""
        # File upload widget
        self.file_upload = widgets.FileUpload(description="제출물 업로드:")
        
        # Submission text area
        self.submission_text = widgets.Textarea(
            description="제출물 내용:",
            placeholder='학생 제출물을 직접 입력하거나, 위에서 파일을 업로드하세요.',
            layout=widgets.Layout(width='80%', height='200px')
        )
        
        # Add file upload handler
        self.file_upload.observe(self.process_uploaded_file, names='value')
    
    def get_widgets(self):
        """Return all widgets in this component"""
        return [
            self.file_upload,
            self.submission_text
        ]
    
    def create_layout(self):
        """Create a layout for the submission widgets"""
        return widgets.VBox(self.get_widgets())
    
    def process_uploaded_file(self, change):
        """Process uploaded file and update submission text"""
        if not self.file_upload.value:
            return
            
        try:
            # Get uploaded file content
            uploaded_content = self.file_upload.value[0]['content']
            self.submission_text.value = io.BytesIO(uploaded_content).read().decode('utf-8')
            
            # Extract file name without extension
            file_name = self.file_upload.value[0]['name']
            self.file_name = os.path.splitext(file_name)[0]
        except Exception as e:
            print(f"파일 처리 중 오류: {str(e)}")
    
    def get_submission_text(self):
        """Get the current submission text"""
        return self.submission_text.value
    
    def get_file_name(self):
        """Get the current file name"""
        return self.file_name


class UploadPdfWidgetsComponent(BaseComponent):
    """Component for PDF file upload widgets for API requests"""
    
    def __init__(self, manager):
        """Initialize PDF upload widgets component"""
        super().__init__(manager)
        self.pdf_file_path = None
        self.create_widgets()
    
    def create_widgets(self):
        """Create PDF upload widgets"""
        # PDF upload widget for API requests
        self.pdf_upload = widgets.FileUpload(
            description="PDF 업로드:",
            accept=".pdf",
            layout=widgets.Layout(width='300px')
        )
        
        # Add file upload handler
        self.pdf_upload.observe(self.process_uploaded_pdf, names='value')
    
    def get_widgets(self):
        """Return all widgets in this component"""
        return [
            widgets.HTML("<h4>PDF 업로드</h4>"),
            self.pdf_upload
        ]
    
    def create_layout(self):
        """Create a layout for the PDF upload widgets"""
        return widgets.VBox(self.get_widgets())
    
    def process_uploaded_pdf(self, change):
        """Process uploaded PDF file for API requests"""
        if not self.pdf_upload.value:
            return
            
        try:
            # Clean up previous temporary file if it exists
            if self.pdf_file_path and os.path.exists(self.pdf_file_path):
                try:
                    os.remove(self.pdf_file_path)
                    self.manager.logger.info(f"이전 임시 PDF 파일 삭제: {self.pdf_file_path}")
                except Exception as e:
                    self.manager.logger.error(f"이전 임시 PDF 파일 삭제 중 오류: {str(e)}")
            
            # Get uploaded file content
            uploaded_file = self.pdf_upload.value[0]
            
            # Create a temporary file path
            temp_path = f"./temp/temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file['content'])
            
            self.pdf_file_path = temp_path
            
            # Register this temporary file with the manager for cleanup
            self.manager.add_temp_file(temp_path)
            
            self.manager.logger.info(f"PDF uploaded and saved to {temp_path}")
        except Exception as e:
            self.manager.logger.error(f"PDF 업로드 중 오류: {str(e)}")
            
    def get_pdf_path(self):
        """Get the current PDF file path"""
        return self.pdf_file_path
