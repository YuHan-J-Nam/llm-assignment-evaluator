"""
Summarize submission manager for the educational assessment system.
Handles the workflow for summarizing student submissions into structured reports.
"""
import ipywidgets as widgets
from IPython.display import display

from ..base_classes import BaseWidgetManager
from ..components.input_components import InputWidgetsComponent
from ..components.template_components import TemplateWidgetsComponent
from ..components.model_components import ModelSelectionComponent
from ..components.upload_components import UploadPdfWidgetsComponent, StudentSubmissionWidgetsComponent
from ..output_handlers.summarize_output import SummarizeOutputComponent
from ..constants import (
    DEFAULT_SYSTEM_INSTRUCTION_SUMMARIZE,
    DEFAULT_PROMPT_SUMMARIZE,
    SUMMARIZE_SCHEMA
)


class SummarizeSubmissionManager(BaseWidgetManager):
    """Manager for summarizing submission task"""

    def __init__(self):
        """Initialize the summarize submission manager"""
        super().__init__()
        
        # Create components
        input_component = InputWidgetsComponent(self, include_grade=False)

        # Create both submission components
        student_submission_component = StudentSubmissionWidgetsComponent(self)
        pdf_upload_component = UploadPdfWidgetsComponent(self)

        template_component = TemplateWidgetsComponent(
            self, 
            DEFAULT_SYSTEM_INSTRUCTION_SUMMARIZE,
            DEFAULT_PROMPT_SUMMARIZE
        )

        template_component.set_rag_enabled(False)
        template_component.rag_toggle_button.description = 'RAG 사용하지 않음'
        template_component.rag_toggle_button.button_style = 'danger'
        template_component.rag_toggle_button.tooltip = 'OpenSearch 설정이 필요합니다.'

        model_component = ModelSelectionComponent(self)
        model_component.set_action_button_text("학생 보고서 요약")
        model_component.set_action_handler(self.summarize_submission)
        
        output_component = SummarizeOutputComponent(self)
        
        # Add components
        self.add_component('input', input_component)
        self.add_component('template', template_component)
        self.add_component('model', model_component)
        self.add_component('output', output_component)
        self.add_component('student_submission', student_submission_component)
        self.add_component('pdf_upload', pdf_upload_component)
        
        # Set save handlers
        for model in ['Gemini', 'Anthropic', 'OpenAI']:
            model_component.set_save_handler(model, output_component.create_save_handler(model))
    
    def summarize_submission(self, b=None):
        """Execute the summarize submission process"""
        with self.output_area:
            self.output_area.clear_output()
            self.error_area.clear_output()
            
            # Validate submission text
            submission_text = self.student_submission_component.get_submission_text()
            if not submission_text.strip():
                self.display_error("학생 제출물이 비어있습니다.")
                return

            # Validate inputs
            if not self.input_component.validate_inputs():
                return
                
            # Format system instruction with input values
            system_instruction = self.template_component.get_formatted_system_instruction()
            
            # Format prompt with submission
            prompt = self.template_component.get_formatted_prompt({
                '학생 제출물': submission_text
            })
            
            print("\n보고서 요약 중...")
            
            # Get PDF path if uploaded
            pdf_path = self.pdf_upload_component.get_pdf_path()
            
            # Use common model processing method
            self.process_with_models(
                prompt=prompt,
                system_instruction=system_instruction,
                schema=SUMMARIZE_SCHEMA,
                pdf_path=pdf_path,
                completion_message="보고서 요약 완료"
            )
    
    def display_all(self):
        """Display all widgets for summarize submission"""
        # Create tabs
        tab_content = widgets.Tab()
        
        # Tab 0: 인풋 입력
        basic_info = widgets.VBox([
            widgets.HTML("<h3>기본 정보 입력</h3>"),
            self.input_component.create_layout(),
            widgets.HTML("<h3>학생 제출물</h3>"),
            self.student_submission_component.create_layout(),
            widgets.HTML("<h3>PDF 업로드 (선택사항)</h3>"),
            self.pdf_upload_component.create_layout(),
            self.output_area, 
            self.error_area
        ])
        
        # Tab 1: 템플릿 편집
        templates = widgets.VBox([
            widgets.HTML("<h3>템플릿 편집</h3>"),
            self.template_component.create_layout()
        ])
        
        # Tab 2: 모델 설정
        model_settings = widgets.VBox([
            widgets.HTML("<h3>모델 설정</h3>"),
            self.model_component.create_layout()
        ])
        
        # Tab 3: 결과 보기
        results = widgets.VBox([
            widgets.HTML("<h3>요약 결과</h3>"),
            self.output_component.create_layout()
        ])
        
        # Tab 4: 로그 및 토큰
        logs = widgets.VBox([
            widgets.HTML("<h3>시스템 로그</h3>"),
            self.log_output,
            widgets.HTML("<h3>토큰 사용량</h3>"),
            self.output_component.token_usage_output
        ])
        
        # Set tab contents
        tab_content.children = [basic_info, templates, model_settings, results, logs]
        
        # Set tab titles
        tab_content.set_title(0, "기본 설정")
        tab_content.set_title(1, "템플릿 편집")
        tab_content.set_title(2, "모델 설정")
        tab_content.set_title(3, "요약 결과")
        tab_content.set_title(4, "로그 및 토큰")
        
        # Display the tab widget
        display(tab_content)
