"""
Checklist creation manager for the educational assessment system.
Handles the workflow for creating evaluation checklists using LLM models.
"""
import ipywidgets as widgets
from IPython.display import display

from ..base_classes import BaseWidgetManager
from ..components.input_components import InputWidgetsComponent
from ..components.template_components import TemplateWidgetsComponent
from ..components.model_components import ModelSelectionComponent
from ..components.upload_components import UploadPdfWidgetsComponent
from ..output_handlers.checklist_output import ChecklistOutputComponent
from ..constants import (
    DEFAULT_SYSTEM_INSTRUCTION_CHECKLIST,
    DEFAULT_PROMPT_CHECKLIST,
    CHECKLIST_SCHEMA
)


class ChecklistCreationManager(BaseWidgetManager):
    """Manager for checklist creation task"""
    
    def __init__(self):
        """Initialize the checklist creation manager"""
        super().__init__()
        
        # Create components
        input_component = InputWidgetsComponent(self, include_grade=False)
        template_component = TemplateWidgetsComponent(
            self, 
            DEFAULT_SYSTEM_INSTRUCTION_CHECKLIST,
            DEFAULT_PROMPT_CHECKLIST
        )
        model_component = ModelSelectionComponent(self)
        model_component.set_action_button_text("체크리스트 생성")
        model_component.set_action_handler(self.create_checklist)
        
        output_component = ChecklistOutputComponent(self)
        
        # Add PDF upload component
        pdf_upload_component = UploadPdfWidgetsComponent(self)
        
        # Add components
        self.add_component('input', input_component)
        self.add_component('template', template_component)
        self.add_component('model', model_component)
        self.add_component('output', output_component)
        self.add_component('pdf_upload', pdf_upload_component)
        
        # Set save handlers
        for model in ['Gemini', 'Claude', 'OpenAI']:
            model_component.set_save_handler(model, output_component.create_save_handler(model))
    
    def create_checklist(self, b=None):
        """Execute the checklist creation process"""
        with self.output_area:
            self.output_area.clear_output()
            self.error_area.clear_output()
            
            # Validate inputs
            if not self.input_component.validate_inputs():
                return
                
            # Get template values
            system_instruction = self.template_component.get_formatted_system_instruction()
            prompt = self.template_component.get_formatted_prompt()
            
            print("System Instruction:")
            print(system_instruction)
            print("\n체크리스트 생성 중.")
            
            # Get PDF path if uploaded
            pdf_path = self.pdf_upload_component.get_pdf_path()
            
            # Get thinking settings from model component
            thinking_settings = self.model_component.get_thinking_settings()
            
            # Use common model processing method from base class
            self.process_with_models(
                prompt=prompt,
                system_instruction=system_instruction,
                schema=CHECKLIST_SCHEMA,
                pdf_path=pdf_path,
                completion_message="체크리스트 생성 완료",
                enable_thinking=thinking_settings['enable_thinking'],
                thinking_budget=thinking_settings['thinking_budget']
            )
    
    def display_all(self):
        """Display all widgets for checklist creation"""
        # Create tabs
        tab_content = widgets.Tab()
        
        # Tab 0: 인풋 입력
        basic_info = widgets.VBox([
            widgets.HTML("<h3>인풋 입력</h3>"),
            self.input_component.create_layout(),
            self.pdf_upload_component.create_layout(),
        ])
        
        # Tab 1: 템플릿 편집
        templates = self.template_component.create_layout()
        
        # Tab 2: 모델 설정
        model_settings = widgets.VBox([
            widgets.HTML("<h3>모델 설정</h3>"),
            self.model_component.create_layout()
        ])
        
        # Tab 3: 결과 보기
        results = widgets.VBox([
            widgets.HTML("<h3>결과 보기</h3>"),
            self.output_component.create_layout(),
            self.output_area,
            self.error_area
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
        tab_content.set_title(0, "인풋 입력")
        tab_content.set_title(1, "템플릿 편집")
        tab_content.set_title(2, "모델 설정")
        tab_content.set_title(3, "결과 보기")
        tab_content.set_title(4, "로그 및 토큰")
        
        # Display the tab widget
        display(tab_content)
