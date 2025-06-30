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

# Import RAG functionality
try:
    from ..rag.rag_integration import RAGChecklistEnhancer
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("RAG functionality not available. Install required dependencies if needed.")


class ChecklistCreationManager(BaseWidgetManager):
    """Manager for checklist creation task"""
    
    def __init__(self):
        """Initialize the checklist creation manager"""
        super().__init__()
        
        # Initialize RAG enhancer
        self.rag_enhancer = RAGChecklistEnhancer() if RAG_AVAILABLE else None
        
        # Create components
        input_component = InputWidgetsComponent(self, include_grade=False)
        template_component = TemplateWidgetsComponent(
            self, 
            DEFAULT_SYSTEM_INSTRUCTION_CHECKLIST,
            DEFAULT_PROMPT_CHECKLIST
        )
        
        # Disable RAG toggle if RAG is not available
        # if not RAG_AVAILABLE or (self.rag_enhancer and not self.rag_enhancer.is_available()):
        #     template_component.set_rag_enabled(False)
        #     template_component.rag_toggle_button.disabled = True
        #     template_component.rag_toggle_button.description = 'RAG 사용 불가'
        #     template_component.rag_toggle_button.button_style = 'danger'
        #     template_component.rag_toggle_button.tooltip = 'OpenSearch 설정이 필요합니다.'

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
        for model in ['Gemini', 'Anthropic', 'OpenAI']:
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
            original_prompt = self.template_component.get_formatted_prompt()
            
            # Enhance prompt with RAG if available and enabled
            if (self.rag_enhancer and 
                self.rag_enhancer.is_available() and 
                self.template_component.is_rag_enabled()):
                try:
                    subject = self.input_component.subject_widget.value
                    assessment_type = self.input_component.assessment_type_widget.value
                    assessment_title = self.input_component.title_widget.value
                    assessment_description = self.input_component.description_widget.value
                    
                    enhanced_prompt = self.rag_enhancer.get_enhanced_prompt(
                        original_prompt, subject, assessment_type,
                        assessment_title, assessment_description
                    )
                    print("✓ RAG enhancement applied to prompt")
                    prompt = enhanced_prompt
                    
                except Exception as e:
                    print(f"RAG enhancement failed, using original prompt: {e}")
                    prompt = original_prompt
            else:
                prompt = original_prompt
                if self.rag_enhancer and not self.rag_enhancer.is_available():
                    print("ⓘ RAG functionality initialized but OpenSearch not configured")
                elif not self.template_component.is_rag_enabled():
                    print("ⓘ RAG functionality disabled by user")

            # Reconfigure template_component values
            self.template_component.system_instruction_widget.value = system_instruction
            self.template_component.prompt_widget.value = prompt
            
            print("\n체크리스트 생성 중...")
            
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
            self.output_area,
            self.output_component.create_layout()
        ])
        
        # Tab 4: 로그 및 토큰
        logs = widgets.VBox([
            widgets.HTML("<h3>시스템 로그</h3>"),
            self.log_output,
            widgets.HTML("<h3>토큰 사용량</h3>"),
            self.output_component.token_usage_output,
            widgets.HTML("<h3>오류 메시지</h3>"),
            self.error_area
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
