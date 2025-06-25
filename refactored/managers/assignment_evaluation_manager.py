"""
Assignment evaluation manager for the educational assessment system.
Handles the workflow for evaluating student assignments using predefined checklists.
"""
import json
import ipywidgets as widgets
from IPython.display import display

from ..base_classes import BaseWidgetManager
from ..components.input_components import InputWidgetsComponent
from ..components.template_components import TemplateWidgetsComponent
from ..components.model_components import ModelSelectionComponent
from ..components.upload_components import UploadPdfWidgetsComponent, StudentSubmissionWidgetsComponent
from ..components.checklist_components import ChecklistComponent
from ..output_handlers.evaluation_output import EvaluationOutputComponent
from ..constants import (
    DEFAULT_SYSTEM_INSTRUCTION_EVALUATION,
    DEFAULT_PROMPT_EVALUATION,
    EVALUATION_SCHEMA
)


class AssignmentEvaluationManager(BaseWidgetManager):
    """Manager for assignment evaluation task"""
    
    def __init__(self):
        """Initialize the assignment evaluation manager"""
        super().__init__()
        
        # Create components
        input_component = InputWidgetsComponent(self, include_grade=True)
        checklist_component = ChecklistComponent(self)
        
        # Create both submission components
        student_submission_component = StudentSubmissionWidgetsComponent(self)
        pdf_upload_component = UploadPdfWidgetsComponent(self)
        
        template_component = TemplateWidgetsComponent(
            self, 
            DEFAULT_SYSTEM_INSTRUCTION_EVALUATION,
            DEFAULT_PROMPT_EVALUATION
        )
        model_component = ModelSelectionComponent(self)
        model_component.set_action_button_text("평가 시작")
        model_component.set_action_handler(self.run_evaluation)
        
        output_component = EvaluationOutputComponent(self)
        
        # Add components
        self.add_component('input', input_component)
        self.add_component('checklist', checklist_component)
        self.add_component('student_submission', student_submission_component)
        self.add_component('pdf_upload', pdf_upload_component)
        self.add_component('template', template_component)
        self.add_component('model', model_component)
        self.add_component('output', output_component)
        
        # Set save handlers
        for model in ['Gemini', 'Claude', 'OpenAI']:
            model_component.set_save_handler(model, output_component.create_save_handler(model))
    
    def run_evaluation(self, b=None):
        """Execute the evaluation process"""
        with self.output_area:
            self.output_area.clear_output()
            self.error_area.clear_output()
            
            # Validate checklist selection
            checklist_value = self.checklist_component.get_selected_checklist()
            if not checklist_value or checklist_value == '체크리스트 없음':
                self.display_error("체크리스트를 선택해주세요.")
                return
            
            # Load selected checklist
            checklist_path = f'./checklists/{checklist_value}.json'
            criteria = self.checklist_component.load_checklist(checklist_path)
            if not criteria:
                self.display_error(f"체크리스트를 로드할 수 없습니다: {checklist_path}")
                return
            
            # Validate submission text
            submission_text = self.student_submission_component.get_submission_text()
            if not submission_text.strip():
                self.display_error("학생 제출물이 비어있습니다.")
                return
            
            # Validate input fields
            if not self.input_component.validate_inputs():
                return
            
            # Format system instruction with input values
            system_instruction = self.template_component.get_formatted_system_instruction()
            
            # Format prompt with checklist and submission
            prompt = self.template_component.get_formatted_prompt({
                '평가 기준': json.dumps(criteria, ensure_ascii=False, indent=2),
                '학생 제출물': submission_text
            })
            
            print("System Instruction:")
            print(system_instruction)
            print("\\n평가 중...")
            
            # Get PDF path if uploaded
            pdf_path = self.pdf_upload_component.get_pdf_path()
            
            # Use common model processing method
            self.process_with_models(
                prompt=prompt,
                system_instruction=system_instruction,
                schema=EVALUATION_SCHEMA,
                pdf_path=pdf_path,
                completion_message="평가 완료"
            )

    def display_all(self):
        """Display all widgets for assignment evaluation"""
        # Create tabs
        tab_content = widgets.Tab()
        
        # Tab 0: 인풋 입력
        content_input = widgets.VBox([
            widgets.HTML("<h3>기본 정보 입력</h3>"),
            self.input_component.create_layout(),
            widgets.HTML("<h3>체크리스트 선택</h3>"),
            self.checklist_component.create_layout(),
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
            widgets.HTML("<h3>평가 결과</h3>"),
            self.output_component.create_layout()
        ])
        
        # Tab 4: 로그 확인
        logs = widgets.VBox([
            widgets.HTML("<h3>시스템 로그</h3>"),
            self.log_output,
            widgets.HTML("<h3>토큰 사용량</h3>"),
            self.output_component.token_usage_output
        ])
        
        # Set tab contents
        tab_content.children = [content_input, templates, model_settings, results, logs]
        
        # Set tab titles
        tab_content.set_title(0, "평가 설정")
        tab_content.set_title(1, "템플릿 편집")
        tab_content.set_title(2, "모델 설정")
        tab_content.set_title(3, "평가 결과")
        tab_content.set_title(4, "로그 확인")
        
        # Display the tab widget
        display(tab_content)
