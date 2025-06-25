"""
LLM Call manager for direct API calls without schema enforcement.
Allows raw interaction with LLM APIs for testing and general purpose queries.
"""
import ipywidgets as widgets
from IPython.display import display

from ..base_classes import BaseWidgetManager
from ..components.template_components import TemplateWidgetsComponent
from ..components.model_components import ModelSelectionComponent
from ..components.upload_components import UploadPdfWidgetsComponent
from ..output_handlers.llm_call_output import LlmCallOutputComponent
from ..utils import sanitize_filename


class LlmCallManager(BaseWidgetManager):
    """Manager for direct LLM API calls without schema enforcement"""
    
    def __init__(self):
        """Initialize the LLM call manager"""
        super().__init__()
        
        # Create components with empty templates for free-form input
        template_component = TemplateWidgetsComponent(
            self, 
            system_template="",  # Empty system template
            prompt_template=""   # Empty prompt template
        )
        model_component = ModelSelectionComponent(self)
        model_component.set_action_button_text("LLM 요청 실행")
        model_component.set_action_handler(self.make_llm_call)
        
        output_component = LlmCallOutputComponent(self)
        
        # Add PDF upload component
        pdf_upload_component = UploadPdfWidgetsComponent(self)
        
        # Add components
        self.add_component('template', template_component)
        self.add_component('model', model_component)
        self.add_component('output', output_component)
        self.add_component('pdf_upload', pdf_upload_component)
        
        # Set save handlers
        for model in ['Gemini', 'Claude', 'OpenAI']:
            model_component.set_save_handler(model, output_component.create_save_handler(model))
    
    def make_llm_call(self, b=None):
        """Execute the LLM API call process"""
        with self.output_area:
            self.output_area.clear_output()
            self.error_area.clear_output()
            
            # Get template values
            system_instruction = self.template_component.get_formatted_system_instruction()
            prompt = self.template_component.get_formatted_prompt()
            
            # Validate that we have at least a prompt
            if not prompt.strip():
                self.display_error("프롬프트를 입력해주세요.")
                return
            
            print("System Instruction:")
            print(system_instruction if system_instruction.strip() else "(없음)")
            print("\\nPrompt:")
            print(prompt)
            print("\\nLLM API 요청 중...")
            
            # Get PDF path if uploaded
            pdf_path = self.pdf_upload_component.get_pdf_path()
            
            # Use process_with_models but explicitly pass None for schema
            self.process_with_models_without_schema(
                prompt=prompt,
                system_instruction=system_instruction if system_instruction.strip() else None,
                pdf_path=pdf_path,
                completion_message="LLM API 요청 완료"
            )
    
    def process_with_models_without_schema(self, prompt, system_instruction, pdf_path=None, completion_message=None):
        """Process request with selected models without requiring a schema
        
        Args:
            prompt: The prompt to send to the models
            system_instruction: The system instruction (can be None)
            pdf_path: Optional path to a PDF file to process
            completion_message: Optional custom completion message to display
            
        Returns:
            True if processing completed, False otherwise
        """
        # Initialize API client
        client = self.initialize_api_client()
        
        # Process with selected models
        models = self.model_component.get_selected_models()
        
        if not models:
            self.display_error("최소 하나의 모델을 선택해주세요.")
            return False
        
        for model_name, model_params in models.items():
            if not model_params.get('model'):
                continue
                
            try:
                print(f"\\n{model_name}로 처리 중...")
                
                # Call the appropriate processing method based on model name
                if model_name == 'Gemini':
                    response_dict = client._process_with_gemini(
                        file_path=pdf_path,
                        prompt=prompt,
                        model=model_params['model'],
                        system_instruction=system_instruction,
                        response_schema=None,  # No schema enforcement
                        temperature=model_params.get('temperature', 0.1),
                        max_tokens=model_params.get('max_tokens', 4096),
                        enable_thinking=model_params.get('enable_thinking', False),
                        thinking_budget=model_params.get('thinking_budget')
                    )
                    # Extract text from Gemini response
                    response_text = response_dict['response'].candidates[0].content.parts[0].text
                    
                elif model_name == 'Claude':
                    response_dict = client._process_with_anthropic(
                        file_path=pdf_path,
                        prompt=prompt,
                        model=model_params['model'],
                        system_instruction=system_instruction,
                        response_schema=None,  # No schema enforcement
                        temperature=model_params.get('temperature', 0.2),
                        max_tokens=model_params.get('max_tokens', 4096),
                        enable_thinking=model_params.get('enable_thinking', False),
                        thinking_budget=model_params.get('thinking_budget')
                    )
                    # Extract text from Claude response
                    response_text = response_dict['response'].content[0].text
                    
                elif model_name == 'OpenAI':
                    response_dict = client._process_with_openai(
                        file_path=pdf_path,
                        prompt=prompt,
                        model=model_params['model'],
                        system_instruction=system_instruction,
                        response_schema=None,  # No schema enforcement
                        temperature=model_params.get('temperature', 0.2),
                        max_tokens=model_params.get('max_tokens', 4096),
                        enable_thinking=model_params.get('enable_thinking', False),
                        thinking_budget=model_params.get('thinking_budget')
                    )
                    # Extract text from OpenAI response
                    if model_params.get('enable_thinking', False):
                        # For thinking models, get the final output
                        response_text = response_dict['response'].output[1].content[0].text
                    else:
                        response_text = response_dict['response'].choices[0].message.content
                else:
                    continue
                
                # Store result in output component
                self.output_component.store_result(model_name, response_text)
                
                if completion_message:
                    print(f"{model_name} {completion_message}")
                else:
                    print(f"{model_name} 처리 완료")
                
                # Enable save button
                self.model_component.enable_save_button(model_name)
                
            except Exception as e:
                error_msg = f"{model_name} 처리 중 오류: {str(e)}"
                with self.error_area:
                    print(error_msg)
                if hasattr(self, 'logger'):
                    self.logger.error(error_msg)
        
        # Display token usage summary
        try:
            usage_summary = client.get_token_usage_summary()
            self.output_component.display_token_usage(usage_summary)
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"토큰 사용량 표시 중 오류: {str(e)}")
        
        return True
    
    def display_all(self):
        """Display all widgets for LLM API calls"""
        # Create tabs
        tab_content = widgets.Tab()
        
        # Tab 0: 프롬프트 편집
        templates = widgets.VBox([
            widgets.HTML("<h3>프롬프트 편집</h3>"),
            self.template_component.create_layout(),
            widgets.HTML("<h3>PDF 업로드</h3>"),
            self.pdf_upload_component.create_layout()
        ])
        
        # Tab 1: 모델 설정
        model_settings = widgets.VBox([
            widgets.HTML("<h3>모델 설정</h3>"),
            self.model_component.create_layout()
        ])
        
        # Tab 2: 결과 보기
        results = widgets.VBox([
            widgets.HTML("<h3>결과 보기</h3>"),
            self.output_component.create_layout(),
            self.output_area,
            self.error_area
        ])
        
        # Tab 3: 로그 및 토큰
        logs = widgets.VBox([
            widgets.HTML("<h3>시스템 로그</h3>"),
            self.log_output,
            widgets.HTML("<h3>토큰 사용량</h3>"),
            self.output_component.token_usage_output
        ])
        
        # Set tab contents
        tab_content.children = [templates, model_settings, results, logs]
        
        # Set tab titles
        tab_content.set_title(0, "프롬프트 편집")
        tab_content.set_title(1, "모델 설정")
        tab_content.set_title(2, "결과 보기")
        tab_content.set_title(3, "로그 및 토큰")
        
        # Display the tab widget
        display(tab_content)
