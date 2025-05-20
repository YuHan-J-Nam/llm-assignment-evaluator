import os
import io
import json
import re
import logging
from datetime import datetime
from dotenv import load_dotenv
from llm_api_client import LLMAPIClient
import ipywidgets as widgets
from IPython.display import display, HTML

# Default template texts
DEFAULT_SYSTEM_INSTRUCTION_EVALUATION = """
너는 선생님이다. 학생이 제출한 수행평가 과제에 대하여, 각 평가항목을 기반으로 논리적으로 평가하라. 
학생의 점수와 왜 그 점수를 받았는지에 대해 서술하고, 가능하다면 학생이 작성한 과제중 관련 텍스트를 증거로 제시하라.

수행평가에 대한 세부정보는 다음과 같다:

학년: [학년]
과목: [과목]
수행평가 제목: [수행평가 제목]
수행평가 유형: [수행평가 유형]
수행평가 설명: [수행평가 설명]

최종 평가내용을 JSON 형식으로 반환하라.
"""

DEFAULT_PROMPT_EVALUATION = """
다음은 학생이 제출한 수행평가 과제이다. 수행평가 과제에 대한 세부정보와 평가 기준을 고려하여, 각 평가항목에 대하여 논리적으로 평가하라.

평가 기준은 다음과 같다:
[평가 기준]

학생의 수행평가 과제는 다음과 같다:
[학생 제출물]
"""

DEFAULT_SYSTEM_INSTRUCTION_CHECKLIST = """
너는 [과목] 과목의 교사다. 학생들에게 [수행평가 유형] 형태의 수행평가 과제를 부여했다.
사용자가 제시한 수행평가 제목과 설명에 대해서 수행평가 기준을 생성하는 역할을 수행하라.

이 수행평가 과제를 공정하고 체계적으로 평가하기 위해, 다음 조건에 맞는 평가 기준(또는 체크리스트)을 생성하라. 평가기준 체크리스트를 생성할 때, 아래 기준에 맞추어 어떤 평가 기준을 만드는 것이 좋을 지 차근차근 생각해봐라.

1. 4~6개의 평가 항목을 제시할 것
2. 각 항목에는 명확한 평가 목적을 반영할 것 (예: 논리성, 창의성, 과제 이해 등)
3. 학생과 교사가 모두 이해하기 쉬운 언어로 작성할 것
4. 가능하면 과목 및 수행평가 유형의 특성을 반영할 것
5. 체크리스트는 아래와 같은 JSON 형식의 구조로 응답할 것:
    
```json
{
    "checklist": [
    {
        "title": "표현",
        "subcategories": [
        {
            "name": "문법의 정확성",
            "description": "문법의 정확성을 평가",
            "levels": {
            "high": "글이 문법적 오류 없이 완벽함",
            "medium": "글에 문법적 오류가 일부 있음",
            "low": "글에 문법적 오류가 다소 많음"
            }
        }
        ]
    }
    ]
}
```
"""

DEFAULT_PROMPT_CHECKLIST = """
다음 자료를 바탕으로 '[수행평가 제목]' 수행평가에 대한 평가 기준(체크리스트)을 생성해주세요.
이 수행평가는 [과목] 과목의 [수행평가 유형] 유형의 수행평가 입니다.
수행평가에 대한 설명은 다음과 같습니다:
[수행평가 설명]
"""

# Schema definitions
CHECKLIST_SCHEMA = {
    "type": "object",
    "properties": {
        "checklist": {
            "type": "array",
            "description": "체크리스트 대분류 항목들",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "대분류 제목"},
                    "subcategories": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "소분류 제목"},
                                "description": {"type": "string", "description": "소분류 설명"},
                                "levels": {
                                    "type": "object",
                                    "properties": {
                                        "high": {"type": "string", "description": "상 수준 설명"},
                                        "medium": {"type": "string", "description": "중 수준 설명"},
                                        "low": {"type": "string", "description": "하 수준 설명"}
                                    },
                                    "required": ["high", "medium", "low"]
                                }
                            },
                            "required": ["name", "description", "levels"]
                        }
                    }
                },
                "required": ["title", "subcategories"]
            }
        }
    },
    "required": ["checklist"]
}

EVALUATION_SCHEMA = {
    "type": "object",
    "properties": {
        "evaluation": {
            "type": "array",
            "description": "평가 항목별 점수와 이유",
            "items": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "평가 대분류"
                    },
                    "subcategories": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "평가 소분류"
                                },
                                "score": {
                                    "type": "integer",
                                    "description": "0~3 사이의 점수"
                                },
                                "reason": {
                                    "type": "string",
                                    "description": "점수 평가 이유"
                                },
                                "evidence":{
                                    "type": "array",
                                    "description": "관련 있는 텍스트를 증거로 제시",
                                    "items": {
                                        "type": "string",
                                        "description": "증거 텍스트"
                                    }
                                }
                            },
                            "required": ["name", "score", "reason", "evidence"]
                        }
                    }
                },
                "required": ["category", "subcategories"]
            }
        },
        "overall_feedback": {
            "type": "string",
            "description": "전체적인 피드백"
        }
    },
    "required": ["evaluation", "overall_feedback"]
}

class JupyterWidgetHandler(logging.Handler):
    """Custom logging handler for Jupyter widgets"""
    def __init__(self, output_widget):
        super().__init__()
        self.output_widget = output_widget
    
    def emit(self, record):
        msg = self.format(record)
        with self.output_widget:
            print(msg)

class BaseWidgetManager:
    """Base class for widget managers with common functionality"""
    
    def __init__(self):
        """Initialize the base widget manager"""
        # Create directories if they don't exist
        for directory in ['./checklists', './evaluations', './temp']:
            os.makedirs(directory, exist_ok=True)
        
        # Setup logging
        self.log_output = widgets.Output(
            layout=widgets.Layout(height='150px', overflow='auto', border='1px solid gray')
        )
        self.log_output.add_class('scrollable')
        self.setup_logging()
        
        # Error and output widgets
        self.output_area = widgets.Output()
        self.error_area = widgets.Output()
        
        # Initialize component references
        self.input_component = None
        self.template_component = None
        self.model_component = None
        self.output_component = None
        
        # Initialize components dictionary
        self.components = {}
        
        # Initialize API client
        self.api_client = None
        
        # Track temporary files for cleanup
        self.temp_files = []
        
    def setup_logging(self):
        """Set up the logger with custom widget handler"""
        # Remove all handlers associated with the root logger object
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Set up the logger to use the Jupyter widget handler
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        widget_handler = JupyterWidgetHandler(self.log_output)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        widget_handler.setFormatter(formatter)
        self.logger.addHandler(widget_handler)
    
    def initialize_api_client(self):
        """Initialize the LLM API client with logger"""
        if not self.api_client:
            self.api_client = LLMAPIClient(log_level=logging.INFO)
            self.api_client.logger = self.logger
        return self.api_client
    
    def add_component(self, name, component):
        """Add a component to this widget manager"""
        self.components[name] = component
        setattr(self, f"{name}_component", component)
    
    def get_component(self, name):
        """Get a component by name"""
        return self.components.get(name)
        
    def display_all(self):
        """Display all widgets (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement display_all method")
    
    def display_error(self, message):
        """Display an error message in the error area"""
        with self.error_area:
            self.error_area.clear_output()
            print(f"오류: {message}")
    
    def display_log(self, message):
        """Log a message to the log output"""
        self.logger.info(message)
    
    def add_temp_file(self, file_path):
        """Track a temporary file for later cleanup"""
        if file_path and os.path.exists(file_path):
            self.temp_files.append(file_path)
            
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    self.logger.info(f"임시 파일 삭제: {file_path}")
            except Exception as e:
                self.logger.error(f"임시 파일 삭제 중 오류: {str(e)}")
        
        # Clear the list of temp files
        self.temp_files = []
        
    def __del__(self):
        """Destructor to clean up resources"""
        self.cleanup_temp_files()
    
    def process_with_models(self, prompt, system_instruction, schema, pdf_path=None, completion_message=None):
        """Process request with selected models and return results
        
        Args:
            prompt: The prompt to send to the models
            system_instruction: The system instruction to send to the models
            schema: The JSON schema for structured output
            pdf_path: Optional path to a PDF file to process
            completion_message: Optional custom completion message to display
            
        Returns:
            True if processing completed, False otherwise
        """
        # Initialize API client
        client = self.initialize_api_client()
        
        # Process with selected models
        models = self.model_component.get_selected_models()
        
        # Define the processing method mapping
        process_methods = {
            'Gemini': client.process_pdf_with_gemini,
            'Claude': client.process_pdf_with_claude,
            'OpenAI': client.process_pdf_with_openai
        }
        
        for model_name, model_params in models.items():
            if not model_params['model_name']:
                continue
                
            try:
                print(f"\n{model_name}로 처리 중...")
                
                response = process_methods[model_name](
                    file_path=pdf_path,
                    prompt=prompt,
                    model_name=model_params['model_name'],
                    schema=schema,
                    system_instruction=system_instruction,
                    temperature=model_params['temperature'],
                    max_tokens=model_params['max_tokens']
                )
                
                # Extract response text based on model type
                if model_name == 'Gemini':
                    response_text = response.text
                elif model_name == 'Claude':
                    response_text = response.content[0].text
                    # Extract JSON content if wrapped in markdown code block
                    if "```json" in response_text:
                        response_text = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL).group(1)
                else:  # OpenAI
                    response_text = response.choices[0].message.content
                
                # Store result in output component
                self.output_component.store_result(model_name, response_text)
                
                if completion_message:
                    print(f"{model_name} {completion_message}")
                else:
                    print(f"{model_name} 처리 완료")
                
                # Enable save button
                self.model_component.enable_save_button(model_name)
                
            except Exception as e:
                with self.error_area:
                    print(f"{model_name} 처리 중 오류: {str(e)}")
        
        # Display token usage summary
        usage_summary = client.get_token_usage_summary()
        self.output_component.display_token_usage(usage_summary)
        
        return True

class BaseComponent:
    """Base class for widget components"""
    
    def __init__(self, manager):
        """Initialize the component with a reference to its manager"""
        self.manager = manager
        
    def get_widgets(self):
        """Return a list of all widgets in this component"""
        raise NotImplementedError("Subclasses must implement get_widgets method")
        
    def create_layout(self):
        """Create and return a layout widget containing all widgets in this component"""
        raise NotImplementedError("Subclasses must implement create_layout method")

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
                options=['고등학교 1학년', '고등학교 2학년', '고등학교 3학년'],
                value='고등학교 1학년',
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
            options=['찬성반대', '독서와 작문', '나의 위인전'],
            value='찬성반대',
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
            'subject': self.subject_widget.value,
            'title': self.title_widget.value,
            'assessment_type': self.assessment_type_widget.value,
            'description': self.description_widget.value,
        }
        
        if self.include_grade:
            values['grade'] = self.grade_widget.value
            
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

class TemplateWidgetsComponent(BaseComponent):
    """Component for template editing widgets"""
    
    def __init__(self, manager, system_template, prompt_template):
        """Initialize template widgets component"""
        super().__init__(manager)
        self.system_template = system_template
        self.prompt_template = prompt_template
        self.create_widgets()
    
    def create_widgets(self):
        """Create template editing widgets"""
        self.system_instruction_widget = widgets.Textarea(
            description="시스템 지시:",
            value=self.system_template,
            layout=widgets.Layout(width='90%', height='300px')
        )
        
        self.prompt_widget = widgets.Textarea(
            description="프롬프트:",
            value=self.prompt_template,
            layout=widgets.Layout(width='90%', height='200px')
        )

        self.update_templates_button = widgets.Button(
            description="인풋 적용",
            button_style='info',
            tooltip='입력된 정보로 템플릿을 업데이트합니다.'
        )
        self.update_templates_button.on_click(self.update_templates)
    
    def get_widgets(self):
        """Return all widgets in this component"""
        return [
            self.system_instruction_widget,
            self.prompt_widget,
            self.update_templates_button
        ]
    
    def create_layout(self):
        """Create a layout for the template widgets"""
        return widgets.VBox([
            widgets.HTML("<h3>템플릿 편집</h3>"),
            self.system_instruction_widget,
            self.prompt_widget,
            self.update_templates_button
        ])
    
    def format_template(self, template, values):
        """Format a template with the given values
        
        Args:
            template: The template string with [placeholders]
            values: Dictionary of placeholder values
            
        Returns:
            Formatted template with placeholders replaced
        """
        formatted = template
        replacements = {
            '학년': values.get('grade', ''),
            '과목': values.get('subject', ''),
            '수행평가 제목': values.get('title', ''),
            '수행평가 유형': values.get('assessment_type', ''),
            '수행평가 설명': values.get('description', '')
        }
        
        for placeholder, value in replacements.items():
            formatted = formatted.replace(f"[{placeholder}]", value)
            
        return formatted
    
    def update_templates(self, b=None):
        """Update templates when the update button is clicked"""
        # Check if input component exists and has valid values
        input_component = self.manager.get_component('input')
        if not input_component or not input_component.validate_inputs():
            return
            
        # Get input values
        values = input_component.get_values()
        
        # Format templates using the helper method
        self.system_instruction_widget.value = self.format_template(self.system_template, values)
        self.prompt_widget.value = self.format_template(self.prompt_template, values)
        
        print("템플릿이 업데이트되었습니다.")
    
    def get_formatted_system_instruction(self, replacements=None):
        """Get the current system instruction with optional additional replacements"""
        system_instruction = self.system_instruction_widget.value
        
        if replacements:
            for key, value in replacements.items():
                system_instruction = system_instruction.replace(f"[{key}]", str(value))
                
        return system_instruction
        
    def get_formatted_prompt(self, replacements=None):
        """Get the current prompt with optional additional replacements"""
        prompt = self.prompt_widget.value
        
        if replacements:
            for key, value in replacements.items():
                prompt = prompt.replace(f"[{key}]", str(value))
                
        return prompt

class ModelSelectionComponent(BaseComponent):
    """Component for model selection widgets"""
    
    def __init__(self, manager):
        """Initialize model selection component"""
        super().__init__(manager)
        self.create_widgets()
    
    def create_widgets(self):
        """Create model selection widgets"""
        # Model selection dropdowns
        self.gemini_model_selection = widgets.Dropdown(
            options=['gemini-2.5-flash-preview-04-17', 'gemini-2.0-flash', 'gemini-2.0-flash-lite', 'None'],
            value='gemini-2.0-flash',
            description='Gemini 모델:'
        )
        
        self.claude_model_selection = widgets.Dropdown(
            options=['claude-3-7-sonnet-20250219', 'claude-3-5-haiku-20241022', 'claude-3-opus-20240229', 'None'],
            value='claude-3-7-sonnet-20250219',
            description='Claude 모델:'
        )
        
        self.openai_model_selection = widgets.Dropdown(
            options=['gpt-4.1', 'gpt-4.1-mini', 'o4-mini', 'None'],
            value='gpt-4.1',
            description='OpenAI 모델:'
        )
        
        # Create model parameters
        self.gemini_params = self.create_model_params_widgets('Gemini')
        self.claude_params = self.create_model_params_widgets('Claude')
        self.openai_params = self.create_model_params_widgets('OpenAI')
        
        # Create save buttons
        self.save_buttons = {
            'Gemini': widgets.Button(description="Gemini 결과 저장", disabled=True),
            'Claude': widgets.Button(description="Claude 결과 저장", disabled=True),
            'OpenAI': widgets.Button(description="OpenAI 결과 저장", disabled=True)
        }
        
        # Create action button - will be overridden by concrete classes
        self.action_button = widgets.Button(description="실행")
    
    def create_model_params_widgets(self, model_name):
        """Create parameter widgets for a specific model"""
        return {
            'temperature': widgets.FloatSlider(value=0.10, min=0, max=1, step=0.01, description='Temperature:'),
            'max_tokens': widgets.IntText(value=4096, description='Max Tokens:')
        }
    
    def get_widgets(self):
        """Return all widgets in this component"""
        widgets_list = [
            self.action_button,
            self.gemini_model_selection, 
            self.gemini_params['temperature'], 
            self.gemini_params['max_tokens'],
            self.save_buttons['Gemini'],
            self.claude_model_selection, 
            self.claude_params['temperature'], 
            self.claude_params['max_tokens'],
            self.save_buttons['Claude'],
            self.openai_model_selection, 
            self.openai_params['temperature'], 
            self.openai_params['max_tokens'],
            self.save_buttons['OpenAI']
        ]
        return widgets_list
    
    def create_layout(self):
        """Create a layout for the model selection widgets"""
        return widgets.VBox([
            self.action_button,
            widgets.HBox([
                widgets.VBox([self.gemini_model_selection, self.gemini_params['temperature'], 
                             self.gemini_params['max_tokens'], self.save_buttons['Gemini']]),
                widgets.VBox([self.claude_model_selection, self.claude_params['temperature'], 
                             self.claude_params['max_tokens'], self.save_buttons['Claude']]),
                widgets.VBox([self.openai_model_selection, self.openai_params['temperature'], 
                             self.openai_params['max_tokens'], self.save_buttons['OpenAI']])
            ])
        ])
    
    def get_selected_models(self):
        """Get a dictionary of selected models and their parameters"""
        return {
            'Gemini': {
                'model_name': self.gemini_model_selection.value if self.gemini_model_selection.value != "None" else None,
                'temperature': self.gemini_params['temperature'].value,
                'max_tokens': self.gemini_params['max_tokens'].value
            },
            'Claude': {
                'model_name': self.claude_model_selection.value if self.claude_model_selection.value != "None" else None,
                'temperature': self.claude_params['temperature'].value,
                'max_tokens': self.claude_params['max_tokens'].value
            },
            'OpenAI': {
                'model_name': self.openai_model_selection.value if self.openai_model_selection.value != "None" else None,
                'temperature': self.openai_params['temperature'].value,
                'max_tokens': self.openai_params['max_tokens'].value
            }
        }
        
    def set_save_handler(self, model, handler):
        """Set the save handler for a specific model's save button"""
        self.save_buttons[model].on_click(handler)
    
    def enable_save_button(self, model):
        """Enable the save button for a specific model"""
        self.save_buttons[model].disabled = False
        
    def disable_save_button(self, model):
        """Disable the save button for a specific model"""
        self.save_buttons[model].disabled = True
        
    def set_action_handler(self, handler):
        """Set the handler for the action button"""
        self.action_button.on_click(handler)
        
    def set_action_button_text(self, text):
        """Set the text for the action button"""
        self.action_button.description = text

class OutputComponent(BaseComponent):
    """Component for output display and tokens usage"""
    
    def __init__(self, manager):
        """Initialize output component"""
        super().__init__(manager)
        self.create_widgets()
        self.results = {}
    
    def create_widgets(self):
        """Create output widgets"""
        self.visualization_output = widgets.Output()
        self.token_usage_output = widgets.Output()
        self.visualize_button = widgets.Button(description="결과 출력")
        self.visualize_button.on_click(self.display_results)
    
    def get_widgets(self):
        """Return all widgets in this component"""
        return [
            self.visualize_button,
            self.visualization_output,
            self.token_usage_output
        ]
    
    def create_layout(self):
        """Create a layout for the output widgets"""
        return widgets.VBox([
            self.visualize_button,
            self.visualization_output,
        ])
    
    def store_result(self, model_name, result):
        """Store a result for later display"""
        self.results[model_name] = result
    
    def get_results(self):
        """Get all stored results"""
        return self.results
    
    def clear_results(self):
        """Clear all stored results"""
        self.results = {}
    
    def clear_outputs(self):
        """Clear all output displays"""
        with self.visualization_output:
            self.visualization_output.clear_output()
        with self.token_usage_output:
            self.token_usage_output.clear_output()
    
    def display_results(self, b=None):
        """Display stored results (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement display_results method")
    
    def display_token_usage(self, usage_summary):
        """Display token usage as an HTML table"""
        html = "<table border='1' style='border-collapse: collapse; margin: 10px;'>"
        
        # Table headers
        headers = ['Model', 'Num Requests', 'Input Tokens', 'Output Tokens', 'Total Tokens', 'Estimated Cost']
        html += "<tr>" + "".join(f"<th>{header}</th>" for header in headers) + "</tr>"
        
        # Data for each model
        for model, stats in usage_summary['models'].items():
            html += "<tr>"
            html += f"<td>{model}</td>"
            html += f"<td>{stats.get('requests', 0)}</td>"
            html += f"<td>{stats.get('prompt_tokens', 0)}</td>"
            html += f"<td>{stats.get('completion_tokens', 0)}</td>"
            html += f"<td>{stats.get('total_tokens', 0)}</td>"
            html += f"<td>${stats.get('estimated_cost', 0):.6f}</td>"
            html += "</tr>"

        # Total usage row
        html += "<tr><th>Total</th>"
        html += f"<td>{usage_summary['total_requests']}</td>"
        html += f"<td>{usage_summary['total_prompt_tokens']}</td>"
        html += f"<td>{usage_summary['total_completion_tokens']}</td>"
        html += f"<td>{usage_summary['total_tokens']}</td>"
        html += f"<td>${usage_summary['total_estimated_cost']:.6f}</td>"
        html += "</tr>"
        html += "</table>"
        
        # Update token usage output widget
        with self.token_usage_output:
            self.token_usage_output.clear_output()
            display(HTML(html))

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
            description="PDF 업로드 (API용):",
            accept=".pdf",
            layout=widgets.Layout(width='300px')
        )
        
        # Add file upload handler
        self.pdf_upload.observe(self.process_uploaded_pdf, names='value')
    
    def get_widgets(self):
        """Return all widgets in this component"""
        return [
            widgets.HTML("<h4>PDF 업로드 (API 요청용)</h4>"),
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

class ChecklistComponent(BaseComponent):
    """Component for checklist selection and management"""
    
    def __init__(self, manager):
        """Initialize checklist component"""
        super().__init__(manager)
        self.create_widgets()
    
    def create_widgets(self):
        """Create checklist widgets"""
        # Get checklist files
        self.checklist_files = [f for f in os.listdir('./checklists') if f.endswith('.json')]
        
        self.select_checklist_widget = widgets.Dropdown(
            options=[checklist_name.replace('.json', '') for checklist_name in self.checklist_files] if self.checklist_files else ['체크리스트 없음'],
            description='체크리스트:',
            layout=widgets.Layout(width='60%')
        )
    
    def get_widgets(self):
        """Return all widgets in this component"""
        return [self.select_checklist_widget]
    
    def create_layout(self):
        """Create a layout for the checklist widgets"""
        return widgets.VBox(self.get_widgets())
    
    def refresh_checklists(self):
        """Refresh the list of available checklists"""
        self.checklist_files = [f for f in os.listdir('./checklists') if f.endswith('.json')]
        self.select_checklist_widget.options = [checklist_name.replace('.json', '') for checklist_name in self.checklist_files] if self.checklist_files else ['체크리스트 없음']
    
    def get_selected_checklist(self):
        """Get the currently selected checklist"""
        return self.select_checklist_widget.value
    
    def load_checklist(self, file_path):
        """Load a checklist from a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            self.manager.logger.error(f"체크리스트 파일을 찾을 수 없습니다: {file_path}")
            return {}
        except json.JSONDecodeError:
            self.manager.logger.error(f"체크리스트 파일이 유효한 JSON 형식이 아닙니다: {file_path}")
            return {} 

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
            print("\n체크리스트 생성 중...")
            
            # Get PDF path if uploaded
            pdf_path = self.pdf_upload_component.get_pdf_path()
            
            # Use common model processing method
            self.process_with_models(
                prompt=prompt,
                system_instruction=system_instruction,
                schema=CHECKLIST_SCHEMA,
                pdf_path=pdf_path,
                completion_message="체크리스트 생성 완료"
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

class ChecklistOutputComponent(OutputComponent):
    """Component for checklist output display"""
    
    def display_results(self, b=None):
        """Display checklist results"""
        with self.visualization_output:
            self.visualization_output.clear_output()
            
            results = self.get_results()
            if not results:
                print("표시할 체크리스트가 없습니다.")
                return
            
            for model_name, result_text in results.items():
                try:
                    # Parse JSON result
                    result = json.loads(result_text)
                    
                    # Display formatted result
                    html_output = f"<h2>{model_name} 체크리스트</h2>"
                    
                    checklist = result.get('checklist', [])
                    
                    for category in checklist:
                        title = category.get('title', '')
                        html_output += f"<h3>{title}</h3>"
                        
                        subcategories = category.get('subcategories', [])
                        html_output += "<table border='1'>"
                        html_output += "<style>td {text-align: left;}</style>"
                        html_output += "<tr><th>평가 기준</th><th>설명</th><th>상</th><th>중</th><th>하</th></tr>"
                        
                        for subcategory in subcategories:
                            html_output += "<tr>"
                            html_output += f"<td>{subcategory.get('name', '')}</td>"
                            html_output += f"<td>{subcategory.get('description', '')}</td>"
                            
                            levels = subcategory.get('levels', {})
                            html_output += f"<td>{levels.get('high', '')}</td>"
                            html_output += f"<td>{levels.get('medium', '')}</td>"
                            html_output += f"<td>{levels.get('low', '')}</td>"
                            html_output += "</tr>"
                        
                        html_output += "</table><br>"
                    
                    # Display the HTML output
                    display(HTML(html_output))
                    
                except json.JSONDecodeError:
                    print(f"{model_name} 결과를 파싱할 수 없습니다. 유효한 JSON 형식이 아닙니다.")
                    print(f"Response: {result_text[:500]}...")
                except Exception as e:
                    print(f"{model_name} 체크리스트 출력 중 오류: {str(e)}")
    
    def create_save_handler(self, model):
        """Create a closure for saving checklist results"""
        def on_click(b):
            results = self.get_results()
            if model not in results:
                print(f"{model} 체크리스트가 없습니다.")
                return
                
            try:
                # Get input component
                input_component = self.manager.get_component('input')
                
                # Get title from input component
                title = input_component.title_widget.value if input_component else "untitled"
                
                # Create a safe filename with timestamp
                safe_filename = sanitize_filename(title)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # Get model name from model component
                model_component = self.manager.get_component('model')
                model_used = None
                if model_component:
                    models = model_component.get_selected_models()
                    model_used = models[model]['model_name']
                
                # Fallback if model name is not available
                if not model_used:
                    model_used = model.lower()
                
                # Create output directory if it doesn't exist
                os.makedirs('./checklists', exist_ok=True)
                
                # Create filename
                result_file = f"./checklists/{model_used}_평가기준_{safe_filename}_{timestamp}.json"
                
                # Check if directory is writable
                if not os.access('./checklists', os.W_OK):
                    raise PermissionError(f"체크리스트 디렉토리에 쓰기 권한이 없습니다: './checklists'")
                
                # Save to file
                with open(result_file, 'w', encoding='utf-8') as f:
                    f.write(results[model])
                
                print(f"{model} 체크리스트가 {result_file}에 저장되었습니다.")
                
                # Refresh the checklist component if it exists
                checklist_component = self.manager.get_component('checklist')
                if checklist_component:
                    checklist_component.refresh_checklists()
                
            except PermissionError as e:
                print(f"권한 오류: {str(e)}")
            except IOError as e:
                print(f"파일 저장 오류: {str(e)}")
            except Exception as e:
                print(f"{model} 체크리스트 저장 중 오류: {str(e)}")
        
        return on_click

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
            print("\n평가 중...")
            
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
            widgets.HTML("<h3>인풋 입력</h3>"),
            self.input_component.create_layout(),
            self.checklist_component.create_layout(),
            widgets.HTML("<h3>학생 제출물</h3>"),
            self.student_submission_component.create_layout(),
            self.pdf_upload_component.create_layout(),
            self.output_area, self.error_area
        ])
        
        # Tab 1: 템플릿 편집
        templates = self.template_component.create_layout()
        
        # Tab 2: 모델 설정
        model_settings = self.model_component.create_layout()
        
        # Tab 3: 결과 보기
        results = self.output_component.create_layout()
        
        # Tab 4: 로그 확인
        logs = widgets.VBox([
            widgets.HTML("<h3>시스템 로그</h3>"),
            widgets.HTML("<h3>토큰 사용량</h3>"),
            self.output_component.token_usage_output,
            self.log_output
        ])
        
        # Set tab contents
        tab_content.children = [content_input, templates, model_settings, results, logs]
        
        # Set tab titles
        tab_content.set_title(0, "인풋 입력")
        tab_content.set_title(1, "템플릿 편집")
        tab_content.set_title(2, "모델 설정")
        tab_content.set_title(3, "결과 보기")
        tab_content.set_title(4, "로그 확인")
        
        # Display the tab widget
        display(tab_content)

class EvaluationOutputComponent(OutputComponent):
    """Component for evaluation output display"""
    
    def display_results(self, b=None):
        """Display evaluation results"""
        with self.visualization_output:
            self.visualization_output.clear_output()
            
            results = self.get_results()
            if not results:
                print("표시할 평가 결과가 없습니다.")
                return
            
            for model_name, result_text in results.items():
                try:
                    # Parse JSON result
                    result = json.loads(result_text)
                    
                    # Format result as HTML
                    html_output = f"<h2>{model_name} 평가 결과 요약</h2>"
                    
                    # Overall feedback
                    html_output += f"<h3>전체 피드백</h3>"
                    html_output += f"<p>{result.get('overall_feedback', '피드백 없음')}</p>"
                    
                    # Detailed evaluation by category
                    html_output += f"<h3>세부 평가</h3>"
                    html_output += "<table border='1'>"
                    html_output += "<style>td {text-align: left;}</style>"
                    html_output += "<tr><th>카테고리</th><th>항목</th><th>점수</th><th>이유</th><th>근거</th></tr>"
                    
                    for category in result.get('evaluation', []):
                        category_name = category.get('category', '')
                        subcategories = category.get('subcategories', [])
                        
                        for i, subcategory in enumerate(subcategories):
                            html_output += "<tr>"
                            if i == 0:  # Show category name only for first subcategory
                                html_output += f"<td rowspan='{len(subcategories)}'>{category_name}</td>"
                            html_output += f"<td>{subcategory.get('name', '')}</td>"
                            html_output += f"<td>{subcategory.get('score', '')}</td>"
                            html_output += f"<td>{subcategory.get('reason', '')}</td>"
                            
                            # Properly format evidence which can be an array
                            evidence = subcategory.get('evidence', [])
                            if isinstance(evidence, list):
                                evidence_text = "<ul>" + "".join([f"<li>{item}</li>" for item in evidence]) + "</ul>"
                            else:
                                evidence_text = str(evidence)
                            html_output += f"<td>{evidence_text}</td>"
                            
                            html_output += "</tr>"
                    
                    html_output += "</table>"
                    
                    # Display HTML
                    display(HTML(html_output))
                    
                except json.JSONDecodeError:
                    print(f"{model_name} 결과를 파싱할 수 없습니다. 유효한 JSON 형식이 아닙니다.")
                    print("Response text:")
                    print(result_text[:500] + "..." if len(result_text) > 500 else result_text)
                except Exception as e:
                    print(f"{model_name} 평가 결과 출력 중 오류: {str(e)}")
    
    def create_save_handler(self, model):
        """Create a closure for saving evaluation results"""
        def on_click(b):
            results = self.get_results()
            if model not in results:
                print(f"{model} 평가 결과가 없습니다.")
                return
                
            try:
                # Get student submission component
                student_submission_component = self.manager.get_component('student_submission')
                
                # Get model component
                model_component = self.manager.get_component('model')
                model_used = None
                if model_component:
                    models = model_component.get_selected_models()
                    model_used = models[model]['model_name']
                
                # Fallback if model name is not available
                if not model_used:
                    model_used = model.lower()
                
                # Create output directory if it doesn't exist
                os.makedirs('./evaluations', exist_ok=True)
                
                # Get file name from submission component
                file_name = student_submission_component.get_file_name() if student_submission_component else "unnamed"
                
                # Use a safe filename with timestamp
                safe_filename = sanitize_filename(file_name)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                result_file = f"./evaluations/{model_used}_평가결과_{safe_filename}_{timestamp}.json"
                
                # Check if directory is writable
                if not os.access('./evaluations', os.W_OK):
                    raise PermissionError(f"평가 결과 디렉토리에 쓰기 권한이 없습니다: './evaluations'")
                
                # Save result to file
                with open(result_file, 'w', encoding='utf-8') as f:
                    f.write(results[model])
                
                print(f"{model} 평가 결과가 {result_file}에 저장되었습니다.")
                
            except PermissionError as e:
                print(f"권한 오류: {str(e)}")
            except IOError as e:
                print(f"파일 저장 오류: {str(e)}")
            except Exception as e:
                print(f"{model} 평가 결과 저장 중 오류: {str(e)}")
        
        return on_click

def sanitize_filename(name, fallback="untitled"):
    """Sanitize a string to be used as a filename
    
    Args:
        name: The string to sanitize
        fallback: Fallback name if the sanitized string is empty
        
    Returns:
        A sanitized filename
    """
    if not name:
        return fallback
        
    # Replace problematic characters with underscores
    # More comprehensive than just removing non-alphanumeric chars
    name = re.sub(r'[\\/*?:"<>|]', '_', name)
    
    # Replace multiple spaces with single space
    name = re.sub(r'\s+', ' ', name)
    
    # Trim spaces from beginning and end
    name = name.strip()
    
    # Ensure the name is not too long
    if len(name) > 100:
        name = name[:97] + '...'
        
    # If sanitized name is empty, use fallback
    return name if name else fallback


###########################
#############################
# LLM Call Manager

class LlmCallOutputComponent(widgets.Output):
    """Component for raw LLM API call output display"""
    
    def __init__(self, manager):
        """Initialize output component"""
        super().__init__()
        self.manager = manager
        self.create_widgets()
        self.results = {}
    
    def create_widgets(self):
        """Create output widgets"""
        self.visualization_output = widgets.Output()
        self.token_usage_output = widgets.Output()
        self.visualize_button = widgets.Button(description="결과 출력")
        self.visualize_button.on_click(self.display_results)
    
    def get_widgets(self):
        """Return all widgets in this component"""
        return [
            self.visualize_button,
            self.visualization_output,
            self.token_usage_output
        ]
    
    def create_layout(self):
        """Create a layout for the output widgets"""
        return widgets.VBox([
            self.visualize_button,
            self.visualization_output,
        ])
    
    def store_result(self, model_name, result):
        """Store a result for later display"""
        self.results[model_name] = result
    
    def get_results(self):
        """Get all stored results"""
        return self.results
    
    def clear_results(self):
        """Clear all stored results"""
        self.results = {}
    
    def clear_outputs(self):
        """Clear all output displays"""
        with self.visualization_output:
            self.visualization_output.clear_output()
        with self.token_usage_output:
            self.token_usage_output.clear_output()
    
    def display_results(self, b=None):
        """Display raw LLM results"""
        with self.visualization_output:
            self.visualization_output.clear_output()
            
            results = self.get_results()
            if not results:
                print("표시할 결과가 없습니다.")
                return
            
            for model_name, result_text in results.items():
                html_output = f"<h2>{model_name} 결과</h2>"
                html_output += f"<pre style='white-space: pre-wrap; word-wrap: break-word; background-color: #f5f5f5; padding: 10px;'>{result_text}</pre>"
                display(HTML(html_output))
    
    def display_token_usage(self, usage_summary):
        """Display token usage as an HTML table"""
        html = "<table border='1' style='border-collapse: collapse; margin: 10px;'>"
        
        # Table headers
        headers = ['Model', 'Num Requests', 'Input Tokens', 'Output Tokens', 'Total Tokens', 'Estimated Cost']
        html += "<tr>" + "".join(f"<th>{header}</th>" for header in headers) + "</tr>"
        
        # Data for each model
        for model, stats in usage_summary['models'].items():
            html += "<tr>"
            html += f"<td>{model}</td>"
            html += f"<td>{stats.get('requests', 0)}</td>"
            html += f"<td>{stats.get('prompt_tokens', 0)}</td>"
            html += f"<td>{stats.get('completion_tokens', 0)}</td>"
            html += f"<td>{stats.get('total_tokens', 0)}</td>"
            html += f"<td>${stats.get('estimated_cost', 0):.6f}</td>"
            html += "</tr>"

        # Total usage row
        html += "<tr><th>Total</th>"
        html += f"<td>{usage_summary['total_requests']}</td>"
        html += f"<td>{usage_summary['total_prompt_tokens']}</td>"
        html += f"<td>{usage_summary['total_completion_tokens']}</td>"
        html += f"<td>{usage_summary['total_tokens']}</td>"
        html += f"<td>${usage_summary['total_estimated_cost']:.6f}</td>"
        html += "</tr>"
        html += "</table>"
        
        # Update token usage output widget
        with self.token_usage_output:
            self.token_usage_output.clear_output()
            display(HTML(html))
    
    def create_save_handler(self, model):
        """Create a handler for saving results"""
        def on_click(b):
            results = self.get_results()
            if model not in results:
                print(f"{model} 결과가 없습니다.")
                return
                
            try:
                # Get input component for title information
                input_component = self.manager.get_component('input')
                
                # Get title from input component, or use timestamp if not available
                title = input_component.title_widget.value if input_component else "call"
                
                # Create a safe filename with timestamp
                safe_filename = self.manager.sanitize_filename(title) if hasattr(self.manager, 'sanitize_filename') else title.replace(' ', '_')
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # Get model name from model component
                model_component = self.manager.get_component('model')
                model_used = None
                if model_component:
                    models = model_component.get_selected_models()
                    model_used = models[model]['model_name']
                
                # Fallback if model name is not available
                if not model_used:
                    model_used = model.lower()
                
                # Create output directory if it doesn't exist
                os.makedirs('./results', exist_ok=True)
                
                # Create filename
                result_file = f"./results/{model_used}_{safe_filename}_{timestamp}.txt"
                
                # Save to file
                with open(result_file, 'w', encoding='utf-8') as f:
                    f.write(results[model])
                
                print(f"{model} 결과가 {result_file}에 저장되었습니다.")
                
            except Exception as e:
                print(f"{model} 결과 저장 중 오류: {str(e)}")
        
        return on_click

class LlmCallManager(BaseWidgetManager):
    """Manager for direct LLM API calls without schema enforcement"""
    
    def __init__(self):
        """Initialize the LLM call manager"""
        super().__init__()
        
        # Create components with empty templates
        input_component = InputWidgetsComponent(self, include_grade=False)
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
        self.add_component('input', input_component)
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
            
            print("System Instruction:")
            print(system_instruction)
            print("\nPrompt:")
            print(prompt)
            print("\nLLM API 요청 중...")
            
            # Get PDF path if uploaded
            pdf_path = self.pdf_upload_component.get_pdf_path()
            
            # Use process_with_models but explicitly pass None for schema
            self.process_with_models_without_schema(
                prompt=prompt,
                system_instruction=system_instruction,
                pdf_path=pdf_path,
                completion_message="LLM API 요청 완료"
            )
    
    def process_with_models_without_schema(self, prompt, system_instruction, pdf_path=None, completion_message=None):
        """Process request with selected models without requiring a schema
        
        Overrides the schema parameter to None to accommodate models that don't require schema
        """
        # Initialize API client
        client = self.initialize_api_client()
        
        # Process with selected models
        models = self.model_component.get_selected_models()
        
        # Define the processing method mapping
        process_methods = {
            'Gemini': client.process_pdf_with_gemini,
            'Claude': client.process_pdf_with_claude,
            'OpenAI': client.process_pdf_with_openai
        }
        
        for model_name, model_params in models.items():
            if not model_params['model_name']:
                continue
                
            try:
                print(f"\n{model_name}로 처리 중...")
                
                response = process_methods[model_name](
                    file_path=pdf_path,
                    prompt=prompt,
                    model_name=model_params['model_name'],
                    schema=None,  # No schema enforcement
                    system_instruction=system_instruction,
                    temperature=model_params['temperature'],
                    max_tokens=model_params['max_tokens']
                )
                
                # Extract response text based on model type
                if model_name == 'Gemini':
                    response_text = response.text
                elif model_name == 'Claude':
                    response_text = response.content[0].text
                else:  # OpenAI
                    response_text = response.choices[0].message.content
                
                # Store result in output component
                self.output_component.store_result(model_name, response_text)
                
                if completion_message:
                    print(f"{model_name} {completion_message}")
                else:
                    print(f"{model_name} 처리 완료")
                
                # Enable save button
                self.model_component.enable_save_button(model_name)
                
            except Exception as e:
                with self.error_area:
                    print(f"{model_name} 처리 중 오류: {str(e)}")
        
        # Display token usage summary
        usage_summary = client.get_token_usage_summary()
        self.output_component.display_token_usage(usage_summary)
        
        return True
    
    def display_all(self):
        """Display all widgets for LLM API calls"""
        # Create tabs
        tab_content = widgets.Tab()
        
        # Tab 0: 템플릿 편집
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
        
    def sanitize_filename(self, name, fallback="untitled"):
        """Sanitize a string to be used as a filename"""
        if not name:
            return fallback
            
        # Replace problematic characters with underscores
        name = re.sub(r'[\\/*?:"<>|]', '_', name)
        
        # Replace multiple spaces with single space
        name = re.sub(r'\s+', ' ', name)
        
        # Trim spaces from beginning and end
        name = name.strip()
        
        # Ensure the name is not too long
        if len(name) > 100:
            name = name[:97] + '...'
            
        # If sanitized name is empty, use fallback
        return name if name else fallback