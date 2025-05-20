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
DEFAULT_SYSTEM_INSTRUCTION = """
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

DEFAULT_PROMPT = """
다음은 학생이 제출한 수행평가 과제이다. 수행평가 과제에 대한 세부정보와 평가 기준을 고려하여, 각 평가항목에 대하여 논리적으로 평가하라.

평가 기준은 다음과 같다:
[평가 기준]

학생의 수행평가 과제는 다음과 같다:
[학생 제출물]
"""

class JupyterWidgetHandler(logging.Handler):
    """Custom logging handler for Jupyter widgets"""
    def __init__(self, output_widget):
        super().__init__()
        self.output_widget = output_widget
    
    def emit(self, record):
        msg = self.format(record)
        with self.output_widget:
            print(msg)

class AssignmentEvaluationWidgets:
    """Class to handle all widgets for assignment evaluation"""
    
    def __init__(self):
        """Initialize all widgets and variables"""
        # Create directories if they don't exist
        for directory in ['./checklists', './evaluations', './temp']:
            os.makedirs(directory, exist_ok=True)
            
        # Load checklist files
        self.checklist_files = [f for f in os.listdir('./checklists') if f.endswith('.json')]
        
        # Setup logging
        self.log_output = widgets.Output(
            layout=widgets.Layout(height='150px', overflow='auto', border='1px solid gray')
        )
        self.log_output.add_class('scrollable')
        self.setup_logging()
        
        # Create all widgets
        self.create_info_widgets()
        self.create_submission_widgets()
        self.create_template_widgets()
        self.create_model_selection_widgets()
        self.create_output_widgets()
        
        # Initialize evaluation results storage
        self.evaluation_results = {}
        self.file_name = ""
        
        # Define evaluation schema
        self.define_schema()
        
        # Initialize PDF file path
        self.pdf_file_path = None
        
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
    
    def create_info_widgets(self):
        """Create input widgets for assessment information"""
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
        
        self.select_checklist_widget = widgets.Dropdown(
            options=[checklist_name.replace('.json', '') for checklist_name in self.checklist_files] if self.checklist_files else ['체크리스트 없음'],
            description='체크리스트:',
            layout=widgets.Layout(width='60%')
        )
        
        self.description_widget = widgets.Textarea(
            description="설명:", 
            placeholder='수행평가에 대한 설명을 입력하세요',
            layout=widgets.Layout(width='60%', height='80px')
        )
    
    def create_submission_widgets(self):
        """Create widgets for student submission"""
        # File upload widget
        self.file_upload = widgets.FileUpload(description="제출물 업로드:")
        
        # PDF upload widget for API requests
        self.pdf_upload = widgets.FileUpload(
            description="PDF 업로드 (API용):",
            accept=".pdf",
            layout=widgets.Layout(width='300px')
        )
        
        # Submission text area
        self.submission_text = widgets.Textarea(
            description="제출물 내용:",
            placeholder='학생 제출물을 직접 입력하거나, 위에서 파일을 업로드하세요.',
            layout=widgets.Layout(width='80%', height='200px')
        )
        
        # Add file upload handlers
        self.file_upload.observe(self.process_uploaded_file, names='value')
        self.pdf_upload.observe(self.process_uploaded_pdf, names='value')
    
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
    
    def process_uploaded_pdf(self, change):
        """Process uploaded PDF file for API requests"""
        if not self.pdf_upload.value:
            return
            
        try:
            # Get uploaded file content
            uploaded_file = list(self.pdf_upload.value.values())[0]
            
            # Create a temporary file path
            temp_path = f"./temp/temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file['content'])
            
            self.pdf_file_path = temp_path
            self.logger.info(f"PDF uploaded and saved to {temp_path}")
        except Exception as e:
            self.logger.error(f"PDF 업로드 중 오류: {str(e)}")
    
    def create_template_widgets(self):
        """Create widgets for system instruction and prompt templates"""
        # Store template texts
        self.system_instruction_template = DEFAULT_SYSTEM_INSTRUCTION
        self.prompt_template = DEFAULT_PROMPT
        
        # Create text areas for editing
        self.system_instruction_widget = widgets.Textarea(
            description="시스템 지시:",
            value=self.system_instruction_template,
            layout=widgets.Layout(width='90%', height='300px')
        )
        
        self.prompt_widget = widgets.Textarea(
            description="프롬프트:",
            value=self.prompt_template,
            layout=widgets.Layout(width='90%', height='200px')
        )

        # Create update button
        self.update_templates_button = widgets.Button(
            description="인풋 적용",
            button_style='info',
            tooltip='입력된 정보로 템플릿을 업데이트합니다.'
        )
        self.update_templates_button.on_click(self.update_templates)
    
    def update_templates(self, b=None):
        """Update templates when the update button is clicked"""
        # Check if all required values are set
        if not all([self.grade_widget.value, self.subject_widget.value, 
                   self.assessment_type_widget.value, self.title_widget.value, 
                   self.description_widget.value]):
            print("모든 입력 필드를 채워주세요.")
            return
            
        # Get current values
        grade = self.grade_widget.value
        subject = self.subject_widget.value
        assessment_type = self.assessment_type_widget.value
        assessment_title = self.title_widget.value
        assessment_description = self.description_widget.value
        
        # Format system instruction
        formatted_system = self.system_instruction_template
        formatted_system = formatted_system.replace('[학년]', grade)
        formatted_system = formatted_system.replace('[과목]', subject)
        formatted_system = formatted_system.replace('[수행평가 제목]', assessment_title)
        formatted_system = formatted_system.replace('[수행평가 유형]', assessment_type)
        formatted_system = formatted_system.replace('[수행평가 설명]', assessment_description)
        
        # Format prompt
        formatted_prompt = self.prompt_template
        formatted_prompt = formatted_prompt.replace('[학년]', grade)
        formatted_prompt = formatted_prompt.replace('[과목]', subject)
        formatted_prompt = formatted_prompt.replace('[수행평가 제목]', assessment_title)
        formatted_prompt = formatted_prompt.replace('[수행평가 유형]', assessment_type)
        formatted_prompt = formatted_prompt.replace('[수행평가 설명]', assessment_description)
        
        # Update the widgets with formatted text
        self.system_instruction_widget.value = formatted_system
        self.prompt_widget.value = formatted_prompt
        print("템플릿이 업데이트되었습니다.")
    
    def create_model_selection_widgets(self):
        """Create widgets for LLM model selection and parameters"""
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
        
        # Create action buttons
        self.evaluate_button = widgets.Button(description="평가 시작")
        self.visualize_button = widgets.Button(description="평가 결과 출력")
        
        # Connect button events
        self.evaluate_button.on_click(self.run_evaluation)
        self.visualize_button.on_click(self.run_visualization)
        
        for model, button in self.save_buttons.items():
            button.on_click(self.save_result(model))
    
    def create_model_params_widgets(self, model_name):
        """Create parameter widgets for a specific model"""
        return {
            'temperature': widgets.FloatSlider(value=0.10, min=0, max=1, step=0.01, description='Temperature:'),
            'max_tokens': widgets.IntText(value=4096, description='Max Tokens:')
        }
    
    def create_output_widgets(self):
        """Create output widgets for displaying results"""
        self.output_area = widgets.Output()
        self.error_area = widgets.Output()
        self.visualization_output = widgets.Output()
        self.token_usage_output = widgets.Output()
    
    def define_schema(self):
        """Define the evaluation schema"""
        self.custom_schema = {
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
    
    def load_checklist(self, file_path):
        """Load checklist from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f"체크리스트 파일을 찾을 수 없습니다: {file_path}")
            return {}
        except json.JSONDecodeError:
            self.logger.error(f"체크리스트 파일이 유효한 JSON 형식이 아닙니다: {file_path}")
            return {}
    
    def run_evaluation(self, b):
        """Execute the evaluation process with selected models"""
        with self.output_area:
            self.output_area.clear_output()
            self.error_area.clear_output()
            
            # Validate input data
            if not self.select_checklist_widget.value or self.select_checklist_widget.value == '체크리스트 없음':
                with self.error_area:
                    print("오류: 체크리스트를 선택해주세요.")
                return
                
            # Load selected checklist
            checklist_path = f'./checklists/{self.select_checklist_widget.value}.json'
            criteria = self.load_checklist(checklist_path)
            if not criteria:
                with self.error_area:
                    print(f"오류: 체크리스트를 로드할 수 없습니다: {checklist_path}")
                return
            
            # Get student submission
            submission_content = self.submission_text.value
            if not submission_content.strip():
                with self.error_area:
                    print("오류: 학생 제출물이 비어있습니다.")
                return
            
            # Initialize LLM API client
            client = LLMAPIClient(log_level=logging.INFO)
            client.logger = self.logger
            
            # Format the system instruction
            grade = self.grade_widget.value
            subject = self.subject_widget.value
            assessment_title = self.title_widget.value
            assessment_type = self.assessment_type_widget.value
            assessment_description = self.description_widget.value
            
            system_instruction = self.system_instruction_widget.value
            system_instruction = system_instruction.replace('[학년]', grade)
            system_instruction = system_instruction.replace('[과목]', subject)
            system_instruction = system_instruction.replace('[수행평가 제목]', assessment_title)
            system_instruction = system_instruction.replace('[수행평가 유형]', assessment_type)
            system_instruction = system_instruction.replace('[수행평가 설명]', assessment_description)
            
            # Format the prompt
            prompt = self.prompt_widget.value
            prompt = prompt.replace('[평가 기준]', json.dumps(criteria, ensure_ascii=False, indent=2))
            prompt = prompt.replace('[학생 제출물]', submission_content)
            
            print("System Instruction:")
            print(system_instruction)
            
            print("\n평가 중...") 
            
            # Get PDF file path if uploaded
            pdf_path = self.pdf_file_path
            
            # Process with selected models
            models_to_process = {
                'Gemini': self.gemini_model_selection.value if self.gemini_model_selection.value != "None" else None,
                'Claude': self.claude_model_selection.value if self.claude_model_selection.value != "None" else None,
                'OpenAI': self.openai_model_selection.value if self.openai_model_selection.value != "None" else None
            }
            
            params_map = {
                'Gemini': self.gemini_params,
                'Claude': self.claude_params,
                'OpenAI': self.openai_params
            }
            
            process_methods = {
                'Gemini': client.process_pdf_with_gemini,
                'Claude': client.process_pdf_with_claude,
                'OpenAI': client.process_pdf_with_openai
            }
            
            for model_name, model_value in models_to_process.items():
                if not model_value:
                    continue
                    
                try:
                    print(f"\n{model_name}로 처리 중...")
                    params = params_map[model_name]
                    
                    response = process_methods[model_name](
                        file_path=pdf_path,
                        prompt=prompt,
                        model_name=model_value,
                        schema=self.custom_schema,
                        system_instruction=system_instruction,
                        temperature=params['temperature'].value,
                        max_tokens=params['max_tokens'].value
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
                    
                    self.evaluation_results[model_name] = response_text
                    print(f"{model_name} 평가 완료")
                    
                    # Enable save button
                    self.save_buttons[model_name].disabled = False
                    
                except Exception as e:
                    with self.error_area:
                        print(f"{model_name} 처리 중 오류: {str(e)}")
            
            # Display token usage summary
            usage_summary = client.get_token_usage_summary()
            self.display_token_usage(usage_summary)
    
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
    
    def run_visualization(self, b):
        """Visualize evaluation results"""
        with self.visualization_output:
            self.visualization_output.clear_output()
            self.visualize_results(self.evaluation_results)
    
    def visualize_results(self, results_dict):
        """Format and display evaluation results as HTML"""
        if not results_dict:
            print("시각화할 평가 결과가 없습니다.")
            return
        
        for model_name, result_text in results_dict.items():
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
    
    def save_result(self, model):
        """Create a closure for saving model results"""
        def on_click(b):
            if model not in self.evaluation_results:
                print(f"{model} 평가 결과가 없습니다.")
                return
                
            try:
                model_used = None
                if model == 'Gemini':
                    model_used = self.gemini_model_selection.value
                elif model == 'Claude':
                    model_used = self.claude_model_selection.value
                elif model == 'OpenAI':
                    model_used = self.openai_model_selection.value
                
                # Use a safe filename with timestamp
                safe_filename = re.sub(r'[^\w\-_.]', '_', self.file_name) if self.file_name else 'unnamed'
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                result_file = f"./evaluations/{model_used}_평가결과_{safe_filename}_{timestamp}.json"
                
                # Save result to file
                with open(result_file, 'w', encoding='utf-8') as f:
                    f.write(self.evaluation_results[model])
                
                print(f"{model} 평가 결과가 {result_file}에 저장되었습니다.")
            except Exception as e:
                print(f"{model} 평가 결과 저장 중 오류: {str(e)}")
        return on_click
    
    def display_all_widgets(self):
        """Display all widgets in the appropriate layout"""
        # Create tabs
        tab_content = widgets.Tab()
        
        # Tab 0: 내용 입력
        content_input = widgets.VBox([
            self.grade_widget, self.subject_widget, self.title_widget,
            self.assessment_type_widget, self.select_checklist_widget, self.description_widget,
            self.file_upload, self.submission_text,
            widgets.HTML("<h4>PDF 업로드 (API 요청용)</h4>"),
            self.pdf_upload,
            self.output_area, self.error_area
        ])
        
        # Tab 1: 템플릿 편집
        templates = widgets.VBox([
            widgets.HTML("<h3>템플릿 편집</h3>"),
            self.system_instruction_widget,
            self.prompt_widget,
            self.update_templates_button
        ])
        
        # Tab 2: 모델 세팅
        model_settings = widgets.VBox([
            self.evaluate_button,
            widgets.HBox([
                widgets.VBox([self.gemini_model_selection, self.gemini_params['temperature'], 
                             self.gemini_params['max_tokens'], self.save_buttons['Gemini']]),
                widgets.VBox([self.claude_model_selection, self.claude_params['temperature'], 
                             self.claude_params['max_tokens'], self.save_buttons['Claude']]),
                widgets.VBox([self.openai_model_selection, self.openai_params['temperature'], 
                             self.openai_params['max_tokens'], self.save_buttons['OpenAI']])
            ])
        ])
        
        # Tab 3: 결과 보기
        results_view = widgets.VBox([
            self.visualize_button,
            self.visualization_output
        ])
        
        # Tab 4: 로그 확인
        log_view = widgets.VBox([
            widgets.HTML("<h2>시스템 로그</h2>"),
            self.log_output,
            widgets.HTML("<h2>토큰 사용량</h2>"),
            self.token_usage_output
        ])
        
        # Set tab contents
        tab_content.children = [content_input, templates, model_settings, results_view, log_view]
        
        # Set tab titles
        tab_content.set_title(0, "내용 입력")
        tab_content.set_title(1, "템플릿 편집")
        tab_content.set_title(2, "모델 세팅")
        tab_content.set_title(3, "결과 보기")
        tab_content.set_title(4, "로그 확인")
        
        # Display the tab widget
        display(tab_content)