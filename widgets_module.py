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
        # Load checklist files
        self.checklist_files = [f for f in os.listdir('./checklists') if f.endswith('.json')]
        
        # Setup logging
        self.log_output = widgets.Output(
            layout=widgets.Layout(height='150px', overflow='auto', overflow_x='auto', border='1px solid gray')
        )
        self.setup_logging()
        
        # Create all widgets
        self.create_info_widgets()
        self.create_submission_widgets()
        self.create_model_selection_widgets()
        self.create_output_widgets()
        
        # Initialize evaluation results storage
        self.evaluation_results = {}
        self.file_name = ""
        
        # Define evaluation schema
        self.define_schema()
        
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
            options=[checklist_name.replace('.json', '') for checklist_name in self.checklist_files],
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
        
        # Submission text area
        self.submission_text = widgets.Textarea(
            description="제출물 내용:",
            placeholder='학생 제출물을 직접 입력하거나, 위에서 파일을 업로드하세요.',
            layout=widgets.Layout(width='80%', height='200px')
        )
        
        # Add file upload handler
        self.file_upload.observe(self.process_uploaded_file, names='value')
    
    def process_uploaded_file(self, change):
        """Process uploaded file and update submission text"""
        if self.file_upload.value:
            # Get uploaded file content
            uploaded_content = self.file_upload.value[0]['content']
            self.submission_text.value = io.BytesIO(uploaded_content).read().decode('utf-8')
            
            # Extract file name
            try:
                self.file_name = self.file_upload.value[0]['name'][:-4]
            except IndexError:
                print("File name extraction error")
    
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
        self.visualize_button = widgets.Button(description="결과 시각화")
        
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
        # Add token usage output widget
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
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def create_system_instruction(self, grade, subject, assessment_title, assessment_type, assessment_description):
        """Generate formatted system instruction from user inputs"""
        template = """
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
        
        # Replace placeholders with user inputs
        formatted_instruction = template.replace('[학년]', grade)
        formatted_instruction = formatted_instruction.replace('[과목]', subject)
        formatted_instruction = formatted_instruction.replace('[수행평가 제목]', assessment_title)
        formatted_instruction = formatted_instruction.replace('[수행평가 유형]', assessment_type)
        formatted_instruction = formatted_instruction.replace('[수행평가 설명]', assessment_description)
        
        return formatted_instruction
    
    def run_evaluation(self, b):
        """Execute the evaluation process with selected models"""
        with self.output_area:
            self.output_area.clear_output()
            self.error_area.clear_output()
            
            # Load selected checklist
            criteria = self.load_checklist('./checklists/' + self.select_checklist_widget.value + '.json')
            
            # Create evaluations directory if it doesn't exist
            evaluation_dir = "./evaluations"
            if not os.path.exists(evaluation_dir):
                os.makedirs(evaluation_dir)
            
            # Get student submission
            submission_content = self.submission_text.value
            
            if not submission_content.strip():
                with self.error_area:
                    print("오류: 학생 제출물이 비어있습니다.")
                return
            
            # Initialize LLM API client
            client = LLMAPIClient(log_level=logging.INFO)
            client.logger = self.logger
            
            # Generate system instruction from user inputs
            system_instruction = self.create_system_instruction(
                self.grade_widget.value,
                self.subject_widget.value,
                self.title_widget.value,
                self.assessment_type_widget.value,
                self.description_widget.value
            )
            
            # Define prompt
            prompt = f"""
            다음은 학생이 제출한 수행평가 과제이다. 수행평가 과제에 대한 세부정보와 평가 기준을 고려하여, 각 평가항목에 대하여 논리적으로 평가하라.
            
            평가 기준은 다음과 같다:
            {json.dumps(criteria, ensure_ascii=False, indent=2)}
            
            학생의 수행평가 과제는 다음과 같다:
            {submission_content}
            """
            
            print("System Instruction:")
            print(system_instruction)
            
            print("\n평가 중...") 
            
            # Process with Gemini if selected
            if self.gemini_model_selection.value != "None":
                try:
                    print("\nGemini로 처리 중...")
                    gemini_response = client.process_pdf_with_gemini(
                        file_path=None,
                        prompt=prompt,
                        model_name=self.gemini_model_selection.value,
                        schema=self.custom_schema,
                        system_instruction=system_instruction,
                        temperature=self.gemini_params['temperature'].value,
                        max_tokens=self.gemini_params['max_tokens'].value
                    )
                    gemini_response_text = gemini_response.text
                    self.evaluation_results['Gemini'] = gemini_response_text
                    print("Gemini 평가 완료")
                    
                    # Enable save button
                    self.save_buttons['Gemini'].disabled = False
                    
                except Exception as e:
                    with self.error_area:
                        print(f"Gemini 처리 중 오류: {str(e)}")
            
            # Process with Claude if selected
            if self.claude_model_selection.value != "None":
                try:
                    print("\nClaude로 처리 중...")
                    claude_response = client.process_pdf_with_claude(
                        file_path=None,
                        prompt=prompt,
                        model_name=self.claude_model_selection.value,
                        schema=self.custom_schema,
                        system_instruction=system_instruction,
                        temperature=self.claude_params['temperature'].value,
                        max_tokens=self.claude_params['max_tokens'].value
                    )
                    claude_response_text = claude_response.content[0].text
                    # Extract JSON content if wrapped in markdown code block
                    if "```json" in claude_response_text:
                        claude_response_text = re.search(r'```json\s*(.*?)\s*```', claude_response_text, re.DOTALL).group(1)
                    self.evaluation_results['Claude'] = claude_response_text
                    print("Claude 평가 완료")
                    
                    # Enable save button
                    self.save_buttons['Claude'].disabled = False
                    
                except Exception as e:
                    with self.error_area:
                        print(f"Claude 처리 중 오류: {str(e)}")
            
            # Process with OpenAI if selected
            if self.openai_model_selection.value != "None":
                try:
                    print("\nOpenAI로 처리 중...")
                    openai_response = client.process_pdf_with_openai(
                        file_path=None,
                        prompt=prompt,
                        model_name=self.openai_model_selection.value,
                        schema=self.custom_schema,
                        system_instruction=system_instruction,
                        temperature=self.openai_params['temperature'].value,
                        max_tokens=self.openai_params['max_tokens'].value
                    )
                    openai_response_text = openai_response.choices[0].message.content
                    self.evaluation_results['OpenAI'] = openai_response_text
                    print("OpenAI 평가 완료")
                    
                    # Enable save button
                    self.save_buttons['OpenAI'].disabled = False
                    
                except Exception as e:
                    with self.error_area:
                        print(f"OpenAI 처리 중 오류: {str(e)}")
            
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
        
        # Update token usage output widget instead of directly displaying
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
                        html_output += f"<td>{subcategory.get('evidence', '')}</td>"
                        html_output += "</tr>"
                
                html_output += "</table>"
                
                # Display HTML
                display(HTML(html_output))
                
            except json.JSONDecodeError:
                print(f"{model_name} 결과를 파싱할 수 없습니다. 유효한 JSON 형식이 아닙니다.")
            except Exception as e:
                print(f"{model_name} 결과 시각화 중 오류: {str(e)}")
    
    def save_result(self, model):
        """Create a closure for saving model results"""
        def on_click(b):
            if model in self.evaluation_results:
                if model == 'Gemini':
                    model_used = self.gemini_model_selection.value
                elif model == 'Claude':
                    model_used = self.claude_model_selection.value
                elif model == 'OpenAI':
                    model_used = self.openai_model_selection.value
                
                # Create filename with timestamp
                result_file = f"./evaluations/{model_used}_평가결과_{self.file_name}_시간_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                # Save result to file
                with open(result_file, 'w', encoding='utf-8') as f:
                    f.write(self.evaluation_results[model])
                
                print(f"{model} 평가 결과가 {result_file}에 저장되었습니다.")
            else:
                print(f"{model} 평가 결과가 없습니다.")
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
            self.output_area, self.error_area
        ])
        
        # Tab 1: 모델 세팅
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
        
        # Tab 2: 결과 보기
        results_view = widgets.VBox([
            self.visualize_button,
            self.visualization_output
        ])
        
        # Tab 3: 로그 확인 - Updated to include token usage output
        log_view = widgets.VBox([
            widgets.HTML("<h2>시스템 로그</h2>"),
            self.log_output,
            widgets.HTML("<h2>토큰 사용량</h2>"),
            self.token_usage_output
        ])
        
        # Set tab contents
        tab_content.children = [content_input, model_settings, results_view, log_view]
        
        # Set tab titles
        tab_content.set_title(0, "내용 입력")
        tab_content.set_title(1, "모델 세팅")
        tab_content.set_title(2, "결과 보기")
        tab_content.set_title(3, "로그 확인")
        
        # Display the tab widget
        display(tab_content)