"""
Base classes for the educational assessment system.
Provides common functionality for all managers and components.
"""
import os
import logging
from typing import Dict, Any, Optional
import ipywidgets as widgets
from llm_api_client import LLMAPIClient
from .constants import DIRECTORIES
from .utils import ensure_directories_exist, extract_model_response_text


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
        ensure_directories_exist(list(DIRECTORIES.values()))
        
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
    
    def add_component(self, name: str, component):
        """Add a component to this widget manager"""
        self.components[name] = component
        setattr(self, f"{name}_component", component)
    
    def get_component(self, name: str):
        """Get a component by name"""
        return self.components.get(name)
        
    def display_all(self):
        """Display all widgets (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement display_all method")
    
    def display_error(self, message: str):
        """Display an error message in the error area"""
        with self.error_area:
            self.error_area.clear_output()
            print(f"오류: {message}")
    
    def display_log(self, message: str):
        """Log a message to the log output"""
        self.logger.info(message)
    
    def add_temp_file(self, file_path: str):
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
    
    def process_with_models(
        self, 
        prompt: str, 
        system_instruction: str, 
        schema: Dict[str, Any], 
        pdf_path: Optional[str] = None, 
        completion_message: Optional[str] = None,
        enable_thinking: bool = False,
        thinking_budget: Optional[int] = None
    ) -> bool:
        """Process request with selected models and return results
        
        Args:
            prompt: The prompt to send to the models
            system_instruction: The system instruction to send to the models
            schema: The JSON schema for structured output
            pdf_path: Optional path to a PDF file to process
            completion_message: Optional custom completion message to display
            enable_thinking: Whether to enable thinking for supported models
            thinking_budget: Optional token budget for thinking
            
        Returns:
            True if processing completed, False otherwise
        """
        # Initialize API client
        client = self.initialize_api_client()
        
        # Get model component
        model_component = self.get_component('model')
        if not model_component:
            self.display_error("모델 컴포넌트를 찾을 수 없습니다.")
            return False
        
        # Process with selected models
        models = model_component.get_selected_models()
        
        for model_name, model_params in models.items():
            if not model_params['model_name']:
                continue
                
            try:
                print(f"\n{model_name}로 처리 중...")
                
                # Use the correct internal method names from LLMAPIClient
                if model_name == 'Gemini':
                    response_dict = client._process_with_gemini(
                        file_path=pdf_path,
                        prompt=prompt,
                        model=model_params['model_name'],
                        system_instruction=system_instruction,
                        response_schema=schema,
                        temperature=model_params['temperature'],
                        max_tokens=model_params['max_tokens'],
                        enable_thinking=enable_thinking,
                        thinking_budget=thinking_budget
                    )
                elif model_name == 'Claude':
                    response_dict = client._process_with_anthropic(
                        file_path=pdf_path,
                        prompt=prompt,
                        model=model_params['model_name'],
                        system_instruction=system_instruction,
                        response_schema=schema,
                        temperature=model_params['temperature'],
                        max_tokens=model_params['max_tokens'],
                        enable_thinking=enable_thinking,
                        thinking_budget=thinking_budget
                    )
                elif model_name == 'OpenAI':
                    response_dict = client._process_with_openai(
                        file_path=pdf_path,
                        prompt=prompt,
                        model=model_params['model_name'],
                        system_instruction=system_instruction,
                        response_schema=schema,
                        temperature=model_params['temperature'],
                        max_tokens=model_params['max_tokens'],
                        enable_thinking=enable_thinking,
                        thinking_budget=thinking_budget
                    )
                
                # Extract response text from structured response
                response_text = extract_model_response_text(response_dict, model_name)
                
                # Store result in output component
                output_component = self.get_component('output')
                if output_component:
                    output_component.store_result(model_name, response_text)
                
                if completion_message:
                    print(f"{model_name} {completion_message}")
                else:
                    print(f"{model_name} 처리 완료")
                
                # Enable save button
                model_component.enable_save_button(model_name)
                
            except Exception as e:
                with self.error_area:
                    print(f"{model_name} 처리 중 오류: {str(e)}")
        
        # Display token usage summary
        output_component = self.get_component('output')
        if output_component:
            usage_summary = client.get_token_usage_summary()
            output_component.display_token_usage(usage_summary)
        
        return True


class BaseComponent:
    """Base class for widget components"""
    
    def __init__(self, manager: BaseWidgetManager):
        """Initialize the component with a reference to its manager"""
        self.manager = manager
        
    def get_widgets(self):
        """Return a list of all widgets in this component"""
        raise NotImplementedError("Subclasses must implement get_widgets method")
        
    def create_layout(self):
        """Create and return a layout widget containing all widgets in this component"""
        raise NotImplementedError("Subclasses must implement create_layout method")
    
    def validate_inputs(self) -> bool:
        """Validate component inputs (override in subclasses as needed)"""
        return True
