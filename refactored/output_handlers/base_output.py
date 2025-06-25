"""
Abstract base classes for output components.
Provides common functionality for all output handlers.
"""
import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any
import ipywidgets as widgets
from IPython.display import display, HTML

from ..base_classes import BaseComponent
from ..utils import sanitize_filename, generate_timestamped_filename, check_directory_writable


class BaseOutputComponent(BaseComponent, ABC):
    """Abstract base class for output components"""
    
    def __init__(self, manager):
        """Initialize output component"""
        super().__init__(manager)
        self.create_widgets()
        self.results = {}
    
    def create_widgets(self):
        """Create common output widgets"""
        self.visualization_output = widgets.Output()
        self.token_usage_output = widgets.Output()
        self.visualize_button = widgets.Button(description="결과 출력")
        self.visualize_button.on_click(self.display_results)
    
    def get_widgets(self):
        """Return all widgets in this component"""
        return [
            self.visualize_button,
            self.visualization_output,
        ]
    
    def create_layout(self):
        """Create a layout for the output widgets"""
        return widgets.VBox([
            self.visualize_button,
            self.visualization_output,
        ])
    
    def store_result(self, model_name: str, result: str):
        """Store a result for later display"""
        self.results[model_name] = result
    
    def get_results(self) -> Dict[str, str]:
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
    
    @abstractmethod
    def display_results(self, b=None):
        """Display stored results (to be implemented by subclasses)"""
        pass
    
    @abstractmethod
    def create_save_handler(self, model: str):
        """Create a save handler for a specific model (to be implemented by subclasses)"""
        pass
    
    def display_token_usage(self, usage_summary: Dict[str, Any]):
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

    def _save_result_to_file(
        self, 
        model: str, 
        directory: str, 
        file_prefix: str,
        success_message_template: str
    ) -> bool:
        """Common save functionality for all output components
        
        Args:
            model: Model name
            directory: Directory to save to
            file_prefix: Prefix for the filename
            success_message_template: Template for success message (should contain {model} and {file_path})
            
        Returns:
            True if saved successfully, False otherwise
        """
        results = self.get_results()
        if model not in results:
            print(f"{model} 결과가 없습니다.")
            return False
            
        try:
            # Get input component for title
            input_component = self.manager.get_component('input')
            title = input_component.title_widget.value if input_component else "untitled"
            
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
            os.makedirs(directory, exist_ok=True)
            
            # Generate filename
            result_file = generate_timestamped_filename(
                f"{model_used}_{file_prefix}",
                title,
                directory=directory
            )
            
            # Check if directory is writable
            if not check_directory_writable(directory):
                raise PermissionError(f"디렉토리에 쓰기 권한이 없습니다: '{directory}'")
            
            # Save to file
            with open(result_file, 'w', encoding='utf-8') as f:
                f.write(results[model])
            
            print(success_message_template.format(model=model, file_path=result_file))
            return True
            
        except PermissionError as e:
            print(f"권한 오류: {str(e)}")
        except IOError as e:
            print(f"파일 저장 오류: {str(e)}")
        except Exception as e:
            print(f"{model} 결과 저장 중 오류: {str(e)}")
        
        return False
