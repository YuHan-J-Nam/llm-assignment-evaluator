"""
LLM Call output handler for raw API responses.
Displays unstructured responses from LLM API calls without schema enforcement.
"""
import os
import json
from datetime import datetime
import ipywidgets as widgets
from IPython.display import display, HTML

from ..base_classes import BaseComponent
from ..utils import sanitize_filename


class LlmCallOutputComponent(BaseComponent):
    """Component for raw LLM API call output display"""
    
    def __init__(self, manager):
        """Initialize output component"""
        super().__init__(manager)
        self.create_widgets()
        self.results = {}
    
    def create_widgets(self):
        """Create output widgets"""
        self.visualization_output = widgets.Output()
        self.token_usage_output = widgets.Output()
        self.visualize_button = widgets.Button(
            description="결과 출력",
            button_style='info',
            layout=widgets.Layout(width='150px')
        )
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
            widgets.HTML("<h3>LLM 호출 결과</h3>"),
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
                html_output += f"<pre style='white-space: pre-wrap; word-wrap: break-word; background-color: #f5f5f5; padding: 10px; border-radius: 5px; border: 1px solid #ddd;'>{result_text}</pre>"
                display(HTML(html_output))
    
    def display_token_usage(self, usage_summary):
        """Display token usage as an HTML table"""
        html = "<h3>토큰 사용량 요약</h3>"
        html += "<table border='1' style='border-collapse: collapse; margin: 10px; width: 100%;'>"
        
        # Table headers
        headers = ['Model', 'Requests', 'Input Tokens', 'Output Tokens', 'Total Tokens', 'Est. Cost']
        html += "<tr style='background-color: #f0f0f0;'>" + "".join(f"<th style='padding: 8px; text-align: left;'>{header}</th>" for header in headers) + "</tr>"
        
        # Data for each model
        for model, stats in usage_summary.get('models', {}).items():
            html += "<tr>"
            html += f"<td style='padding: 8px;'>{model}</td>"
            html += f"<td style='padding: 8px;'>{stats.get('requests', 0)}</td>"
            html += f"<td style='padding: 8px;'>{stats.get('prompt_tokens', 0):,}</td>"
            html += f"<td style='padding: 8px;'>{stats.get('completion_tokens', 0):,}</td>"
            html += f"<td style='padding: 8px;'>{stats.get('total_tokens', 0):,}</td>"
            html += f"<td style='padding: 8px;'>${stats.get('estimated_cost', 0):.6f}</td>"
            html += "</tr>"

        # Total usage row
        html += "<tr style='background-color: #f9f9f9; font-weight: bold;'>"
        html += "<td style='padding: 8px;'>Total</td>"
        html += f"<td style='padding: 8px;'>{usage_summary.get('total_requests', 0)}</td>"
        html += f"<td style='padding: 8px;'>{usage_summary.get('total_prompt_tokens', 0):,}</td>"
        html += f"<td style='padding: 8px;'>{usage_summary.get('total_completion_tokens', 0):,}</td>"
        html += f"<td style='padding: 8px;'>{usage_summary.get('total_tokens', 0):,}</td>"
        html += f"<td style='padding: 8px;'>${usage_summary.get('total_estimated_cost', 0):.6f}</td>"
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
                if input_component and hasattr(input_component, 'title_widget'):
                    title = input_component.title_widget.value or "llm_call"
                else:
                    title = "llm_call"
                
                # Create a safe filename with timestamp
                safe_filename = sanitize_filename(title)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # Get model name from model component
                model_component = self.manager.get_component('model')
                model_used = None
                if model_component:
                    models = model_component.get_selected_models()
                    if model in models:
                        model_used = models[model].get('model', model.lower())
                
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
                if hasattr(self.manager, 'logger'):
                    self.manager.logger.error(f"{model} 결과 저장 중 오류: {str(e)}")
        
        return on_click
