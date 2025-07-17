"""
Summarize output handler for student submission summaries.
Displays structured summaries of student submissions.
"""
import os
import json
from datetime import datetime
import ipywidgets as widgets
from IPython.display import display, HTML

from ..output_handlers.base_output import BaseOutputComponent
from ..utils import sanitize_filename


class SummarizeOutputComponent(BaseOutputComponent):
    """Component for summary output display"""
    
    def display_results(self, b=None):
        """Display summary results"""
        with self.visualization_output:
            self.visualization_output.clear_output()
            
            results = self.get_results()
            if not results:
                print("보고서 요약 결과물이 없습니다.")
                return
            
            for model_name, result_text in results.items():
                try:
                    # Parse JSON result
                    if isinstance(result_text, str):
                        result = json.loads(result_text)['summary']
                    else:
                        result = result_text['summary']
                    
                    # Display formatted result
                    html_output = f"<h2>{model_name} 보고서 요약</h2>"
                    html_output += "<div style='border: 1px solid #ddd; padding: 15px; border-radius: 5px; margin: 10px 0;'>"
                    
                    # Display each summary category
                    summary_categories = [
                        ('탐구_주제_선정_동기', '탐구 주제 선정 동기'),
                        ('탐구_과정', '탐구 과정'),
                        ('탐구_과정에서_배우고_느낀_점', '탐구 과정에서 배우고 느낀 점'),
                        ('탐구_이후의_방향성', '탐구 이후의 방향성')
                    ]
                    
                    for key, title in summary_categories:
                        if key in result:
                            html_output += f"<h3 style='color: #333; margin-top: 20px;'>{title}</h3>"
                            summary_content = result[key]
                            
                            if summary_content and summary_content.lower() != 'null':
                                html_output += f"<p style='background-color: #f9f9f9; padding: 10px; border-left: 4px solid #007acc; margin: 5px 0;'>{summary_content}</p>"
                            else:
                                html_output += f"<p style='color: #888; font-style: italic;'>해당 항목에 대한 내용이 명시적으로 언급되지 않았습니다.</p>"
                    
                    html_output += "</div>"
                    
                    # Display the HTML output
                    display(HTML(html_output))
                    
                except json.JSONDecodeError:
                    print(f"{model_name} 결과를 파싱할 수 없습니다. 유효한 JSON 형식이 아닙니다.")
                    print(f"Response: {str(result_text)[:500]}...")
                except Exception as e:
                    print(f"{model_name} 요약 출력 중 오류: {str(e)}")
                    if hasattr(self.manager, 'logger'):
                        self.manager.logger.error(f"{model_name} 요약 출력 중 오류: {str(e)}")
    
    def create_save_handler(self, model):
        """Create a closure for saving summary results"""
        def on_click(b):
            results = self.get_results()
            if model not in results:
                print(f"{model} 요약 결과가 없습니다.")
                return
                
            try:
                # Get input component
                input_component = self.manager.get_component('input')
                
                # Get title from input component
                if input_component and hasattr(input_component, 'title_widget'):
                    title = input_component.title_widget.value or "untitled"
                else:
                    title = "untitled"
                
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
                os.makedirs('./summary', exist_ok=True)
                
                # Create filename
                result_file = f"./summary/{model_used}_보고서요약_{safe_filename}_{timestamp}.json"
                
                # Check if directory is writable
                if not os.access('./summary', os.W_OK):
                    raise PermissionError(f"보고서 요약본을 디렉토리에 쓰기 권한이 없습니다: './summary'")
                
                # Save to file
                with open(result_file, 'w', encoding='utf-8') as f:
                    if isinstance(results[model], str):
                        f.write(results[model])
                    else:
                        json.dump(results[model], f, ensure_ascii=False, indent=2)
                
                print(f"{model} 보고서 요약본이 {result_file}에 저장되었습니다.")
                
                # Refresh the summary component if it exists
                summary_component = self.manager.get_component('summary')
                if summary_component and hasattr(summary_component, 'refresh_summary'):
                    summary_component.refresh_summary()
                
            except PermissionError as e:
                print(f"권한 오류: {str(e)}")
            except IOError as e:
                print(f"파일 저장 오류: {str(e)}")
            except Exception as e:
                print(f"{model} 보고서 요약본 저장 중 오류: {str(e)}")
                if hasattr(self.manager, 'logger'):
                    self.manager.logger.error(f"{model} 보고서 요약본 저장 중 오류: {str(e)}")
        
        return on_click
