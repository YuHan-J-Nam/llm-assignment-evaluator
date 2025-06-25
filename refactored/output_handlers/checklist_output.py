"""
Checklist output component for displaying and saving checklist results.
"""
import json
from IPython.display import display, HTML

from ..output_handlers.base_output import BaseOutputComponent
from ..constants import DIRECTORIES


class ChecklistOutputComponent(BaseOutputComponent):
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
    
    def create_save_handler(self, model: str):
        """Create a closure for saving checklist results"""
        def on_click(b):
            success = self._save_result_to_file(
                model=model,
                directory=DIRECTORIES['CHECKLISTS'],
                file_prefix="평가기준",
                success_message_template="{model} 체크리스트가 {file_path}에 저장되었습니다."
            )
            
            # Refresh the checklist component if save was successful
            if success:
                checklist_component = self.manager.get_component('checklist')
                if checklist_component and hasattr(checklist_component, 'refresh_checklists'):
                    checklist_component.refresh_checklists()
        
        return on_click
