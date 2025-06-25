"""
Evaluation output handler for displaying assignment evaluation results.
Shows structured evaluation data with scores and feedback.
"""
import os
import json
from datetime import datetime
import ipywidgets as widgets
from IPython.display import display, HTML

from ..output_handlers.base_output import BaseOutputComponent
from ..utils import sanitize_filename


class EvaluationOutputComponent(BaseOutputComponent):
    """Component for evaluation output display"""
    
    def display_results(self, b=None):
        """Display evaluation results"""
        with self.visualization_output:
            self.visualization_output.clear_output()
            
            results = self.get_results()
            if not results:
                print("평가 결과가 없습니다.")
                return
            
            for model_name, result_text in results.items():
                try:
                    # Parse JSON result
                    if isinstance(result_text, str):
                        result = json.loads(result_text)
                    else:
                        result = result_text
                    
                    # Display formatted result
                    html_output = f"<h2>{model_name} 평가 결과</h2>"
                    html_output += "<div style='border: 1px solid #ddd; padding: 15px; border-radius: 5px; margin: 10px 0;'>"
                    
                    # Display evaluation results
                    if 'evaluation' in result:
                        evaluation_data = result['evaluation']
                        
                        # Create a summary table
                        html_output += "<h3>평가 요약</h3>"
                        html_output += "<table border='1' style='border-collapse: collapse; width: 100%; margin: 10px 0;'>"
                        html_output += "<tr style='background-color: #f0f0f0;'>"
                        html_output += "<th style='padding: 8px; text-align: left;'>대분류</th>"
                        html_output += "<th style='padding: 8px; text-align: left;'>소분류</th>"
                        html_output += "<th style='padding: 8px; text-align: center;'>점수</th>"
                        html_output += "<th style='padding: 8px; text-align: left;'>평가 이유</th>"
                        html_output += "</tr>"
                        
                        total_score = 0
                        total_items = 0
                        
                        for category in evaluation_data:
                            category_name = category.get('category', '')
                            subcategories = category.get('subcategories', [])
                            
                            for i, subcategory in enumerate(subcategories):
                                name = subcategory.get('name', '')
                                score = subcategory.get('score', 0)
                                reason = subcategory.get('reason', '')
                                evidence = subcategory.get('evidence', [])
                                
                                total_score += score
                                total_items += 1
                                
                                # Color code the score
                                if score >= 2:
                                    score_color = "green"
                                elif score >= 1:
                                    score_color = "orange"
                                else:
                                    score_color = "red"
                                
                                html_output += "<tr>"
                                if i == 0:
                                    html_output += f"<td rowspan='{len(subcategories)}' style='padding: 8px; background-color: #f9f9f9; vertical-align: top;'><strong>{category_name}</strong></td>"
                                html_output += f"<td style='padding: 8px;'>{name}</td>"
                                html_output += f"<td style='padding: 8px; text-align: center; color: {score_color}; font-weight: bold;'>{score}/3</td>"
                                html_output += f"<td style='padding: 8px;'>{reason}"
                                
                                # Add evidence if available
                                if evidence:
                                    html_output += "<br><small><strong>증거:</strong> "
                                    for j, ev in enumerate(evidence[:2]):  # Limit to first 2 pieces of evidence
                                        if j > 0:
                                            html_output += "; "
                                        html_output += f'"{ev[:100]}..."'
                                    html_output += "</small>"
                                
                                html_output += "</td></tr>"
                        
                        # Add total score row
                        if total_items > 0:
                            avg_score = total_score / total_items
                            html_output += f"<tr style='background-color: #e6f3ff; font-weight: bold;'>"
                            html_output += f"<td colspan='2' style='padding: 8px; text-align: right;'>총 평균 점수</td>"
                            html_output += f"<td style='padding: 8px; text-align: center;'>{avg_score:.2f}/3</td>"
                            html_output += f"<td style='padding: 8px;'>총 {total_items}개 항목 평가</td>"
                            html_output += "</tr>"
                        
                        html_output += "</table>"
                    
                    # Display overall feedback
                    if 'overall_feedback' in result:
                        feedback = result['overall_feedback']
                        html_output += "<h3>종합 피드백</h3>"
                        html_output += f"<div style='background-color: #f0f8ff; padding: 15px; border-left: 4px solid #007acc; margin: 10px 0;'>"
                        html_output += f"<p>{feedback}</p>"
                        html_output += "</div>"
                    
                    html_output += "</div>"
                    
                    # Display the HTML output
                    display(HTML(html_output))
                    
                except json.JSONDecodeError:
                    print(f"{model_name} 결과를 파싱할 수 없습니다. 유효한 JSON 형식이 아닙니다.")
                    print(f"Response: {str(result_text)[:500]}...")
                except Exception as e:
                    print(f"{model_name} 평가 결과 출력 중 오류: {str(e)}")
                    if hasattr(self.manager, 'logger'):
                        self.manager.logger.error(f"{model_name} 평가 결과 출력 중 오류: {str(e)}")
    
    def create_save_handler(self, model):
        """Create a closure for saving evaluation results"""
        def on_click(b):
            results = self.get_results()
            if model not in results:
                print(f"{model} 평가 결과가 없습니다.")
                return
                
            try:
                # Get input component for metadata
                input_component = self.manager.get_component('input')
                student_component = self.manager.get_component('student_submission')
                
                # Get title and student info
                if input_component and hasattr(input_component, 'title_widget'):
                    title = input_component.title_widget.value or "evaluation"
                else:
                    title = "evaluation"
                
                # Get student file name if available
                student_file = ""
                if student_component and hasattr(student_component, 'get_file_name'):
                    student_file = student_component.get_file_name()
                
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
                os.makedirs('./evaluations', exist_ok=True)
                
                # Create filename with more descriptive naming
                if student_file:
                    result_file = f"./evaluations/{model_used}_평가_{safe_filename}_{student_file}_{timestamp}.json"
                else:
                    result_file = f"./evaluations/{model_used}_평가_{safe_filename}_{timestamp}.json"
                
                # Save to file
                with open(result_file, 'w', encoding='utf-8') as f:
                    if isinstance(results[model], str):
                        f.write(results[model])
                    else:
                        json.dump(results[model], f, ensure_ascii=False, indent=2)
                
                print(f"{model} 평가 결과가 {result_file}에 저장되었습니다.")
                
            except Exception as e:
                print(f"{model} 평가 결과 저장 중 오류: {str(e)}")
                if hasattr(self.manager, 'logger'):
                    self.manager.logger.error(f"{model} 평가 결과 저장 중 오류: {str(e)}")
        
        return on_click
