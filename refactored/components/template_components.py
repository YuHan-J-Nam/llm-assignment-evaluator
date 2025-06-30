"""
Template editing components for system instructions and prompts.
"""
import ipywidgets as widgets
from ..base_classes import BaseComponent
from ..utils import format_template


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

        # RAG toggle button
        self.rag_toggle_button = widgets.ToggleButton(
            value=True,  # Default enabled
            description='RAG 사용',
            button_style='success',
            tooltip='RAG(검색 증강 생성)를 사용하여 체크리스트 생성을 향상시킵니다.'
        )
        self.rag_toggle_button.observe(self._on_rag_toggle, names='value')
    
    def get_widgets(self):
        """Return all widgets in this component"""
        return [
            self.system_instruction_widget,
            self.prompt_widget,
            self.update_templates_button,
            self.rag_toggle_button
        ]
    
    def create_layout(self):
        """Create a layout for the template widgets"""
        return widgets.VBox([
            widgets.HTML("<h3>템플릿 편집</h3>"),
            self.system_instruction_widget,
            self.prompt_widget,
            widgets.HBox([self.update_templates_button, self.rag_toggle_button])
        ])
    
    def update_templates(self, b=None):
        """Update templates when the update button is clicked"""
        # Check if input component exists and has valid values
        input_component = self.manager.get_component('input')
        if not input_component or not input_component.validate_inputs():
            return
            
        # Get input values
        values = input_component.get_values()
        
        # Format templates using the helper method
        self.system_instruction_widget.value = format_template(self.system_template, values)
        
        self.prompt_widget.value = format_template(self.prompt_template, values)
        
        print("템플릿이 업데이트되었습니다.")
    
    def get_formatted_system_instruction(self, replacements=None):
        """Get the current system instruction with optional additional replacements"""
        system_instruction = self.system_instruction_widget.value
        
        if replacements:
            system_instruction = format_template(system_instruction, replacements)
                
        return system_instruction
        
    def get_formatted_prompt(self, replacements=None):
        """Get the current prompt with optional additional replacements"""
        prompt = self.prompt_widget.value
        
        if replacements:
            prompt = format_template(prompt, replacements)
                
        return prompt
    
    def _on_rag_toggle(self, change):
        """Handle RAG toggle button state change"""
        if change['new']:
            self.rag_toggle_button.button_style = 'success'
            # print("✓ RAG 기능이 활성화되었습니다.")
        else:
            self.rag_toggle_button.button_style = 'warning'
            # print("⚠ RAG 기능이 비활성화되었습니다.")
    
    def is_rag_enabled(self):
        """Check if RAG is enabled"""
        return self.rag_toggle_button.value
    
    def set_rag_enabled(self, enabled):
        """Set RAG enabled state"""
        self.rag_toggle_button.value = enabled
