"""
Model selection component for choosing and configuring LLM models.
"""
import ipywidgets as widgets
from ..base_classes import BaseComponent
from ..constants import MODEL_OPTIONS, MODEL_DEFAULTS, THINKING_MODELS


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
            options=MODEL_OPTIONS['GEMINI'],
            value=MODEL_DEFAULTS['GEMINI'],
            description='Gemini 모델:'
        )
        
        self.claude_model_selection = widgets.Dropdown(
            options=MODEL_OPTIONS['ANTHROPIC'],
            value=MODEL_DEFAULTS['ANTHROPIC'],
            description='Anthropic 모델:'
        )
        
        self.openai_model_selection = widgets.Dropdown(
            options=MODEL_OPTIONS['OPENAI'],
            value=MODEL_DEFAULTS['OPENAI'],
            description='OpenAI 모델:'
        )
        
        # Create model parameters
        self.gemini_params = self.create_model_params_widgets('Gemini')
        self.claude_params = self.create_model_params_widgets('Anthropic')
        self.openai_params = self.create_model_params_widgets('OpenAI')
        
        # Create thinking/reasoning options
        self.enable_thinking = widgets.Checkbox(
            value=False,
            description='고급 추론 모드 활성화',
            tooltip='지원되는 모델에서 향상된 추론 기능을 활성화합니다'
        )
        
        self.thinking_budget = widgets.IntText(
            value=10000,
            description='추론 토큰 예산:',
            tooltip='추론 단계에 사용할 최대 토큰 수'
        )
        
        # Create save buttons
        self.save_buttons = {
            'Gemini': widgets.Button(description="Gemini 결과 저장", disabled=True),
            'Anthropic': widgets.Button(description="Anthropic 결과 저장", disabled=True),
            'OpenAI': widgets.Button(description="OpenAI 결과 저장", disabled=True)
        }
        
        # Create action button - will be overridden by concrete classes
        self.action_button = widgets.Button(description="실행")
        
        # Add observers for thinking mode visibility
        self.enable_thinking.observe(self._on_thinking_change, names='value')
    
    def _on_thinking_change(self, change):
        """Handle thinking mode checkbox changes"""
        if change['new']:
            self.thinking_budget.layout.visibility = 'visible'
        else:
            self.thinking_budget.layout.visibility = 'hidden'
    
    def create_model_params_widgets(self, model_name: str):
        """Create parameter widgets for a specific model"""
        return {
            'temperature': widgets.FloatSlider(
                value=0.10, 
                min=0, 
                max=1, 
                step=0.01, 
                description='Temperature:'
            ),
            'max_tokens': widgets.IntText(
                value=4096, 
                description='Max Tokens:'
            )
        }
    
    def get_widgets(self):
        """Return all widgets in this component"""
        widgets_list = [
            self.action_button,
            self.enable_thinking,
            self.thinking_budget,
            self.gemini_model_selection, 
            self.gemini_params['temperature'], 
            self.gemini_params['max_tokens'],
            self.save_buttons['Gemini'],
            self.claude_model_selection, 
            self.claude_params['temperature'], 
            self.claude_params['max_tokens'],
            self.save_buttons['Anthropic'],
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
            widgets.VBox([
                widgets.HTML("<h4>고급 설정</h4>"),
                self.enable_thinking,
                self.thinking_budget
            ]),
            widgets.HBox([
                widgets.VBox([
                    widgets.HTML("<h4>Gemini</h4>"),
                    self.gemini_model_selection, 
                    self.gemini_params['temperature'], 
                    self.gemini_params['max_tokens'], 
                    self.save_buttons['Gemini']
                ]),
                widgets.VBox([
                    widgets.HTML("<h4>Anthropic</h4>"),
                    self.claude_model_selection, 
                    self.claude_params['temperature'], 
                    self.claude_params['max_tokens'], 
                    self.save_buttons['Anthropic']
                ]),
                widgets.VBox([
                    widgets.HTML("<h4>OpenAI</h4>"),
                    self.openai_model_selection, 
                    self.openai_params['temperature'], 
                    self.openai_params['max_tokens'], 
                    self.save_buttons['OpenAI']
                ])
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
            'Anthropic': {
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
    
    def get_thinking_settings(self):
        """Get thinking/reasoning settings"""
        return {
            'enable_thinking': self.enable_thinking.value,
            'thinking_budget': self.thinking_budget.value if self.enable_thinking.value else None
        }
    
    def is_thinking_supported(self, model_name: str, provider: str) -> bool:
        """Check if a model supports thinking/reasoning"""
        if provider in THINKING_MODELS:
            return model_name in THINKING_MODELS[provider]
        return False
        
    def set_save_handler(self, model: str, handler):
        """Set the save handler for a specific model's save button"""
        self.save_buttons[model].on_click(handler)
    
    def enable_save_button(self, model: str):
        """Enable the save button for a specific model"""
        self.save_buttons[model].disabled = False
        
    def disable_save_button(self, model: str):
        """Disable the save button for a specific model"""
        self.save_buttons[model].disabled = True
        
    def set_action_handler(self, handler):
        """Set the handler for the action button"""
        self.action_button.on_click(handler)
        
    def set_action_button_text(self, text: str):
        """Set the text for the action button"""
        self.action_button.description = text
