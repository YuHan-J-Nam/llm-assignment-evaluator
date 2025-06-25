"""
Checklist management components for evaluation criteria.
"""
import os
import json
import ipywidgets as widgets
from ..base_classes import BaseComponent
from ..constants import DIRECTORIES


class ChecklistComponent(BaseComponent):
    """Component for checklist selection and management"""
    
    def __init__(self, manager):
        """Initialize checklist component"""
        super().__init__(manager)
        self.create_widgets()
    
    def create_widgets(self):
        """Create checklist widgets"""
        # Get checklist files
        checklist_dir = DIRECTORIES['CHECKLISTS']
        self.checklist_files = [f for f in os.listdir(checklist_dir) if f.endswith('.json')]
        
        self.select_checklist_widget = widgets.Dropdown(
            options=[checklist_name.replace('.json', '') for checklist_name in self.checklist_files] if self.checklist_files else ['체크리스트 없음'],
            description='체크리스트:',
            layout=widgets.Layout(width='60%')
        )
    
    def get_widgets(self):
        """Return all widgets in this component"""
        return [self.select_checklist_widget]
    
    def create_layout(self):
        """Create a layout for the checklist widgets"""
        return widgets.VBox(self.get_widgets())
    
    def refresh_checklists(self):
        """Refresh the list of available checklists"""
        checklist_dir = DIRECTORIES['CHECKLISTS']
        self.checklist_files = [f for f in os.listdir(checklist_dir) if f.endswith('.json')]
        self.select_checklist_widget.options = [checklist_name.replace('.json', '') for checklist_name in self.checklist_files] if self.checklist_files else ['체크리스트 없음']
    
    def get_selected_checklist(self):
        """Get the currently selected checklist"""
        return self.select_checklist_widget.value
    
    def load_checklist(self, file_path):
        """Load a checklist from a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            self.manager.logger.error(f"체크리스트 파일을 찾을 수 없습니다: {file_path}")
            return {}
        except json.JSONDecodeError:
            self.manager.logger.error(f"체크리스트 파일이 유효한 JSON 형식이 아닙니다: {file_path}")
            return {}


class SummarizeSubmissionComponent(BaseComponent):
    """Component for submission summarization selection and management"""
    
    def __init__(self, manager):
        """Initialize summary component"""
        super().__init__(manager)
        self.create_widgets()
    
    def create_widgets(self):
        """Create summary widgets"""
        # Get summary files
        summary_dir = DIRECTORIES.get('SUMMARY', './summary')
        if os.path.exists(summary_dir):
            self.summary_files = [f for f in os.listdir(summary_dir) if f.endswith('.json')]
        else:
            self.summary_files = []
        
        self.select_summary_widget = widgets.Dropdown(
            options=[summary_name.replace('.json', '') for summary_name in self.summary_files] if self.summary_files else ['요약 없음'],
            description='요약:',
            layout=widgets.Layout(width='60%')
        )
    
    def get_widgets(self):
        """Return all widgets in this component"""
        return [self.select_summary_widget]
    
    def create_layout(self):
        """Create a layout for the summary widgets"""
        return widgets.VBox(self.get_widgets())
    
    def refresh_summary(self):
        """Refresh the list of available summaries"""
        summary_dir = DIRECTORIES.get('SUMMARY', './summary')
        if os.path.exists(summary_dir):
            self.summary_files = [f for f in os.listdir(summary_dir) if f.endswith('.json')]
        else:
            self.summary_files = []
        self.select_summary_widget.options = [summary_name.replace('.json', '') for summary_name in self.summary_files] if self.summary_files else ['요약 없음']
    
    def get_selected_summary(self):
        """Get the currently selected summary"""
        return self.select_summary_widget.value
    
    def load_summary(self, file_path):
        """Load a summary from a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            self.manager.logger.error(f"요약 파일을 찾을 수 없습니다: {file_path}")
            return {}
        except json.JSONDecodeError:
            self.manager.logger.error(f"요약 파일이 유효한 JSON 형식이 아닙니다: {file_path}")
            return {}
