"""
Main entry point for the refactored educational assessment system.
Provides easy access to all manager classes for different assessment tasks.
"""

# Import all managers
from .managers.checklist_manager import ChecklistCreationManager
from .managers.assignment_evaluation_manager import AssignmentEvaluationManager
from .managers.llm_call_manager import LlmCallManager
from .managers.summarize_submission_manager import SummarizeSubmissionManager

# Import all output handlers for standalone use
from .output_handlers.checklist_output import ChecklistOutputComponent
from .output_handlers.evaluation_output import EvaluationOutputComponent
from .output_handlers.llm_call_output import LlmCallOutputComponent
from .output_handlers.summarize_output import SummarizeOutputComponent

# Import all components for custom manager creation
from .components.input_components import InputWidgetsComponent
from .components.template_components import TemplateWidgetsComponent
from .components.model_components import ModelSelectionComponent
from .components.upload_components import UploadPdfWidgetsComponent, StudentSubmissionWidgetsComponent
from .components.checklist_components import ChecklistComponent, SummarizeSubmissionComponent

# Import base classes for extending functionality
from .base_classes import BaseWidgetManager, BaseComponent

# Import constants and utilities
from .constants import *
from .utils import sanitize_filename, ensure_directories_exist, format_template


__all__ = [
    # Managers
    'ChecklistCreationManager',
    'AssignmentEvaluationManager', 
    'LlmCallManager',
    'SummarizeSubmissionManager',
    
    # Output Handlers
    'ChecklistOutputComponent',
    'EvaluationOutputComponent',
    'LlmCallOutputComponent',
    'SummarizeOutputComponent',
    
    # Components
    'InputWidgetsComponent',
    'TemplateWidgetsComponent',
    'ModelSelectionComponent',
    'UploadPdfWidgetsComponent',
    'StudentSubmissionWidgetsComponent',
    'ChecklistComponent',
    'SummarizeSubmissionComponent',
    
    # Base Classes
    'BaseWidgetManager',
    'BaseComponent',
    
    # Utilities
    'sanitize_filename',
    'ensure_directories_exist',
    'format_template'
]


def create_checklist_interface():
    """Create and return a checklist creation interface
    
    Returns:
        ChecklistCreationManager: Ready-to-use checklist creation interface
    """
    return ChecklistCreationManager()


def create_evaluation_interface():
    """Create and return an assignment evaluation interface
    
    Returns:
        AssignmentEvaluationManager: Ready-to-use evaluation interface
    """
    return AssignmentEvaluationManager()


def create_llm_call_interface():
    """Create and return a direct LLM API call interface
    
    Returns:
        LlmCallManager: Ready-to-use LLM call interface
    """
    return LlmCallManager()


def create_summarize_interface():
    """Create and return a submission summarization interface
    
    Returns:
        SummarizeSubmissionManager: Ready-to-use summarization interface
    """
    return SummarizeSubmissionManager()


# Example usage functions
def demo_checklist_creation():
    """Demonstrate checklist creation functionality"""
    print("Creating checklist creation interface...")
    manager = create_checklist_interface()
    manager.display_all()
    return manager


def demo_assignment_evaluation():
    """Demonstrate assignment evaluation functionality"""
    print("Creating assignment evaluation interface...")
    manager = create_evaluation_interface()
    manager.display_all()
    return manager


def demo_llm_calls():
    """Demonstrate direct LLM API calls"""
    print("Creating LLM call interface...")
    manager = create_llm_call_interface()
    manager.display_all()
    return manager


def demo_submission_summarization():
    """Demonstrate submission summarization functionality"""
    print("Creating submission summarization interface...")
    manager = create_summarize_interface()
    manager.display_all()
    return manager
