"""
Utility functions for the educational assessment system.
"""
import os
import re
from datetime import datetime
from typing import Optional


def sanitize_filename(name: str, fallback: str = "untitled") -> str:
    """Sanitize a string to be used as a filename
    
    Args:
        name: The string to sanitize
        fallback: Fallback name if the sanitized string is empty
        
    Returns:
        A sanitized filename
    """
    if not name:
        return fallback
        
    # Replace problematic characters with underscores
    name = re.sub(r'[\\/*?:"<>|]', '_', name)
    
    # Replace multiple spaces with single space
    name = re.sub(r'\s+', ' ', name)
    
    # Trim spaces from beginning and end
    name = name.strip()
    
    # Ensure the name is not too long
    if len(name) > 100:
        name = name[:100]
        
    # If sanitized name is empty, use fallback
    return name if name else fallback


def ensure_directories_exist(directories: list):
    """Ensure that required directories exist"""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def format_template(template: str, replacements: dict) -> str:
    """Format a template string with placeholder replacements
    
    Args:
        template: The template string with [placeholders]
        replacements: Dictionary of placeholder -> value mappings
        
    Returns:
        Formatted template with placeholders replaced
    """
    formatted = template
    for placeholder, value in replacements.items():
        formatted = formatted.replace(f"{{{placeholder}}}", str(value))
    return formatted


def generate_timestamped_filename(
    prefix: str, 
    title: str, 
    extension: str = ".json",
    directory: Optional[str] = None
) -> str:
    """Generate a timestamped filename
    
    Args:
        prefix: Prefix for the filename
        title: Title to include in filename
        extension: File extension (default: .json)
        directory: Optional directory path
        
    Returns:
        Complete file path with timestamp
    """
    safe_filename = sanitize_filename(title)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{prefix}_{safe_filename}_{timestamp}{extension}"
    
    if directory:
        return os.path.join(directory, filename)
    return filename


def check_directory_writable(directory: str) -> bool:
    """Check if a directory is writable
    
    Args:
        directory: Directory path to check
        
    Returns:
        True if writable, False otherwise
    """
    return os.access(directory, os.W_OK)


def extract_model_response_text(response_dict: dict, provider: str) -> str:
    """Extract response text from the structured response dictionary
    
    Args:
        response_dict: The structured response dictionary from LLMAPIClient
        provider: Type of model ('Gemini', 'Anthropic', 'OpenAI')
        
    Returns:
        Extracted response text
    """
    if 'error' in response_dict or 'request_error' in response_dict:
        return f"Error: {response_dict.get('error', response_dict.get('request_error', 'Unknown error'))}"
    
    if 'response' not in response_dict:
        return "Error: No response found in response dictionary"
    
    response = response_dict['response']
    
    try:
        if provider == 'Gemini':
            return response.candidates[0].content.parts[0].text
        elif provider == 'Anthropic':
            response_text = response.content[0].text
            # Extract JSON content if wrapped in markdown code block
            if "```json" in response_text:
                match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
                if match:
                    response_text = match.group(1)
            return response_text
        else:  # OpenAI
            # Handle both regular and thinking models
            if hasattr(response, 'output') and len(response.output) > 1:
                # Thinking model - use the final output
                return response.output[1].content[0].text
            elif hasattr(response, 'choices'):
                # Regular model
                return response.choices[0].message.content
            else:
                return str(response)
    except Exception as e:
        return f"Error extracting response: {str(e)}"
