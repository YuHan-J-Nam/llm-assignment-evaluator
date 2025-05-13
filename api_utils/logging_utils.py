import logging
import os
import json
import re
from datetime import datetime

def setup_logging(log_level=logging.INFO, log_dir="logs"):
    """Configure logging with timestamps and structured format"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"api_interactions_{timestamp}.log")
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("llm_api")
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

def log_api_request(logger, api_name, endpoint, request_data):
    """Log API request details"""
    logger.info(f"Request to {api_name} API - Endpoint: {endpoint}")
    logger.debug(f"Request data: {request_data}")

def log_api_response(logger, api_name, response_status, response_data=None):
    """Log API response details"""
    logger.info(f"Response from {api_name} API - Status: {response_status}")
    if response_data and logger.level <= logging.DEBUG:
        # Only log detailed response data at debug level
        logger.debug(f"Response data: {response_data}")

def log_api_error(logger, api_name, error_message, exception=None):
    """Log API errors with detailed exception information"""
    logger.error(f"Error in {api_name} API: {error_message}")
    if exception:
        logger.exception(exception)

def extract_token_usage_from_log(log_file_path):
    """Extract and analyze token usage information from log files
    
    Parses log files to extract token usage information and returns
    a summary of token usage and cost.
    """
    token_data = []
    token_pattern = re.compile(r'TOKEN_DATA: (\{.*?\})')
    
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                match = token_pattern.search(line)
                if match:
                    try:
                        data = json.loads(match.group(1))
                        if 'token_usage' in data:
                            timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', line)
                            timestamp = timestamp_match.group(1) if timestamp_match else None
                            
                            token_entry = data['token_usage']
                            token_entry['timestamp'] = timestamp
                            token_data.append(token_entry)
                    except json.JSONDecodeError:
                        pass
        
        # Calculate totals
        if token_data:
            total_tokens = sum(entry.get('total_tokens', 0) for entry in token_data)
            total_prompt_tokens = sum(entry.get('prompt_tokens', 0) for entry in token_data)
            total_completion_tokens = sum(entry.get('completion_tokens', 0) for entry in token_data)
            total_cost = sum(entry.get('estimated_cost', 0) for entry in token_data)
            
            return {
                'entries': token_data,
                'summary': {
                    'total_requests': len(token_data),
                    'total_tokens': total_tokens,
                    'total_prompt_tokens': total_prompt_tokens,
                    'total_completion_tokens': total_completion_tokens,
                    'total_estimated_cost': total_cost
                }
            }
        return {'entries': [], 'summary': {'total_requests': 0}}
    
    except Exception as e:
        logging.error(f"Error extracting token usage from log: {str(e)}")
        return {'entries': [], 'summary': {'total_requests': 0, 'error': str(e)}}