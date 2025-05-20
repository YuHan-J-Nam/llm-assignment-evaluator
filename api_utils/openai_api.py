import base64
import logging
import os
from api_utils.config import init_openai_client
from api_utils.pdf_utils import validate_pdf

class OpenAIAPI:
    """Interface for interacting with OpenAI's API"""
    
    def __init__(self, logger=None):
        self.client = init_openai_client()
        self.logger = logger or logging.getLogger("openai_api")
    
    def upload_pdf(self, file_path):
        """Prepare a PDF file for OpenAI API"""
        try:
            validate_pdf(file_path)
            self.logger.info(f"Preparing PDF for OpenAI: {file_path}")
            
            # OpenAI can accept file paths directly
            with open(file_path, "rb") as file:
                file_content = file.read()
                
            file_name = os.path.basename(file_path)
            
            # For message-based models, encode as base64
            base64_pdf = base64.b64encode(file_content).decode("utf-8")
            file_data = {
                "name": file_name,
                "bytes": base64_pdf
            }
            
            self.logger.info(f"Successfully prepared PDF for OpenAI: {file_name}")
            return file_data
            
        except Exception as e:
            self.logger.error(f"Error preparing PDF for OpenAI: {str(e)}")
            raise
    
    def create_message_with_file(self, file_data, prompt):
        """Create a message with file attachment and prompt"""
        if file_data is not None:
            messages = [
                {
                    "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "document",
                        "document": {
                            "type": "pdf",
                            "file_id": file_data['bytes']
                        }
                    }
                    ]
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        return messages
    
    def generate_response(self, model_name, messages, temperature=0.2, max_tokens=2048, system_instruction=None, schema=None):
        """Generate response from OpenAI model"""
        try:
            self.logger.info(f"Sending request to OpenAI model: {model_name}")
            
            # Add system message if provided
            if system_instruction:
                messages.insert(0, {
                    "role": "system",
                    "content": system_instruction
                })
            
            # Create API call parameters
            params = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # If schema is provided, add it to the request
            if schema:
                params["response_format"] = schema

            # Special case for model o4-mini
            if model_name == "o4-mini":
                params.pop("max_tokens", None)  # Remove max_tokens from params
                params["max_completion_tokens"] = max_tokens
                params['temperature'] = 1
                
            # Send request to OpenAI API
            response = self.client.chat.completions.create(**params)
            
            self.logger.info("Successfully received response from OpenAI")
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response from OpenAI: {str(e)}")
            raise
    
    # def is_json_capable_model(self, model_name):
    #     """Check if model supports JSON mode"""
    #     json_capable_models = [
    #         "gpt-4-turbo", "gpt-4-0125-preview", "gpt-4-1106-preview", "gpt-4.1",
    #         "gpt-3.5-turbo", "gpt-3.5-turbo-0125"
    #     ]
    #     return any(model in model_name for model in json_capable_models)
    