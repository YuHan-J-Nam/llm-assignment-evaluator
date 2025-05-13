import base64
import logging
import os
from api_utils.config import init_claude_client
from api_utils.pdf_utils import validate_pdf, get_file_mime_type

class ClaudeAPI:
    """Interface for interacting with Anthropic's Claude API"""
    
    def __init__(self, logger=None):
        self.client = init_claude_client()
        self.logger = logger or logging.getLogger("claude_api")
    
    def upload_pdf(self, file_path):
        """Prepare a PDF file for Claude API"""
        try:
            validate_pdf(file_path)
            self.logger.info(f"Preparing PDF for Claude: {file_path}")
            
            # Claude accepts base64 encoded files in messages
            with open(file_path, "rb") as f:
                bytes_data = f.read()
                base64_data = base64.b64encode(bytes_data).decode("utf-8")
            
            # Prepare file metadata
            file_name = os.path.basename(file_path)
            mime_type = get_file_mime_type(file_path)
            
            media_data = {
                "type": "base64",
                "media_type": mime_type,
                "data": base64_data
            }  
            
            self.logger.info(f"Successfully prepared PDF for Claude: {file_name}")
            return media_data
            
        except Exception as e:
            self.logger.error(f"Error preparing PDF for Claude: {str(e)}")
            raise
    
    def create_message_with_file(self, file_data, prompt):
        """Create a message with file attachment and prompt"""
        if file_data is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": file_data
                        },
                        {
                            "type": "text",
                            "text": prompt
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
    
    def generate_response(self, model_name, messages, temperature=0.2, max_tokens=2048, system_instruction=None):
        """Generate response from Claude model"""
        try:
            self.logger.info(f"Sending request to Claude model: {model_name}")
            
            # Prepare request parameters
            params = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            if system_instruction:
                params["system"] = system_instruction
                
            # Send request to Claude API
            response = self.client.messages.create(**params)
            
            self.logger.info("Successfully received response from Claude")
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response from Claude: {str(e)}")
            raise
    
    def apply_response_schema(self, prompt, schema):
        """Apply response schema by incorporating it into the prompt"""
        # Claude doesn't have a native schema format, so we include it in the prompt
        schema_str = str(schema)
        enhanced_prompt = f"{prompt}\n\nPlease format your response according to this JSON schema:\n{schema_str}"
        return enhanced_prompt