import logging
from google.genai import types
from api_utils.config import init_gemini_client
from api_utils.pdf_utils import validate_pdf

class GeminiAPI:
    """Interface for interacting with Google's Gemini API"""
    
    def __init__(self, logger=None):
        self.client = init_gemini_client()
        self.logger = logger or logging.getLogger("gemini_api")
        self.contents = None
        self.config = None
        self.model_name = "gemini-2.0-flash"
        
    def upload_pdf(self, file_path):
        """Upload a PDF file to Gemini"""
        try:
            validate_pdf(file_path)
            self.logger.info(f"Uploading PDF to Gemini: {file_path}")
            
            # File upload for Gemini                
            file_obj = self.client.files.upload(file=file_path)
            
            self.logger.info(f"Successfully uploaded PDF: {file_path}")
            return file_obj
            
        except Exception as e:
            self.logger.error(f"Error uploading PDF to Gemini: {str(e)}")
            raise

    def set_model(self, model_name):
        """Set the model name to use"""
        self.model_name = model_name
        return self
    
    def set_contents(self, file_obj, prompt):
        """Set content parts combining file and prompt"""
        try:
            # Use proper file reference format for Gemini
            if file_obj is None:
                self.contents = [
                    {"text": prompt}
                ]
            else:
                self.contents = [
                    {"text": prompt},
                    {"file_data": {"file_uri": file_obj.uri}}
                ]
            return self
        except Exception as e:
            self.logger.error(f"Error setting contents: {str(e)}")
            raise
    
    def set_config(self, temperature=0.1, top_p=0.95, top_k=20, max_tokens=2048):
        """Set generation config for Gemini"""
        try:
            self.config = types.GenerateContentConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_output_tokens=max_tokens,
                response_mime_type="application/json"
            )
            return self
        except Exception as e:
            self.logger.error(f"Error setting config: {str(e)}")
            raise

    def set_system_instruction(self, system_instruction):
        """Set system instruction for Gemini"""
        try:
            self.config.system_instruction = system_instruction
            return self
        except Exception as e:
            self.logger.error(f"Error setting system instruction: {str(e)}")
            raise

    def set_response_schema(self, schema):
        """Set response schema for structured output"""
        try:
            if self.config is None:
                self.set_config()  # Use default config if not set
            
            # For Gemini, we add response_schema to the config object
            self.config.response_schema = schema
            return self
        except Exception as e:
            self.logger.error(f"Error setting response schema: {str(e)}")
            raise
    
    def generate_response(self):
        """Generate response from Gemini model using the class attributes"""
        try:
            if self.contents is None:
                raise ValueError("Contents not set. Call set_contents() first.")
            
            if self.config is None:
                self.set_config()  # Use default config if not set
            
            self.logger.info(f"Sending request to Gemini model: {self.model_name}")
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=self.contents,
                config=self.config
            )
            
            self.logger.info("Successfully received response from Gemini")
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response from Gemini: {str(e)}")
            raise