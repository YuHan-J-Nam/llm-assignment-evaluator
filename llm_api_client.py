import logging
from datetime import datetime
from typing import Optional, Dict, List, Any
from api_utils.logging_utils import setup_logging
from api_utils.schema_manager import ResponseSchemaManager
from api_utils.gemini_api import GeminiAPI
from api_utils.claude_api import ClaudeAPI
from api_utils.openai_api import OpenAIAPI
from api_utils.token_counter import TokenCounter

class LLMAPIClient:
    """Unified interface for interacting with multiple LLM APIs"""
    
    def __init__(self, log_level=logging.INFO):
        self.logger = setup_logging(log_level)
        self.schema_manager = ResponseSchemaManager()
        
        # Initialize API clients
        self.gemini = GeminiAPI(self.logger)
        self.claude = ClaudeAPI(self.logger)
        self.openai = OpenAIAPI(self.logger)
        
        # Initialize token counter
        self.token_counter = TokenCounter(self.logger)
        
        self.logger.info("LLM API Client initialized")
    
    def process_pdf_with_gemini(self, file_path, prompt, model_name="gemini-2.0-flash", 
                              temperature=0.1, top_p=0.95, max_tokens=4096, 
                              system_instruction=None, schema=None):
        """Retrieve a response from Gemini API"""
        try:
            if file_path is not None:
                self.logger.info(f"Processing with Gemini (with PDF): {file_path}")
            
                # Upload PDF
                file_obj = self.gemini.upload_pdf(file_path)

            else:
                self.logger.info(f"Processing with Gemini:")
                file_obj = None  # No file to be attached
            
            # Set model, contents, and config using the builder pattern
            self.gemini.set_model(model_name)
            self.gemini.set_contents(file_obj, prompt)
            self.gemini.set_config(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens
            )
            
            # Apply schema if provided
            if schema:
                response_schema = self.schema_manager.format_gemini_schema(schema)
                self.gemini.set_response_schema(response_schema)
            
            # Apply system instruction if provided
            if system_instruction:
                self.gemini.set_system_instruction(system_instruction)
            
            # Generate response
            response = self.gemini.generate_response()
            
            # Extract token usage from Gemini response
            token_usage = self.token_counter.extract_token_usage_from_gemini_response(response, model_name)
            
            # Log the token usage
            self.token_counter.log_token_usage(
                model_name=model_name,
                prompt_tokens=token_usage["prompt_tokens"],
                completion_tokens=token_usage["completion_tokens"],
                total_tokens=token_usage["total_tokens"]
            )
                
            self.logger.info("Successfully retrieved response from Gemini")
            return response
            
        except Exception as e:
            self.logger.error(f"Error retrieving response from Gemini: {str(e)}")
            raise
    
    def process_pdf_with_claude(self, file_path, prompt, model_name="claude-3-sonnet-20240229", 
                              temperature=0.2, max_tokens=4096, 
                              system_instruction=None, schema=None):
        """Process a PDF document with Claude API"""
        try:
            if file_path is not None:
                self.logger.info(f"Processing with Claude (with PDF): {file_path}")

                # Prepare PDF
                file_data = self.claude.upload_pdf(file_path)

            else:
                self.logger.info(f"Processing with Claude:")
                file_data = None  # No file to be attached
            
            # Apply schema if provided
            if schema:
                response_schema = self.schema_manager.format_claude_schema(schema)
                prompt = self.claude.apply_response_schema(prompt, response_schema)
            
            # Create message with file
            messages = self.claude.create_message_with_file(file_data, prompt)
            
            # Generate response
            response = self.claude.generate_response(
                model_name=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                system_instruction=system_instruction
            )
            
            # Extract token usage from Claude response
            token_usage = self.token_counter.extract_token_usage_from_claude_response(response, model_name)
            
            # Log the token usage
            self.token_counter.log_token_usage(
                model_name=model_name,
                prompt_tokens=token_usage["prompt_tokens"],
                completion_tokens=token_usage["completion_tokens"],
                total_tokens=token_usage["total_tokens"]
            )
            
            self.logger.info("Successfully retrieved response from Claude")
            return response
            
        except Exception as e:
            self.logger.error(f"Error retrieving response from Claude: {str(e)}")
            raise
    def generate_openai_response(self, file_path, prompt, model_name="gpt-4", 
                              temperature=0.2, max_tokens=4096, 
                              system_instruction=None, schema=None):
        """Process a PDF document with OpenAI API"""
        try:
            if file_path is not None:
                self.logger.info(f"Processing with OpenAI (with PDF): {file_path}")
            
                # Upload PDF and get file ID
                file_id = self.openai.upload_pdf(file_path)

            else:
                self.logger.info(f"Processing with OpenAI:")
                file_id = None  # No file to be attached
            
            # Create input message with file
            input_messages = self.openai.create_input_message(prompt, file_id)
            
            # Apply schema if provided
            response_schema = None
            if schema:
                response_schema = self.schema_manager.format_openai_schema(schema)
            
            # Generate response
            response = self.openai.generate_response(
                input=input_messages,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                system_instruction=system_instruction,
                response_schema=response_schema
            )
            
            # Extract token usage from OpenAI response
            token_usage = self.token_counter.extract_token_usage_from_openai_response(response, model_name)
            
            # Log the token usage
            self.token_counter.log_token_usage(
                model_name=model_name,
                prompt_tokens=token_usage["prompt_tokens"],
                completion_tokens=token_usage["completion_tokens"],
                total_tokens=token_usage["total_tokens"]
            )
            
            self.logger.info("Successfully retrieved response from OpenAI")
            return response
            
        except Exception as e:
            self.logger.error(f"Error retrieving response from OpenAI: {str(e)}")
            raise
            
    def get_token_usage_summary(self):
        """Get a summary of token usage across all API calls"""
        return self.token_counter.get_usage_summary()