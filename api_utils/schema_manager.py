import json
import jsonschema
import logging

class ResponseSchemaManager:
    """Manage response schemas for different LLM APIs"""
    
    def __init__(self):
        # Common response schema template that can be customized
        self.default_schema = {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Summary of the content"
                },
                "evaluation": {
                    "type": "string",
                    "description": "Evaluation of the content"
                },
                "suggestion": {
                    "type": "string",
                    "description": "Suggestions for improvement"
                }
            },
            "required": ["summary", "evaluation", "suggestion"]
        }
    
    def get_gemini_schema(self, custom_schema=None):
        """Get schema formatted for Gemini API"""
        schema = custom_schema if custom_schema else self.default_schema
        return schema
    
    def get_claude_schema(self, custom_schema=None):
        """Get schema formatted for Claude API"""
        schema = custom_schema if custom_schema else self.default_schema
        # Claude might have specific formatting requirements
        return schema
    
    def get_openai_schema(self, custom_schema=None):
        """Get schema formatted for OpenAI API"""
        schema = custom_schema if custom_schema else self.default_schema
            
        # Create a properly formatted schema for OpenAI API
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "OutputSchema",
                "schema": schema
            }
        }
    
    def validate_response(self, response_data, schema):
        """Validate a response against a schema"""
        try:
            jsonschema.validate(instance=response_data, schema=schema)
            return True
        except jsonschema.exceptions.ValidationError as e:
            logging.error(f"Schema validation failed: {str(e)}")
            return False
    
    def parse_response(self, response_text, api_name):
        """Parse response text to extract structured data"""
        try:
            # First try to parse as JSON
            return json.loads(response_text)
        except json.JSONDecodeError:
            logging.warning(f"Response from {api_name} is not valid JSON. Returning raw text.")
            # Return raw text in a structured format
            return {
                "raw_text": response_text,
                "structured": False
            }