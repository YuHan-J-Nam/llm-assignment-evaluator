import os
import logging
from dotenv import load_dotenv
from google import genai
import anthropic
import openai

# Load environment variables from .env file
load_dotenv()

def get_api_key(api_name):
    """Retrieve API key from environment variables"""
    key_name = f"{api_name.upper()}_API_KEY"
    api_key = os.getenv(key_name)
    if not api_key:
        logging.error(f"{key_name} not found in environment variables")
        raise ValueError(f"Missing API key for {api_name}")
    return api_key

def init_gemini_client():
    """Initialize and return an authenticated Gemini client"""
    api_key = get_api_key("gemini")
    try:
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        logging.error(f"Failed to initialize Gemini client: {str(e)}")
        raise

def init_claude_client():
    """Initialize and return an authenticated Claude client"""
    api_key = get_api_key("claude")
    try:
        client = anthropic.Anthropic(api_key=api_key)
        return client
    except Exception as e:
        logging.error(f"Failed to initialize Claude client: {str(e)}")
        raise

def init_openai_client():
    """Initialize and return an authenticated OpenAI client"""
    api_key = get_api_key("openai")
    try:
        # Setup openai with compatibility mode
        client = openai.Client(api_key=api_key, base_url="https://api.openai.com/v1")
        return client
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI client: {str(e)}")
        raise