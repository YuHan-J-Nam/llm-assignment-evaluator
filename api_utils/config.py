import os
import logging
from dotenv import load_dotenv
from google import genai
import anthropic
import openai

# .env 파일에서 환경 변수 불러오기
load_dotenv()

def get_api_key(api_name):
    """환경 변수에서 API 키를 가져옵니다"""
    key_name = f"{api_name.upper()}_API_KEY"
    api_key = os.getenv(key_name)
    if not api_key:
        logging.error(f"{key_name}가 환경 변수에 없습니다")
        raise ValueError(f"{api_name}의 API 키가 없습니다")
    return api_key

def init_gemini_client():
    """Gemini 클라이언트를 초기화합니다"""
    api_key = get_api_key("google")
    try:
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        logging.error(f"Gemini 클라이언트 초기화 실패: {str(e)}")
        raise

def init_anthropic_client():
    """Anthropic 클라이언트를 초기화합니다"""
    api_key = get_api_key("anthropic")
    try:
        client = anthropic.Anthropic(api_key=api_key)
        return client
    except Exception as e:
        logging.error(f"Anthropic 클라이언트 초기화 실패: {str(e)}")
        raise

def init_openai_client():
    """OpenAI 클라이언트를 초기화합니다"""
    api_key = get_api_key("openai")
    try:
        # 호환 모드로 openai 설정
        client = openai.Client(api_key=api_key)
        return client
    except Exception as e:
        logging.error(f"OpenAI 클라이언트 초기화 실패: {str(e)}")
        raise