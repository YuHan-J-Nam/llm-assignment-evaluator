import logging
import os
from typing import Optional, Dict, List, Any
from api_utils.config import init_openai_client
from api_utils.pdf_utils import validate_pdf

class OpenAIAPI:
    """OpenAI API와 상호작용하기 위한 인터페이스"""

    def __init__(self, logger=None):
        self.client = init_openai_client()
        self.logger = logger or logging.getLogger("openai_api")

    def upload_pdf(self, file_path:str) -> str:
        """OpenAI의 File API에 PDF를 업로드합니다."""
        try:
            validate_pdf(file_path)
            
            with open(file_path, "rb") as file:
                uploaded_file = self.client.files.create(
                    file=file,
                    purpose="user_data"
                )
            
            self.logger.info(f"파일 업로드 성공. 파일 ID: {uploaded_file.id}")
            return uploaded_file.id
            
        except Exception as e:
            self.logger.error(f"Upload failed for {file_path}: {str(e)}")
            raise

    def create_input_message(self, prompt: str, file_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """입력 메시지를 생성합니다. 필요할 경우 파일 첨부를 포함합니다.
        Args:
            prompt (str): 사용자 입력 프롬프트
            file_id (str, optional): 업로드된 파일의 ID. None인 경우 파일 첨부 없음."""
        content = [{"type": "input_text", "text": prompt}]
        
        if file_id:
            content.append({
                "type": "input_file",
                "file_id": file_id
            })
        
        return [{"role": "user", "content": content}]

    def generate_response(
        self, 
        input: List[Dict[str, Any]], 
        model_name: str = "gpt-4o-mini", 
        temperature: float = 0.2,
        max_tokens: int = 2048,
        system_instruction: Optional[str] = None,
        response_schema: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        OpenAI 모델로부터 응답을 생성합니다.

        Args:
            model_name (str): 사용할 모델 이름
            input (list): 입력 메시지
            temperature (float): 생성 다양성 제어
            max_tokens (int): 최대 토큰 수
            system_instruction (str, optional): 시스템 메시지
            response_schema (dict, optional): 응답 스키마

        Returns:
            response: OpenAI API의 응답
        """
        try:
            self.logger.info(f"OpenAI 모델에 요청 전송 중: {model_name}")

            # API 호출 파라미터 생성
            params = {
                "model": model_name,
                "input": input,
                "temperature": temperature,
                "max_output_tokens": max_tokens
            }

            # 시스템 메시지가 제공된 경우 요청에 추가
            if system_instruction:
                params["instructions"] = system_instruction

            # 스키마가 제공된 경우 요청에 추가
            if response_schema:
                params["text"] = response_schema

            # OpenAI API에 요청 전송
            response = self.client.responses.create(**params)

            self.logger.info(f"OpenAI 모델로부터 응답 수신 성공: {model_name}")
            return response

        except Exception as e:
            self.logger.error(f"OpenAI의 {model_name}에서 응답 생성 오류: {str(e)}")
            raise