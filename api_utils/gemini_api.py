import logging
import time
import os
from typing import Optional, Dict, List, Any
from google.genai import types
from api_utils.config import init_gemini_client
from api_utils.pdf_utils import validate_pdf, encode_pdf_base64
from api_utils.token_counter import TokenCounter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class GeminiAPI:
    """Google Gemini API와 상호작용하기 위한 인터페이스"""
    
    def __init__(self, logger=None):
        self.client = init_gemini_client()
        self.logger = logger or logging.getLogger("gemini_api")
        self.token_counter = TokenCounter()

    # --------------------------------------------------------
    # PDF 파일 업로드 관련 메서드
    # --------------------------------------------------------

    def upload_pdf(self, file_path: str) -> Any:
        """Google의 File API에 PDF를 업로드합니다."""
        try:
            validate_pdf(file_path)
            file_name = os.path.basename(file_path)
            
            self.logger.info(f"Gemini용 PDF 업로드 중: {file_name}")
            
            # Gemini File API에 파일 업로드
            uploaded_file = self.client.files.upload(file=file_path)
            
            self.logger.info(f"파일 업로드 성공 {file_name}")
            return uploaded_file
            
        except Exception as e:
            self.logger.error(f"파일 업로드 실패 {file_name}: {str(e)}")
            raise
    
    def prepare_pdf(self, file_path: str) -> Dict[str, Any]:
        """Gemini API용 PDF 파일을 준비합니다"""
        try:
            validate_pdf(file_path)
            file_name = os.path.basename(file_path)
            
            self.logger.info(f"Gemini용 PDF 준비 중: {file_path}")
            
            # PDF를 base64로 인코딩
            pdf_data = encode_pdf_base64(file_path)
                       
            file_data = {
                "mime_type": "application/pdf",
                "data": pdf_data
            }
            
            self.logger.info(f"Gemini 요청용 PDF 준비 완료: {file_name}")
            return file_data
            
        except Exception as e:
            self.logger.error(f"Gemini 요청용 PDF {file_name} 준비 오류: {str(e)}")
            raise

    # --------------------------------------------------------
    # Gemini API의 메시지 생성 및 응답 관련 메서드
    # --------------------------------------------------------

    def create_input_message(self, prompt: str, file: Optional[Any] = None, file_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """입력 메시지를 생성합니다. 필요할 경우 첨부파일을 포함합니다.
        Args:
            prompt (str): 사용자 입력 프롬프트
            file (Any, optional): 첨부할 파일 객체. None인 경우 첨부파일 없음.
            file_data (dict, optional): 업로드된 파일 데이터. None인 경우 첨부파일 없음."""
        
        content = []
        
        # 파일을 먼저 추가 (문서에 따르면 프롬프트 전에 파일을 배치)
        if file_data and "data" in file_data:
            content.append(
                types.Part.from_bytes(
                    data=file_data["data"],
                    mime_type=file_data["mime_type"]
                )
            )
        elif file:
            content.append(file)
        
        # 텍스트 프롬프트 추가
        content.append(prompt)
        
        return content
    
    def create_response_params(
        self,
        input: List[Dict[str, Any]],
        model: str = "gemini-2.0-flash-lite",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        enable_thinking: bool = False,
        thinking_budget: Optional[int] = None,
        top_k: int = 20,
        system_instruction: Optional[str] = None,
        response_schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Gemini API 요청을 위한 파라미터를 생성합니다.

        Args:
            input (list): 입력 메시지
            model (str): 사용할 모델 이름
            temperature (float): 생성 다양성 제어
            max_tokens (int): 최대 토큰 수
            enable_thinking (bool): 사고 기능 활성화 여부
            thinking_budget (int, optional): 사고 토큰 예산
            top_k (int): top-k sampling 파라미터
            system_instruction (str, optional): 시스템 메시지
            response_schema (dict, optional): 응답 스키마

        Returns:
            params: API 요청 파라미터
        """
        # Gemini 생성 설정 구성
        config = types.GenerateContentConfig(
            temperature=temperature,
            top_k=top_k,
            max_output_tokens=max_tokens,
            response_mime_type="application/json"
        )

        # 사고 모델별 사고 예산 설정
        if any(name in model for name in ['gemini-2.5-flash-lite', 'gemini-2.5-flash']):
            if not enable_thinking:
                thinking_budget = 0  # 사고 비활성화
            elif thinking_budget is None:
                thinking_budget = -1  # Dynamic Thinking 활성화
        elif any(name in model for name in ['gemini-2.5-pro']):
            # Pro 모델의 경우 사고 비활성화가 불가능
            if thinking_budget is None:
                thinking_budget = -1  # Dynamic Thinking 활성화

        # 사고가 가능한 모델일 경우 사고 예산 설정
        if any(version in model for version in ['2.5']):
            config.thinking_config = types.ThinkingConfig(
                include_thoughts=True if enable_thinking else None,
                thinking_budget=thinking_budget
            )

        # 시스템 지시사항이 제공된 경우 추가
        if system_instruction:
            config.system_instruction = system_instruction

        # 응답 스키마가 제공된 경우 설정에 추가
        if response_schema:
            config.response_schema = response_schema

        params = {
            "model": model,
            "contents": input,
            "config": config
        }
        
        return params
    
    def generate_response(
        self,
        input: List[Dict[str, Any]],
        model: str = "gemini-2.0-flash-lite",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        enable_thinking: bool = False,
        thinking_budget: Optional[int] = None,
        top_k: int = 20,
        system_instruction: Optional[str] = None,
        response_schema: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Gemini 모델로부터 응답을 생성합니다.

        Args:
            input (list): 입력 메시지
            model (str): 사용할 모델 이름
            temperature (float): 생성 다양성 제어
            max_tokens (int): 최대 토큰 수
            enable_thinking (bool): 사고 기능 활성화 여부
            thinking_budget (int, optional): 사고 토큰 예산
            top_k (int): top-k sampling 파라미터
            system_instruction (str, optional): 시스템 메시지
            response_schema (dict, optional): 응답 스키마

        Returns:
            dict: Gemini 모델의 응답 및 메타데이터
        """
        try:
            self.logger.info(f"Gemini 모델에 요청 전송 중: {model}")
            
            # API 호출 파라미터 생성
            params = self.create_response_params(
                input=input,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
                thinking_budget=thinking_budget,
                top_k=top_k,
                system_instruction=system_instruction,
                response_schema=response_schema
            )

            # 응답 시간 측정을 위한 타이머 시작
            start_time = time.time()
                
            # Gemini API에 요청 전송
            response = self.client.models.generate_content(**params)
            
            # 응답 시간 계산
            end_time = time.time()
            response_time = end_time - start_time
            
            self.logger.info(f"Gemini 모델로부터 응답 수신 성공: {model}, 응답 시간: {response_time:.2f}초")
            return {
                "metadata": {
                    "model": model,
                    "response_time": response_time,
                },
                "response": response
            }
            
        except Exception as e:
            self.logger.error(f"Gemini의 {model}에서 응답 생성 오류: {str(e)}")
            self.logger.debug(f"Gemini API 요청 파라미터: {params}", exc_info=True)
            raise

    def parse_response(self, response) -> Dict[str, Any]:
        """
        Gemini API 응답을 파싱하여 텍스트와 토큰 사용량을 추출
        
        Args:
            response: Gemini API 응답 객체 또는 딕셔너리
            
        Returns:
            파싱된 텍스트와 토큰 사용량이 담긴 딕셔너리
        """
        try:
            # 딕셔너리 형태인 경우
            if isinstance(response, dict):
                text = response.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
                token_usage = response.get('usage_metadata', {})
                
            # 객체 형태인 경우
            else:
                text = response.candidates[0].content.parts[0].text if response.candidates[0].content.parts else ''
                token_usage = response.usage_metadata if hasattr(response, 'usage_metadata') else {}
            
            return {"text": text, "token_usage": token_usage}
            
        except Exception as e:
            self.logger.error(f"Gemini 응답 파싱 중 오류 발생: {str(e)}")
            return {"text": "", "token_usage": {}}