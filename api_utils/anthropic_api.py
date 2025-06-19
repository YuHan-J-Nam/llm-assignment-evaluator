import logging
import time
import os
from typing import Optional, Dict, List, Any
from api_utils.config import init_anthropic_client
from api_utils.pdf_utils import validate_pdf, encode_pdf_base64

class AnthropicAPI:
    """Anthropic API와 상호작용하기 위한 인터페이스"""
    
    def __init__(self, logger=None):
        self.client = init_anthropic_client()
        self.logger = logger or logging.getLogger("anthropic_api")

    def upload_pdf(self, file_path:str) -> str:
        """Anthropic의 File API에 PDF를 업로드합니다."""
        try:
            validate_pdf(file_path)
            
            with open(file_path, "rb") as file:
                uploaded_file = self.client.beta.files.upload(
                    file=(file_path, file, "application/pdf"),
                )
            
            self.logger.info(f"파일 업로드 성공. 파일 ID: {uploaded_file.id}")
            return uploaded_file
            
        except Exception as e:
            self.logger.error(f"파일 업로드 실패 {file_path}: {str(e)}")
            raise
    
    def prepare_pdf(self, file_path: str) -> Dict[str, Any]:
        """Anthropic API용 PDF 파일을 준비합니다"""
        try:
            validate_pdf(file_path)
            self.logger.info(f"Anthropic용 PDF 준비 중: {file_path}")
            
            # Anthropic은 base64로 인코딩된 파일을 허용합니다
            pdf_data = encode_pdf_base64(file_path)
            
            # 파일명 추출
            file_name = os.path.basename(file_path)
            
            media_data = {
                "type": "base64",
                "media_type": "application/pdf",
                "data": pdf_data
            }  
            
            self.logger.info(f"Anthropic 요청용 PDF 준비 완료: {file_name}")
            return media_data
            
        except Exception as e:
            self.logger.error(f"Anthropic 요청용 PDF 준비 오류: {str(e)}")
            raise
    
    def create_input_message(self, prompt: str, file_id: Optional[str] = None, file_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """입력 메시지를 생성합니다. 필요할 경우 첨부파일을 포함합니다.
        Args:
            prompt (str): 사용자 입력 프롬프트
            file_id (str, optional): 업로드된 파일의 ID. None인 경우 첨부파일 없음.
            file_data (dict, optional): 업로드된 파일의 base64 인코딩 데이터. None인 경우 첨부파일 없음."""
        content = [{"type": "text", "text": prompt}]
        
        if file_data:
            content.insert(0,{
                "type": "document",
                "source": file_data
            })

        elif file_id:
            content.insert(0,{
                "type": "document",
                "source": {
                    "type": "file",
                    "file_id": file_id
                }
            })
        
        return [{"role": "user", "content": content}]
    
    def generate_response(
        self,
        input: List[Dict[str, Any]],
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        system_instruction: Optional[str] = None,
        response_schema: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Anthropic 모델로부터 응답을 생성합니다.

        Args:
            input (list): 입력 메시지
            model (str): 사용할 모델 이름
            temperature (float): 생성 다양성 제어
            max_tokens (int): 최대 토큰 수
            system_instruction (str, optional): 시스템 메시지
            response_schema (dict, optional): 응답 스키마

        Returns:
            response: Anthropic API의 응답
        """
        try:
            self.logger.info(f"Anthropic 모델에 요청 전송 중: {model}")
            
            # API 호출 파라미터 생성
            params = {
                "model": model,
                "messages": input,
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            # 시스템 지시사항이 제공된 경우 추가
            if system_instruction:
                # 응답 스키마가 제공된 경우 시스템 지시사항에 적용
                if response_schema:
                    system_instruction = self.apply_response_schema(system_instruction, response_schema)
                params["system"] = [{
                    "type": "text",
                    "text": system_instruction,
                    "cache_control": {"type": "ephemeral"}
                }]

            # 응답 시간 측정을 위한 타이머 시작
            start_time = time.time()
                
            # Anthropic API에 요청 전송
            response = self.client.messages.create(**params)
            
            # 응답 시간 계산
            end_time = time.time()
            response_time = end_time - start_time

            # 응답 객체에 response_time 속성 추가
            response.response_time = response_time
            
            self.logger.info(f"Anthropic 모델로부터 응답 수신 성공: {model}, 응답 시간: {response_time:.2f}초")
            return response
            
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            self.logger.error(f"Anthropic의 {model}에서 응답 생성 오류 (응답 시간: {response_time:.2f}초): {str(e)}")
            raise
    
    def apply_response_schema(self, instruction: str, schema: Dict[str, Any]) -> str:
        """응답 스키마를 지시사항에 통합하여 적용합니다"""
        # Anthropic는 별도의 응답 스키마를 지원하지 않으므로, 지시사항에 스키마를 포함시킵니다.
        enhanced_instruction = f"""{instruction}

다음 JSON 형식으로만 응답하라:

```json
{schema}
```

반드시 위의 JSON 구조를 정확히 따르고, 추가적인 텍스트나 설명은 포함하지 마라."""
        return enhanced_instruction