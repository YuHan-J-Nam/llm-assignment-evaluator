import logging
import time
import os
import json
import re
from typing import Optional, Dict, List, Any, Tuple, Iterator
from api_utils.config import init_anthropic_client
from api_utils.pdf_utils import validate_pdf, encode_pdf_base64

class AnthropicAPI:
    """Anthropic API와 상호작용하기 위한 인터페이스"""
    
    def __init__(self, logger=None):
        self.client = init_anthropic_client()
        self.logger = logger or logging.getLogger("anthropic_api")

    # --------------------------------------------------------
    # PDF 파일 업로드 관련 메서드
    # --------------------------------------------------------

    def upload_pdf(self, file_path:str) -> str:
        """Anthropic의 File API에 PDF를 업로드합니다."""
        try:
            validate_pdf(file_path)
            file_name = os.path.basename(file_path)

            with open(file_path, "rb") as file:
                uploaded_file = self.client.files.upload(
                    file=(file_path, file, "application/pdf"),
                )
            
            self.logger.info(f"파일 {file_name} 업로드 성공. 파일 ID: {uploaded_file.id}")
            return uploaded_file
            
        except Exception as e:
            self.logger.error(f"파일 업로드 실패 {file_name}: {str(e)}")
            raise
    
    def prepare_pdf(self, file_path: str) -> Dict[str, Any]:
        """Anthropic API용 PDF 파일을 준비합니다"""
        try:
            validate_pdf(file_path)
            file_name = os.path.basename(file_path)

            self.logger.info(f"Anthropic용 PDF 준비 중: {file_path}")
            
            # Anthropic은 base64로 인코딩된 파일을 허용합니다
            pdf_data = encode_pdf_base64(file_path)
            
            media_data = {
                "type": "base64",
                "media_type": "application/pdf",
                "data": pdf_data
            }  
            
            self.logger.info(f"Anthropic 요청용 PDF 준비 완료: {file_name}")
            return media_data
            
        except Exception as e:
            self.logger.error(f"Anthropic 요청용 PDF {file_name} 준비 오류: {str(e)}")
            raise
    
    # --------------------------------------------------------
    # Anthropic API의 메시지 생성 및 응답 관련 메서드
    # --------------------------------------------------------

    def create_input_message(self, prompt: str, file_id: Optional[str] = None, file_data: Optional[Dict[str, Any]] = None, cache: bool = False) -> List[Dict[str, Any]]:
        """입력 메시지를 생성합니다. 필요할 경우 첨부파일을 포함합니다.
        Args:
            prompt (str): 사용자 입력 프롬프트
            file_id (str, optional): 업로드된 파일 ID. None인 경우 첨부파일 없음.
            file_data (dict, optional): 업로드된 파일 데이터. None인 경우 첨부파일 없음.
            cache (bool): 캐시 제어 여부. True인 경우 ephemeral 캐시 사용."""
        
        content = [{"type": "text", "text": prompt}]
        if cache:
            content[0]['cache_control'] = {"type": "ephemeral"}

        if file_id:
            content.insert(0, {
                "type": "document",
                "source": {
                    "type": "file",
                    "file_id": file_id
                }
            })

        elif file_data:
            content.insert(0, {
                "type": "document",
                "source": file_data
            })
        
        return [{"role": "user", "content": content}]
    
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
    
    def create_response_params(
        self,
        input: List[Dict[str, Any]],
        model: str = "claude-3-5-haiku-20241022",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        enable_thinking: bool = False,
        thinking_budget: Optional[int] = 5000,
        system_instruction: Optional[str] = None,
        response_schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Anthropic API 요청을 위한 파라미터를 생성합니다.

        Args:
            input (list): 입력 메시지
            model (str): 사용할 모델 이름
            temperature (float): 생성 다양성 제어
            max_tokens (int): 최대 토큰 수
            enable_thinking (bool): 사고 기능 활성화 여부
            thinking_budget (int, optional): 사고 토큰 예산
            system_instruction (str, optional): 시스템 메시지
            response_schema (dict, optional): 응답 스키마

        Returns:
            params: API 요청 파라미터
        """
        params = {
            "model": model,
            "messages": input,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        # 시스템 지시사항이 제공된 경우 추가
        if system_instruction:
            # 응답 스키마가 제공된 경우 시스템 지시사항에 적용 (Anthropic는 별도의 응답 스키마를 지원하지 않음)
            if response_schema:
                system_instruction = self.apply_response_schema(system_instruction, response_schema)
            params["system"] = [{
                "type": "text",
                "text": system_instruction,
                "cache_control": {"type": "ephemeral"}
            }]

        # 사고 기능이 활성화된 경우 추가
        if enable_thinking:
            if thinking_budget is None:
                raise ValueError("사고 기능이 활성화된 경우 thinking_budget를 지정해야 합니다.")
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget
            }

        return params
    
    def generate_response(
        self,
        input: List[Dict[str, Any]],
        model: str = "claude-3-5-haiku-20241022",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        enable_thinking: bool = False,
        thinking_budget: Optional[int] = 5000,
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
            enable_thinking (bool): 사고 기능 활성화 여부
            thinking_budget (int, optional): 사고 토큰 예산
            system_instruction (str, optional): 시스템 메시지
            response_schema (dict, optional): 응답 스키마

        Returns:
            response: Anthropic API의 응답
        """
        try:
            self.logger.info(f"Anthropic 모델에 요청 전송 중: {model}")
            
            # API 호출 파라미터 생성
            params = self.create_response_params(
                input=input,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
                thinking_budget=thinking_budget,
                system_instruction=system_instruction,
                response_schema=response_schema
            )

            # 응답 시간 측정을 위한 타이머 시작
            start_time = time.time()
                
            # Anthropic API에 요청 전송
            response = self.client.messages.create(**params)
            
            # 응답 시간 계산
            end_time = time.time()
            response_time = end_time - start_time
            
            self.logger.info(f"Anthropic 모델로부터 응답 수신 성공: {model}, 응답 시간: {response_time:.2f}초")
            return response, response_time
            
        except Exception as e:
            self.logger.error(f"Anthropic의 {model}에서 응답 생성 오류: {str(e)}")
            self.logger.debug(f"Anthropic API 요청 파라미터: {params}", exc_info=True)
            raise

    # --------------------------------------------------------
    # Anthropic API의 배치 요청을 생성하고 관리하는 메서드
    # --------------------------------------------------------

    def create_batch_request(
        self, 
        prompts: List[str], 
        model: str = "claude-3-5-haiku-20241022",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        system_instruction: Optional[str] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        custom_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        여러 프롬프트를 일괄 처리하기 위한 배치 요청을 생성합니다.

        Args:
            prompts (List[str]): 처리할 사용자 프롬프트 목록
            model (str): 사용할 모델 이름
            temperature (float): 생성 다양성 제어
            max_tokens (int): 최대 토큰 수
            system_instruction (str, optional): 모든 요청에 공통으로 적용할 시스템 지시사항
            response_schema (dict, optional): 응답 스키마
            custom_ids (List[str], optional): 각 요청에 대한 사용자 정의 ID 목록. 제공되지 않으면 자동 생성됨.

        Returns:
            Dict[str, Any]: 생성된 배치 요청의 상세 정보
        """
        try:
            self.logger.info(f"Anthropic 배치 요청 생성 중: {len(prompts)}개의 프롬프트, 모델: {model}")
            
            # 시스템 지시사항이 제공된 경우 응답 스키마 적용
            if system_instruction and response_schema:
                system_instruction = self.apply_response_schema(system_instruction, response_schema)
            
            # 사용자 정의 ID가 제공되지 않은 경우 자동 생성
            if not custom_ids:
                custom_ids = [f"request_{i}" for i in range(len(prompts))]
            elif len(custom_ids) != len(prompts):
                raise ValueError("custom_ids 길이는 prompts 길이와 일치해야 합니다.")
            
            # 배치 요청 준비
            requests = []
            for i, prompt in enumerate(prompts):
                # 메시지 구성
                messages = self.create_input_message(prompt)
                
                # 요청 구성
                params = self.create_response_params(
                    input=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_instruction=system_instruction,
                    response_schema=response_schema
                )
                
                # 배치 요청 항목 추가
                requests.append({
                    "custom_id": custom_ids[i],
                    "params": params
                })
            
            # 배치 요청 전송
            batch_response = self.client.messages.batches.create(
                requests=requests
            )
            
            self.logger.info(f"Anthropic 배치 요청 생성 성공: 배치 ID: {batch_response.id}, 요청 수: {len(prompts)}")
            return batch_response
            
        except Exception as e:
            self.logger.error(f"Anthropic 배치 요청 생성 실패: {str(e)}")
            raise
    
    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        배치 요청의 현재 상태를 확인합니다.

        Args:
            batch_id (str): 조회할 배치 ID

        Returns:
            Dict[str, Any]: 배치 상태 정보
        """
        try:
            self.logger.info(f"Anthropic 배치 상태 확인 중: {batch_id}")
            
            batch_info = self.client.messages.batches.retrieve(
                message_batch_id=batch_id
            )
            
            status = {
                "id": batch_info.id,
                "processing_status": batch_info.processing_status,
                "request_counts": batch_info.request_counts,
                "created_at": batch_info.created_at,
                "ended_at": batch_info.ended_at,
                "results_url": batch_info.results_url
            }
            
            self.logger.info(f"Anthropic 배치 상태: {batch_id}, 처리 상태: {batch_info.processing_status}")
            return status
            
        except Exception as e:
            self.logger.error(f"Anthropic 배치 상태 조회 실패: {batch_id}, 오류: {str(e)}")
            raise
    
    def get_batch_results(self, batch_id: str, wait_for_completion: bool = False, poll_interval: int = 60) -> Iterator[Dict[str, Any]]:
        """
        배치 요청의 결과를 조회합니다. 필요시 완료까지 대기합니다.

        Args:
            batch_id (str): 조회할 배치 ID
            wait_for_completion (bool): 배치 처리가 완료될 때까지 대기할지 여부
            poll_interval (int): 대기 중 폴링 간격(초)

        Yields:
            Dict[str, Any]: 각 요청에 대한 결과
        """
        try:
            # 배치 상태 초기 확인
            batch_info = self.get_batch_status(batch_id)
            
            # 완료될 때까지 대기
            if wait_for_completion and batch_info["processing_status"] != "ended":
                self.logger.info(f"Anthropic 배치 완료 대기 중: {batch_id}")
                while batch_info["processing_status"] != "ended":
                    time.sleep(poll_interval)
                    batch_info = self.get_batch_status(batch_id)
                    processing_info = f"처리 완료: {batch_info['request_counts'].succeeded}, "
                    processing_info += f"오류: {batch_info['request_counts'].errored}, "
                    processing_info += f"처리 중: {batch_info['request_counts'].processing}, "
                    processing_info += f"취소됨: {batch_info['request_counts'].canceled}, "
                    processing_info += f"만료: {batch_info['request_counts'].expired}"
                    self.logger.info(f"Anthropic 배치 진행 상황: {batch_id}, {processing_info}")
            
            # 배치 처리가 완료되지 않은 경우 알림
            if batch_info["processing_status"] != "ended":
                self.logger.warning(f"Anthropic 배치가 아직 완료되지 않음: {batch_id}, 현재 상태: {batch_info['processing_status']}")
                return
            
            # 결과 URL이 없는 경우 오류 발생
            if not batch_info["results_url"]:
                raise ValueError(f"배치 {batch_id}에 대한 결과 URL이 없습니다.")
            
            # 결과 스트림 요청
            self.logger.info(f"Anthropic 배치 결과 다운로드 중: {batch_id}")
            
            # 결과 스트림 설정
            responses = self.client.messages.batches.results(batch_id)
            
            self.logger.info(f"Anthropic 배치 결과 처리 완료: {batch_id}")
            return responses
            
        except Exception as e:
            self.logger.error(f"Anthropic 배치 결과 조회 실패: {batch_id}, 오류: {str(e)}")
            raise

    def process_batch_results(self, batch_id: str, responses: Any) -> Dict[str, Any]:
        """
        배치 요청의 결과를 처리합니다.

        Args:
            responses: self.client.messages.batches.results(batch_id)의 결과 (anthropic._decoders.jsonl.JSONLDecoder)

        Returns:
            Dict[str, Any]: 배치 결과 맵(custom_id -> 결과)
        """
        try:
            self.logger.info(f"Anthropic 배치 결과 처리 중: {batch_id}")

            # 결과 맵 생성
            results = {}
            for response in responses:
                custom_id = response.custom_id
                match response.result.type:
                    case "succeeded":
                        # JSON 문자열을 파싱하여 딕셔너리로 변환
                        raw_content = response.result.message.content[0].text
                        try:
                            content = re.sub(r"^```json\s*|\s*```$", "", raw_content, flags=re.DOTALL).strip()
                            content = json.loads(content)
                        except json.JSONDecodeError as e:
                            self.logger.error(f"{custom_id} JSON 파싱 오류: {e}")
                            content = raw_content  # JSON 파싱 실패 시 원본 문자열 사용

                        results[custom_id] = {
                            "status": response.result.type,
                            "content": content,
                            "usage": response.result.message.usage,
                        }
                    case "errored":
                        results[custom_id] = {
                            "status": response.result.type,
                            "error_type": response.result.error.type,
                            "error_message": response.result.error.message,
                        }
                    case "expired" | "canceled":
                        results[custom_id] = {
                            "status": response.result.type,
                        }
                    case _:
                        results[custom_id] = {
                            "status": "Unknown"
                        }
                        
            self.logger.info(f"Anthropic 배치 결과 처리 완료: {batch_id}, 결과 수: {len(results)}")
            return results
            
        except Exception as e:
            self.logger.error(f"Anthropic 배치 결과 처리 실패: {batch_id}, 오류: {str(e)}")
            raise
            
    def batch_process(
        self, 
        prompts: List[str], 
        model: str = "claude-3-5-haiku-20241022",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        system_instruction: Optional[str] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        custom_ids: Optional[List[str]] = None,
        wait_for_completion: bool = True,
        poll_interval: int = 60
    ) -> Tuple[str, Dict[str, Any]]:
        """
        다수의 프롬프트를 일괄 처리하고 선택적으로 결과를 대기합니다.

        Args:
            prompts (List[str]): 처리할 사용자 프롬프트 목록
            model (str): 사용할 모델 이름
            temperature (float): 생성 다양성 제어
            max_tokens (int): 최대 토큰 수
            system_instruction (str, optional): 모든 요청에 공통으로 적용할 시스템 지시사항
            response_schema (dict, optional): 응답 스키마
            custom_ids (List[str], optional): 각 요청에 대한 사용자 정의 ID 목록. 제공되지 않으면 자동 생성됨.
            wait_for_completion (bool): 배치 처리 완료까지 대기할지 여부
            poll_interval (int): 대기 중 폴링 간격(초)

        Returns:
            Tuple[str, Dict[str, Any]]: 배치 ID와 결과 맵(custom_id -> 결과)
        """
        try:
            # 배치 요청 생성
            batch_response = self.create_batch_request(
                prompts=prompts,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                system_instruction=system_instruction,
                response_schema=response_schema,
                custom_ids=custom_ids
            )
            
            batch_id = batch_response.id
            
            # 완료를 기다리지 않는 경우 배치 ID만 반환
            if not wait_for_completion:
                return batch_id, {}
            
            # 결과가 준비될 때까지 대기
            self.logger.info(f"Anthropic 배치 결과 대기 중: {batch_id}")
            responses = self.get_batch_results(
                batch_id=batch_id,
                wait_for_completion=wait_for_completion,
                poll_interval=poll_interval
            )
            results = self.process_batch_results(batch_id, responses)

            return batch_id, results
            
        except Exception as e:
            self.logger.error(f"Anthropic 배치 요청 실패: {str(e)}")
            raise