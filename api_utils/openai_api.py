import logging
import os
import uuid
import time
import json
from typing import Optional, Dict, List, Any, Tuple, Iterator
from api_utils.config import init_openai_client
from api_utils.pdf_utils import validate_pdf
from api_utils.schema_manager import ResponseSchemaManager

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class OpenAIAPI:
    """OpenAI API와 상호작용하기 위한 인터페이스"""
    def __init__(self, logger=None):
        self.client = init_openai_client()
        self.logger = logger or logging.getLogger("openai_api")
        self.schema_manager = ResponseSchemaManager()

    # --------------------------------------------------------
    # PDF 파일 업로드 관련 메서드
    # --------------------------------------------------------

    def upload_pdf(self, file_path:str) -> str:
        """OpenAI의 File API에 PDF를 업로드합니다."""
        try:
            validate_pdf(file_path)
            file_name = os.path.basename(file_path)
            
            with open(file_path, "rb") as file:
                uploaded_file = self.client.files.create(
                    file=file,
                    purpose="user_data"
                )
            
            self.logger.info(f"파일 {file_name} 업로드 성공. 파일 ID: {uploaded_file.id}")
            return uploaded_file
            
        except Exception as e:
            self.logger.error(f"파일 업로드 실패 {file_name}: {str(e)}")
            raise    

    # --------------------------------------------------------
    # OpenAI API의 메시지 생성 및 응답 관련 메서드
    # --------------------------------------------------------
    
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
        
    def create_response_params(
        self,
        input: List[Dict[str, Any]], 
        model: str = "gpt-4.1-nano-2025-04-14",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        enable_thinking: bool = False,
        thinking_budget: Optional[int] = 10000,
        system_instruction: Optional[str] = None,
        response_schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        OpenAI API 요청을 위한 파라미터를 생성합니다.

        Args:
            model (str): 사용할 모델 이름
            input (list): 입력 메시지
            temperature (float): 생성 다양성 제어
            max_tokens (int): 최대 토큰 수
            enable_thinking (bool): 사고 모드 활성화 여부
            thinking_budget (int, optional): 사고 토큰 예산
            system_instruction (str, optional): 시스템 메시지
            response_schema (dict, optional): 응답 스키마

        Returns:
            Dict[str, Any]: API 요청 파라미터
        """
        # API 호출 파라미터 생성
        params = {
            "model": model,
            "input": input,
            "temperature": temperature,
            "max_output_tokens": max_tokens
        }

        # 시스템 지시사항이 제공된 경우 추가
        if system_instruction:
            params["instructions"] = system_instruction

        # 스키마가 제공된 경우 추가
        if response_schema:
            params["text"] = self.schema_manager.format_openai_schema(response_schema)

        # 사고 모드가 활성화된 경우 추가
        if enable_thinking:
            params["reasoning"] = {
                "effort": "medium",
                "summary": "auto"
            }
            params["max_output_tokens"] = max_tokens + thinking_budget
            
        return params

    def generate_response(
        self, 
        input: List[Dict[str, Any]], 
        model: str = "gpt-4.1-nano-2025-04-14",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        enable_thinking: bool = False,
        thinking_budget: Optional[int] = 10000,
        system_instruction: Optional[str] = None,
        response_schema: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        OpenAI 모델로부터 응답을 생성합니다.

        Args:
            model (str): 사용할 모델 이름
            input (list): 입력 메시지
            temperature (float): 생성 다양성 제어
            max_tokens (int): 최대 토큰 수
            system_instruction (str, optional): 시스템 메시지
            response_schema (dict, optional): 응답 스키마

        Returns:
            response: OpenAI API의 응답
        """
        try:
            self.logger.info(f"OpenAI 모델에 요청 전송 중: {model}")

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

            # OpenAI API에 요청 전송
            response = self.client.responses.create(**params)

            # 응답 시간 계산
            end_time = time.time()
            response_time = end_time - start_time

            # 응답 객체에 response_time 속성 추가
            response.response_time = response_time

            self.logger.info(f"OpenAI 모델로부터 응답 수신 성공: {model}, 응답 시간: {response_time:.2f}초")
            return response

        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            self.logger.error(f"OpenAI의 {model}에서 응답 생성 오류 (응답 시간: {response_time:.2f}초): {str(e)}")
            raise

    # --------------------------------------------------------
    # OpenAI API의 배치 요청을 생성하고 관리하는 메서드
    # --------------------------------------------------------

    def create_batch_input_file(
        self, 
        prompts: List[str], 
        model: str = "gpt-4.1-nano-2025-04-14",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        system_instruction: Optional[str] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        custom_ids: Optional[List[str]] = None
    ) -> Tuple[str, int]:
        """
        배치 요청을 위한 임시 JSONL 파일을 생성합니다. 
        각 행은 API에 대한 개별 요청의 세부 정보를 포함합니다.
        /v1/responses 엔드포인트가 사용됩니다.

        Args:
            prompts (List[str]): 처리할 사용자 프롬프트 목록
            model (str): 사용할 모델 이름
            temperature (float): 생성 다양성 제어
            max_tokens (int): 최대 토큰 수
            system_instruction (str, optional): 모든 요청에 공통으로 적용할 시스템 지시사항
            response_schema (dict, optional): 응답 스키마
            custom_ids (List[str], optional): 각 요청에 대한 사용자 정의 ID 목록. 제공되지 않으면 자동 생성됨.

        Returns:
            Tuple[str, int]: 생성된 JSONL 파일 경로와 요청 수
        """
        try:
            self.logger.info(f"OpenAI 배치 입력 파일 생성 중: {len(prompts)}개의 프롬프트, 모델: {model}")
            
            # 사용자 정의 ID가 제공되지 않은 경우 자동 생성
            if not custom_ids:
                custom_ids = [f"request_{i}" for i in range(len(prompts))]
            elif len(custom_ids) != len(prompts):
                raise ValueError("custom_ids 길이는 prompts 길이와 일치해야 합니다.")
            
            # 임시 디렉토리에 JSONL 파일 생성
            temp_dir = os.path.join(PROJECT_ROOT, 'temp')
            batch_id = str(uuid.uuid4())
            jsonl_path = os.path.join(temp_dir, f"openai_batch_{batch_id}.jsonl")
            
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for i, prompt in enumerate(prompts):

                    # 입력 메시지 생성
                    input = self.create_input_message(prompt)

                    # 요청 구성
                    body = self.create_response_params(
                        input=input,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        system_instruction=system_instruction,
                        response_schema=response_schema
                    )
                    
                    request = {"custom_id": custom_ids[i], "method": "POST", "url": "/v1/responses", "body": body}

                    # JSONL 형식으로 파일에 작성
                    f.write(json.dumps(request) + '\n')
            
            self.logger.info(f"OpenAI 배치 입력 파일 생성 성공: {jsonl_path}, 요청 수: {len(prompts)}")
            return jsonl_path, len(prompts)
            
        except Exception as e:
            self.logger.error(f"OpenAI 배치 입력 파일 생성 실패: {str(e)}")
            raise
    
    def upload_batch_input_file(self, file_path: str) -> Any:
        """
        OpenAI의 Files API를 사용해 배치 입력 JSONL 파일을 업로드합니다.

        Args:
            file_path (str): 업로드할 JSONL 파일 경로

        Returns:
            Any: 업로드된 파일의 상세 정보 객체
        """
        try:
            self.logger.info(f"OpenAI 배치 입력 파일 업로드 중: {file_path}")
            
            with open(file_path, "rb") as file:
                uploaded_file = self.client.files.create(
                    file=file,
                    purpose="batch"
                )
            
            self.logger.info(f"OpenAI 배치 입력 파일 업로드 성공. 파일 ID: {uploaded_file.id}")
            return uploaded_file
            
        except Exception as e:
            self.logger.error(f"OpenAI 배치 입력 파일 업로드 실패: {str(e)}")
            raise
    
    def create_batch_request(
        self, 
        prompts: List[str], 
        model: str = "gpt-4.1-nano-2025-04-14",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        system_instruction: Optional[str] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        custom_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = {}
    ) -> Dict[str, Any]:
        """
        여러 프롬프트를 일괄 처리하기 위한 배치 요청을 생성합니다.
        create_batch_input_file과 upload_batch_input_file 함수를 호출한 다음
        client.batches.create() 함수를 호출하여 배치 객체를 반환합니다.

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
            self.logger.info(f"OpenAI 배치 요청 생성 중: {len(prompts)}개의 프롬프트, 모델: {model}")
            
            # 1. 배치 입력 파일 생성
            jsonl_path, request_count = self.create_batch_input_file(
                prompts=prompts,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                system_instruction=system_instruction,
                response_schema=response_schema,
                custom_ids=custom_ids
            )
            
            # 2. 배치 입력 파일 업로드
            file_id = self.upload_batch_input_file(jsonl_path).id
            
            # 3. 배치 요청 생성
            batch_response = self.client.batches.create(
                input_file_id=file_id,
                endpoint="/v1/responses",
                completion_window="24h",
                metadata=metadata
            )

            # 4. 로컬 임시 파일 삭제
            try:
                os.remove(jsonl_path)
                self.logger.info(f"임시 JSONL 파일 삭제 완료: {jsonl_path}")
            except Exception as e:
                self.logger.warning(f"임시 JSONL 파일 삭제 실패: {str(e)}")
            
            self.logger.info(f"OpenAI 배치 요청 생성 성공: 배치 ID: {batch_response.id}, 요청 수: {request_count}")
            return batch_response
            
        except Exception as e:
            self.logger.error(f"OpenAI 배치 요청 생성 실패: {str(e)}")
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
            self.logger.info(f"OpenAI 배치 상태 확인 중: {batch_id}")
            
            batch_info = self.client.batches.retrieve(
                batch_id=batch_id
            )
            
            status = {
                "id": batch_info.id,
                "status": batch_info.status,
                "errors": batch_info.errors,
                "created_at": batch_info.created_at,
                "completed_at": batch_info.completed_at,
                "total_requests": batch_info.request_counts.total,
                "completed_count": batch_info.request_counts.completed,
                "failed_count": batch_info.request_counts.failed,
                "input_file_id": batch_info.input_file_id,
                "output_file_id": batch_info.output_file_id,
                "error_file_id": batch_info.error_file_id,
            }
            
            self.logger.info(f"OpenAI 배치 상태: {batch_id}, 처리 상태: {batch_info.status}")
            return status
            
        except Exception as e:
            self.logger.error(f"OpenAI 배치 상태 조회 실패: {batch_id}, 오류: {str(e)}")
            raise
    
    def get_batch_results(self, batch_id: str, wait_for_completion: bool = False, poll_interval: int = 60) -> Iterator[Dict[str, Any]]:
        """
        배치 요청의 결과를 조회합니다. 필요시 완료까지 대기합니다.

        Args:
            batch_id (str): 조회할 배치 ID
            wait_for_completion (bool): 배치 처리가 완료될 때까지 대기할지 여부
            poll_interval (int): 대기 중 폴링 간격(초)

        Returns:
            Str[List[Dict[str, Any]]]: 배치 결과 데이터
        """
        try:
            # 배치 상태 초기 확인
            batch_info = self.get_batch_status(batch_id)
            
            # 완료될 때까지 대기
            if wait_for_completion and batch_info["status"] != "completed" and batch_info["status"] != "failed":
                self.logger.info(f"OpenAI 배치 완료 대기 중: {batch_id}")
                while batch_info["status"] not in ["completed", "failed", "cancelled", "expired"]:
                    time.sleep(poll_interval)
                    batch_info = self.get_batch_status(batch_id)
                    processing_info = f"완료: {batch_info['completed_count']}, "
                    processing_info += f"실패: {batch_info['failed_count']}"
                    self.logger.info(f"OpenAI 배치 진행 상황: {batch_id}, {processing_info}")
            
            # 배치 처리가 완료되지 않은 경우 알림
            if batch_info["status"] == "in_progress":
                self.logger.warning(f"OpenAI 배치가 아직 완료되지 않음: {batch_id}, 현재 상태: {batch_info['status']}")
                return []
                
            # 실패한 경우 오류 메시지 로깅
            if batch_info["status"] == "failed":
                self.logger.error(f"OpenAI 배치 처리 실패: {batch_id}, 오류: {batch_info['error']}")
                return []
                
            # 취소되거나 만료된 경우 알림
            if batch_info["status"] in ["cancelled", "expired"]:
                self.logger.warning(f"OpenAI 배치 처리 {batch_info['status']}: {batch_id}")
                return []
                
            # 결과 데이터 가져오기
            self.logger.info(f"OpenAI 배치 결과 다운로드 중: {batch_id}")

            # 배치 결과는 jsonl 형식 파일로 저장되며, FILES API를 통해 접근함
            output_file_id = batch_info['output_file_id']
            if not output_file_id:
                self.logger.error(f"OpenAI 배치 결과 파일이 없습니다: {batch_id}")
                return []
            
            # 결과 파일 다운로드
            output_file = self.client.files.content(output_file_id)
            self.logger.info(f"OpenAI 배치 결과 파일 다운로드 완료: {output_file_id}")

            # 임시저장 디렉토리에 결과 파일 저장
            temp_dir = os.path.join(PROJECT_ROOT, 'temp')
            output_file_path = os.path.join(temp_dir, f"openai_batch_output_{batch_id}.jsonl")
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(output_file.text)
                self.logger.info(f"OpenAI 배치 결과 임시파일 저장 완료: {output_file_path}")

            return output_file.text
            
        except Exception as e:
            self.logger.error(f"OpenAI 배치 결과 조회 실패: {batch_id}, 오류: {str(e)}")
            raise

    def process_batch_results(self, batch_id: str, output_file_text: str) -> Dict[str, Any]:
        """
        배치 요청의 결과를 처리합니다.

        Args:
            batch_id (str): 배치 ID
            responses (List[Dict[str, Any]]): 배치 응답 결과 목록

        Returns:
            Dict[str, Any]: 배치 결과 맵(custom_id -> 결과)
        """
        try:
            self.logger.info(f"OpenAI 배치 결과 처리 중: {batch_id}")

            results = {}
            for entry in output_file_text.splitlines():
                json_entry = json.loads(entry)
                custom_id = json_entry['custom_id']
                response = json_entry['response']['body']

                if response['status'] == "completed":
                    # 성공적으로 완료된 응답 처리
                    results[custom_id] = {
                        "status": "completed",
                        "content": response['output'][0]['content'][0]['text'],
                        "usage": response['usage'],
                    }
                            
                elif response['status'] == "failed":
                    # 실패한 응답 처리
                    results[custom_id] = {
                        "status": "failed",
                        "error": response.error,
                    }
                else:
                    # 기타 상태 처리
                    results[custom_id] = {
                        "status": response['status'],
                    }
                        
            self.logger.info(f"OpenAI 배치 결과 처리 완료: {batch_id}, 결과 수: {len(results)}")
            return results
            
        except Exception as e:
            self.logger.error(f"OpenAI 배치 결과 처리 실패: {batch_id}, 오류: {str(e)}")
            raise
            
    def batch_process(
        self, 
        prompts: List[str], 
        model: str = "gpt-4.1-nano-2025-04-14",
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
            self.logger.info(f"OpenAI 배치 결과 대기 중: {batch_id}")
            responses = self.get_batch_results(
                batch_id=batch_id,
                wait_for_completion=wait_for_completion,
                poll_interval=poll_interval
            )
            
            # 결과 처리
            results = self.process_batch_results(batch_id, responses)

            # 임시 파일 삭제
            temp_dir = os.path.join(PROJECT_ROOT, 'temp')
            output_file_path = os.path.join(temp_dir, f"openai_batch_output_{batch_id}.jsonl")
            if os.path.exists(output_file_path):
                os.remove(output_file_path)
                self.logger.info(f"OpenAI 배치 결과 임시파일 삭제 완료: {output_file_path}")
            else:
                self.logger.warning(f"OpenAI 배치 결과 임시파일이 존재하지 않음: {output_file_path}")
            
            return batch_id, results
            
        except Exception as e:
            self.logger.error(f"OpenAI 배치 요청 실패: {str(e)}")
            raise