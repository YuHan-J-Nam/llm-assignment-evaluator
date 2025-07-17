import logging
import json
import asyncio
import pandas as pd
import concurrent.futures
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple, Union
from api_utils.logging_utils import setup_logging
from api_utils.schema_manager import ResponseSchemaManager
from api_utils.gemini_api import GeminiAPI
from api_utils.anthropic_api import AnthropicAPI
from api_utils.openai_api import OpenAIAPI
from api_utils.token_counter import TokenCounter
import copy

MODEL_DICT = {
    "Gemini": {
        "default_model": "gemini-2.0-flash-lite",
        "supported_models": [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite-preview-06-17",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
        ],
        "thinking_models": [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite-preview-06-17"
        ]
    },
    "Anthropic": {
        "default_model": "claude-3-5-haiku-20241022",
        "supported_models": [
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-3-7-sonnet-20250219",
            "claude-3-5-haiku-20241022",
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
            "claude-3-haiku-20240307"
        ],
        "thinking_models": [
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-3-7-sonnet-20250219"
        ]
    },
    "OpenAI": {
        "default_model": "gpt-4.1-nano",
        "supported_models": [
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "gpt-4o",
            "gpt-4o-mini",
            # "o3",  # organization must be verified to use the model `o3`
            "o4-mini",
            "o3-mini"
        ],
        "thinking_models": [
            # "o3",
            "o4-mini",
            "o3-mini"
        ]
    }
}

class LLMAPIClient:
    """Unified interface for interacting with multiple LLM APIs with async support"""
    
    def __init__(self, log_level=logging.INFO):
        self.logger = setup_logging(log_level)
        self.schema_manager = ResponseSchemaManager()

        # Initialize API clients
        self.gemini = GeminiAPI(self.logger)
        self.anthropic = AnthropicAPI(self.logger)
        self.openai = OpenAIAPI(self.logger)
        
        # Initialize token counter
        self.token_counter = TokenCounter(self.logger)
        
        # Initialize thread pool executor for async operations
        self.executor = concurrent.futures.ThreadPoolExecutor()
        
        self.logger.info("LLM API Client initialized")

    def _get_provider_for_model(self, model: str) -> Optional[str]:
        """Determine the provider for a given model name"""
        for provider, config in MODEL_DICT.items():
            if model in config["supported_models"]:
                return provider
        return None

    def _is_thinking_supported(self, model: str) -> bool:
        """Check if the given model supports thinking/reasoning"""
        for _, config in MODEL_DICT.items():
            if model in config["thinking_models"]:
                return True
        return False
    
    async def generate_responses(
        self, 
        prompt: str, 
        models: List[str], 
        file_path: Optional[str] = None,
        temperature: float = 0.2, 
        max_tokens: int = 4096,
        system_instruction: Optional[str] = None, 
        response_schema: Optional[Dict[str, Any]] = None,
        enable_thinking: bool = False,
        thinking_budget: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Asynchronously generate responses from multiple models with the same prompt.
        
        Args:
            prompt: The prompt to send to the models
            models: List of model names to query
            file_path: Optional path to a PDF file to process
            temperature: Temperature setting for generation
            max_tokens: Maximum tokens in response
            system_instruction: Optional system instruction
            response_schema: Optional response schema
            enable_thinking: Whether to enable thinking for supported models
            thinking_budget: Optional token budget for thinking
            
        Returns:
            List of dictionaries containing model responses and metadata
        """
        tasks = []
        loop = asyncio.get_event_loop()
        
        for model in models:
            provider = self._get_provider_for_model(model)
            if not provider:
                self.logger.error(f"Unknown model: {model}")
                continue
                
            # Check if thinking is supported when requested
            if enable_thinking and not self._is_thinking_supported(model):
                self.logger.error(f"Model {model} does not support thinking/reasoning, but enable_thinking=True")
                # Continue processing without thinking for this model
            
            if provider == "Gemini":
                task = loop.run_in_executor(
                    self.executor,
                    lambda m=model: self._process_with_gemini(
                        file_path=file_path,
                        prompt=prompt,
                        model=m,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        system_instruction=system_instruction,
                        response_schema=response_schema,
                        enable_thinking=enable_thinking and self._is_thinking_supported(m),
                        thinking_budget=thinking_budget
                    )
                )
            elif provider == "Anthropic":
                task = loop.run_in_executor(
                    self.executor,
                    lambda m=model: self._process_with_anthropic(
                        file_path=file_path,
                        prompt=prompt,
                        model=m,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        system_instruction=system_instruction,
                        response_schema=response_schema,
                        enable_thinking=enable_thinking and self._is_thinking_supported(m),
                        thinking_budget=thinking_budget
                    )
                )
            elif provider == "OpenAI":
                task = loop.run_in_executor(
                    self.executor,
                    lambda m=model: self._process_with_openai(
                        file_path=file_path,
                        prompt=prompt,
                        model=m,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        system_instruction=system_instruction,
                        response_schema=response_schema,
                        enable_thinking=enable_thinking and self._is_thinking_supported(m),
                        thinking_budget=thinking_budget
                    )
                )
            
            tasks.append((model, task))
        
        # Gather results
        responses = []
        for model, task in tasks:
            try:
                response_dict = await task
                response_dict["metadata"]["timestamp"] = datetime.now().isoformat()
                responses.append(response_dict)
            except Exception as e:
                self.logger.error(f"Error processing with model {model}: {str(e)}")
                responses.append({
                    "metadata": {
                        "model": model,
                        "timestamp": datetime.now().isoformat()
                    },
                    "request_error": str(e)
                })
        
        return responses

    async def generate_batch_batch_dict(
        self,
        prompts: List[str],
        models: List[str],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        system_instruction: Optional[str] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        custom_ids: Optional[List[str]] = None,
        wait_for_completion: bool = True,
        poll_interval: int = 60
    ) -> Dict[str, Any]:
        """
        Process multiple prompts with multiple models using batch processing.
        The same prompts will be sent to each model in parallel.
        
        Args:
            prompts: List of prompts to process
            models: List of models to use
            temperature: Temperature setting for generation
            max_tokens: Maximum tokens in response
            system_instruction: Optional system instruction
            response_schema: Optional response schema
            custom_ids: Optional custom IDs for batch batch_dict
            wait_for_completion: Whether to wait for batch completion
            poll_interval: Polling interval for batch status checks
            
        Returns:
            Dictionary mapping model names to batch results
        """
        tasks = []
        loop = asyncio.get_event_loop()
        
        for model in models:
            provider = self._get_provider_for_model(model)
            if not provider:
                self.logger.error(f"Unknown model: {model}")
                continue
                
            if provider == "Anthropic":
                task = loop.run_in_executor(
                    self.executor,
                    lambda m=model: self.anthropic.create_batch_request(
                        prompts=prompts,
                        model=m,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        system_instruction=system_instruction,
                        response_schema=response_schema,
                        custom_ids=custom_ids
                    )
                )
                tasks.append((model, task))
                
            elif provider == "OpenAI":
                task = loop.run_in_executor(
                    self.executor,
                    lambda m=model: self.openai.create_batch_request(
                        prompts=prompts,
                        model=m,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        system_instruction=system_instruction,
                        response_schema=response_schema,
                        custom_ids=custom_ids
                    )
                )
                tasks.append((model, task))
                
            else:
                # Gemini doesn't support batch processing in the same way
                continue
        
        # Gather results
        batch_dict = {}
        for model, task in tasks:
            try:
                batch_request = await task
                batch_id = batch_request.id
                batch_dict[batch_id] = {
                    "model": model,
                    "timestamp": datetime.now().isoformat(),
                    "custom_ids": custom_ids
                }
            except Exception as e:
                self.logger.error(f"Error batch processing with model {model}: {str(e)}")
                batch_dict[batch_id] = {
                    "model": model,
                    "timestamp": datetime.now().isoformat(),
                    "custom_ids": custom_ids,
                    "error": str(e)
                }
        
        return batch_dict

    async def check_batch_status(
        self,
        batch_dict: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Check the status of multiple batch batch_dict.
        
        Args:
            batch_dict: Dictionary mapping batch id to batch information
                       Format: {'batch_id': {'model': 'model_name', 'timestamp': 'timestamp', etc.}}
            
        Returns:
            Dictionary mapping batch id to their status information
            Format: {'batch_id': {'status': 'status_info', 'model': 'model', etc.}}
        """
        tasks = []
        loop = asyncio.get_event_loop()
        
        for batch_id, batch_info in batch_dict.items():
            if 'batch_id' not in batch_info:
                self.logger.error(f"No batch_id found for model {model}")
                continue
                
            model = batch_info.get('model', None)
            provider = self._get_provider_for_model(model)
            
            if not provider:
                self.logger.error(f"Unknown model: {model}")
                continue
                
            if provider == "Anthropic":
                task = loop.run_in_executor(
                    self.executor,
                    lambda bid=batch_id: self.anthropic.get_batch_status(bid)
                )
                tasks.append(batch_id, task)
                
            elif provider == "OpenAI":
                task = loop.run_in_executor(
                    self.executor,
                    lambda bid=batch_id: self.openai.get_batch_status(bid)
                )
                tasks.append(batch_id, task)
        

        for batch_id, task in tasks:
            try:
                status = await task
                batch_dict[batch_id]['status'] = status
                
                # # Log status information
                # if model in MODEL_DICT['Anthropic']['supported_models']:
                #     processing_status = status.get('processing_status', 'unknown')
                #     request_counts = status.get('request_counts', {})
                #     self.logger.info(
                #         f"Anthropic batch {status['id']} (model: {model}) - "
                #         f"Status: {processing_status}, "
                #         f"Succeeded: {getattr(request_counts, 'succeeded', 0)}, "
                #         f"Errored: {getattr(request_counts, 'errored', 0)}, "
                #         f"Processing: {getattr(request_counts, 'processing', 0)}"
                #     )
                # elif model in MODEL_DICT['OpenAI']['supported_models']:
                #     batch_status = status.get('status', 'unknown')
                #     completed_count = status.get('completed_count', 0)
                #     failed_count = status.get('failed_count', 0)
                #     total_batch_dict = status.get('total_batch_dict', 0)
                #     self.logger.info(
                #         f"OpenAI batch {status['id']} (model: {model}) - "
                #         f"Status: {batch_status}, "
                #         f"Completed: {completed_count}/{total_batch_dict}, "
                #         f"Failed: {failed_count}"
                #     )
                    
            except Exception as e:
                self.logger.error(f"Error checking batch status for model {model}: {str(e)}")
                batch_dict[batch_id]['status'] = {
                    'error': str(e)
                }
        
        return batch_dict

    async def retrieve_batch_results(
        self,
        batch_dict: Dict[str, Dict[str, Any]],
        wait_for_completion: bool = False,
        poll_interval: int = 60
    ) -> Dict[str, Any]:
        """
        Retrieve results for completed batch batch_dict.
        
        Args:
            batch_dict: Dictionary mapping batch id to batch information
                          Format: {'batch_id': {'model': 'model_name', 'status': 'status', etc.}}
            wait_for_completion: Whether to wait for batch completion
            poll_interval: Polling interval for batch status checks
            
        Returns:
            Dictionary mapping batch id to their results
        """
        tasks = []
        loop = asyncio.get_event_loop()
        
        for batch_id, batch_info in batch_dict.items():
            model = batch_info.get('model', None)
            provider = self._get_provider_for_model(model)
            
            if not provider:
                self.logger.error(f"Unknown model: {model}")
                continue
                
            if provider == "Anthropic":
                task = loop.run_in_executor(
                    self.executor,
                    lambda bid=batch_id: self.anthropic.get_batch_results(
                        batch_id=bid,
                        wait_for_completion=wait_for_completion,
                        poll_interval=poll_interval
                    )
                )

                tasks.append(task)
                
            elif provider == "OpenAI":
                task = loop.run_in_executor(
                    self.executor,
                    lambda bid=batch_id: self.openai.get_batch_results(
                        batch_id=bid,
                        wait_for_completion=wait_for_completion,
                        poll_interval=poll_interval
                    )
                )
                tasks.append(task)
        
        # Gather results
        retrieved_results = {}
        for task in tasks:
            try:
                result = await task
                model = result['model']
                batch_id = result['batch_id']
                results = result['results']
                
                retrieved_results[model] = {
                    'batch_id': batch_id,
                    'raw_results': results
                }
                
                self.logger.info(f"Successfully retrieved batch results for model {model}, batch_id: {batch_id}")
                
            except Exception as e:
                self.logger.error(f"Error retrieving batch results for model {model}: {str(e)}")
                retrieved_results[model] = {"error": str(e)}
        
        return retrieved_results

    async def process_batch_results(
        self,
        batch_results_dict: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process the raw batch results into a standardized format.
        
        Args:
            batch_results_dict: Dictionary containing raw batch results from retrieve_batch_results
                              Format: {'model_name': {'batch_id': 'batch_id', 'raw_results': results}}
            
        Returns:
            Dictionary mapping model names to their processed batch results
        """
        tasks = []
        loop = asyncio.get_event_loop()
        
        for model, batch_data in batch_results_dict.items():
            if 'error' in batch_data:
                self.logger.error(f"Skipping processing for model {model} due to error: {batch_data['error']}")
                continue
                
            if 'batch_id' not in batch_data or 'raw_results' not in batch_data:
                self.logger.error(f"Missing batch_id or raw_results for model {model}")
                continue
                
            batch_id = batch_data['batch_id']
            raw_results = batch_data['raw_results']
            provider = self._get_provider_for_model(model)
            
            if not provider:
                self.logger.error(f"Unknown model: {model}")
                continue
                
            if provider == "Anthropic":
                task = loop.run_in_executor(
                    self.executor,
                    lambda m=model, bid=batch_id, results=raw_results: {
                        'model': m,
                        'processed_results': self.anthropic.process_batch_results(bid, m, results)
                    }
                )
                tasks.append(task)
                
            elif provider == "OpenAI":
                task = loop.run_in_executor(
                    self.executor,
                    lambda m=model, bid=batch_id, results=raw_results: {
                        'model': m,
                        'processed_results': self.openai.process_batch_results(bid, m, results)
                    }
                )
                tasks.append(task)
        
        # Gather results
        processed_results = {}
        for task in tasks:
            try:
                result = await task
                model = result['model']
                processed_data = result['processed_results']
                
                processed_results[model] = processed_data
                
                # Log processing summary
                if isinstance(processed_data, list):
                    self.logger.info(f"Successfully processed {len(processed_data)} batch results for model {model}")
                else:
                    self.logger.info(f"Successfully processed batch results for model {model}")
                    
            except Exception as e:
                self.logger.error(f"Error processing batch results for model {model}: {str(e)}")
                processed_results[model] = {"error": str(e)}
        
        return processed_results

    def save_results(self, results: List[Dict[str, Any]], file_path: str) -> None:
        """
        Save the results of API calls to a JSON file.

        Args:
            results: List of dictionaries containing model responses and metadata
            file_path: Path to save the results JSON file
        """
        # Create a deep copy to prevent modifying original results
        results_copy = copy.deepcopy(results)
        
        # Convert results to JSON serializable format
        for result in results_copy:
            if 'response' in result:
                # Convert any non-serializable objects to string
                result['response'] = result['response'].model_dump(mode='json')
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(results_copy, f, ensure_ascii=False, indent=2)
                self.logger.info(f"Results saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving results to file {file_path}: {str(e)}")
            raise

    def parse_response_by_provider(self, response, provider: str = None, model: str = None) -> Dict[str, Any]:
        """
        제공업체에 따라 적절한 파싱 메서드를 선택하여 응답 파싱
        
        Args:
            response: API 응답 객체 또는 딕셔너리
            provider: API 제공업체 ('gemini', 'anthropic', 'openai') (선택사항)
            model: 모델명 (선택사항)
            
        Returns:
            파싱된 텍스트와 토큰 사용량이 담긴 딕셔너리
        """
        provider = provider.lower()
        
        if provider == 'gemini' or model.startswith('gemini'):
            return self.gemini.parse_response(response)
        elif provider == 'anthropic' or model.startswith('claude'):
            return self.anthropic.parse_response(response)
        elif provider == 'openai' or model.startswith('gpt') or model.startswith('o'):
            return self.openai.parse_response(response, self._is_thinking_supported(model))
        else:
            self.logger.warning(f"알 수 없는 제공업체: {provider}")
            return {"text": "", "token_usage": {}}

    def parse_responses(self, responses: List[Dict[str, Any]], schema: Dict = None, custom_id_func: Optional[callable] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        여러 API 응답을 파싱하여 관련 정보 추출
        
        Args:
            responses: 모델 응답과 메타데이터가 포함된 딕셔너리 목록
            schema: 응답 텍스트를 검증하고 파싱하는 데 사용할 JSON 스키마 (선택사항)
            custom_id_func: custom_id에서 정보를 추출하는 선택적 함수
            **kwargs: 처리된 결과에 포함할 추가 메타데이터
            
        Returns:
            메타데이터(모델명), 응답 텍스트, 토큰 사용량이 포함된 처리된 결과 딕셔너리 목록
        """
        processed_results = []
        for response_dict in responses:
            if 'error' in response_dict:
                continue
            elif 'response' not in response_dict:
                continue
            
            # 모델명과 응답 추출
            model = response_dict['metadata']['model']
            response = response_dict['response']
            
            try:
                # 제공 
                provider = self._get_provider_for_model(model)
                if not provider:
                    self.logger.warning(f"모델 {model}의 제공업체를 확인할 수 없습니다")
                    continue
                
                # 응답 파싱
                parsed_response = self.parse_response_by_provider(response, provider, model)
                text = parsed_response["text"]
                token_usage = parsed_response["token_usage"]
                
                # 토큰 사용량 로깅
                if token_usage:
                    self.token_counter.log_token_usage(
                        model_name=model,
                        input_tokens=token_usage.get("input_tokens", 0),
                        output_tokens=token_usage.get("output_tokens", 0),
                        cached_input_tokens=token_usage.get("cached_input_tokens", 0),
                        total_tokens=token_usage.get("total_tokens", 0)
                    )
                
            except Exception as e:
                self.logger.error(f"모델 {model} 처리 중 오류로 인해 건너뜀: {str(e)}")
                text = ""
                token_usage = {}

            # custom_id_func가 제공된 경우 추가 메타데이터 추출
            if custom_id_func:
                try:
                    custom_id = response_dict['metadata']['custom_id']
                    additional_info = custom_id_func(custom_id)
                    kwargs.update(additional_info)
                except Exception as e:
                    pass

            # 추가 정보가 포함된 메타데이터 딕셔너리 생성
            metadata = response_dict['metadata'].copy()
            metadata.update(kwargs)

            # 스키마가 제공된 경우 스키마 매니저를 사용하여 응답 텍스트 파싱
            if schema and text:
                parsed_text = self.schema_manager.parse_content_to_json(response_text=text, schema=schema, model=model)
            else:
                parsed_text = text

            processed_results.append({
                "metadata": metadata,
                "response": {
                    "text": parsed_text,
                    "token_usage": token_usage
                }
            })
            
        return processed_results
    
    def extract_evaluation_scores(self, processed_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Extract scores from results of API calls.
        
        Args:
            processed_results: List of dictionaries containing processed results with metadata and response text
            
        Returns:
            pd.DataFrame: DataFrame containing extracted scores and metadata
        """
        all_results = []

        # 결과 처리
        try:
            for result_dict in processed_results:
                response = result_dict.get('response', {})
                if 'raw_text' in response['text']:
                    continue  # JSON 파싱 실패로 인해 raw_text가 있는 경우 건너뜀

                data_dict = result_dict['metadata'].copy()
                model = data_dict.get('model', 'unknown_model')

                # Extract scores from the evaluation
                def _extract_scores_recursive(item, data_dict):
                    """Recursively extract name and score pairs from nested structures"""
                    if isinstance(item, dict):
                        # Check if this dict has both 'name' and 'score' keys
                        if 'name' in item and 'score' in item:
                            data_dict[item['name']] = item['score']
                        else:
                            # Recursively check all values in the dict
                            for value in item.values():
                                _extract_scores_recursive(value, data_dict)
                    elif isinstance(item, list):
                        # Recursively check all items in the list
                        for sub_item in item:
                            _extract_scores_recursive(sub_item, data_dict)

                if 'text' not in response or 'evaluation' not in response['text']:
                    self.logger.warning(f"Skipping model {model} due to missing evaluation result in response")
                    continue
                else:
                    _extract_scores_recursive(response['text']['evaluation'], data_dict)

                all_results.append(data_dict)

            # 결과를 DataFrame으로 반환
            return pd.DataFrame(all_results)
        
        except Exception as e:
            self.logger.error(f"Error extracting scores: {str(e)}")
            return pd.DataFrame()
        
    def _process_with_gemini(
        self, 
        file_path: Optional[str], 
        prompt: str, 
        model: str = "gemini-2.0-flash-lite",
        temperature: float = 0.1,
        top_p: float = 0.95,
        max_tokens: int = 4096,
        system_instruction: Optional[str] = None, 
        response_schema: Optional[Dict[str, Any]] = None,
        enable_thinking: bool = False,
        thinking_budget: Optional[int] = None
    ) -> Any:
        """Process a request with Gemini API"""
        try:
            file_obj = None
            if file_path is not None:
                self.logger.info(f"Processing with Gemini {model} (with PDF): {file_path}")
                file_obj = self.gemini.upload_pdf(file_path)
            else:
                self.logger.info(f"Processing with Gemini model: {model}")
            
            # Create input content
            contents = self.gemini.create_input_message(prompt, file_obj)
            
            # Generate response
            response_dict = self.gemini.generate_response(
                input=contents,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
                thinking_budget=thinking_budget,
                system_instruction=system_instruction,
                response_schema=response_schema
            )
            
            return response_dict
            
        except Exception as e:
            raise
    
    def _process_with_anthropic(
        self, 
        file_path: Optional[str], 
        prompt: str, 
        model: str = "claude-3-5-haiku-20241022", 
        temperature: float = 0.2, 
        max_tokens: int = 4096, 
        system_instruction: Optional[str] = None, 
        response_schema: Optional[Dict[str, Any]] = None,
        enable_thinking: bool = False,
        thinking_budget: Optional[int] = 5000
    ) -> Any:
        """Process a request with Anthropic API"""
        try:
            file_id, file_data = None, None

            if file_path is not None:
                self.logger.info(f"Processing with Anthropic {model} (with PDF): {file_path}")
                try:
                    uploaded_file = self.anthropic.upload_pdf(file_path)
                    file_id = uploaded_file.id
                    self.logger.info(f"PDF 업로드 성공, 파일 ID 사용: {file_id}")
                except Exception as upload_error:
                    self.logger.warning(f"PDF 업로드 실패, base64 인코딩으로 대체: {str(upload_error)}")
                    try:
                        file_data = self.anthropic.prepare_pdf(file_path)
                        self.logger.info("PDF base64 인코딩 성공")
                    except Exception as prepare_error:
                        self.logger.error(f"PDF 처리 완전 실패: {str(prepare_error)}")
                        raise prepare_error
            else:
                self.logger.info(f"Processing with Anthropic model: {model}")
            
            # Create message with file
            messages = self.anthropic.create_input_message(prompt, file_data=file_data)
            
            # Generate response
            response_dict = self.anthropic.generate_response(
                input=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
                thinking_budget=thinking_budget if enable_thinking else None,
                system_instruction=system_instruction,
                response_schema=response_schema
            )
            
            return response_dict
            
        except Exception as e:
            raise
    
    def _process_with_openai(
        self, 
        file_path: Optional[str], 
        prompt: str, 
        model: str = "gpt-4.1-nano", 
        temperature: float = 0.2, 
        max_tokens: int = 4096, 
        system_instruction: Optional[str] = None, 
        response_schema: Optional[Dict[str, Any]] = None,
        enable_thinking: bool = False,
        thinking_budget: Optional[int] = 10000
    ) -> Any:
        """Process a request with OpenAI API"""
        try:
            file_id, file_data = None, None

            if file_path is not None:
                self.logger.info(f"Processing with OpenAI {model} (with PDF): {file_path}")
                try:
                    uploaded_file = self.openai.upload_pdf(file_path)
                    file_id = uploaded_file.id
                    self.logger.info(f"PDF 업로드 성공, 파일 ID 사용: {file_id}")
                except Exception as upload_error:
                    self.logger.warning(f"PDF 업로드 실패, base64 인코딩으로 대체: {str(upload_error)}")
                    try:
                        file_data = self.openai.prepare_pdf(file_path)
                        self.logger.info("PDF base64 인코딩 성공")
                    except Exception as prepare_error:
                        self.logger.error(f"PDF 처리 완전 실패: {str(prepare_error)}")
                        raise prepare_error
            else:
                self.logger.info(f"Processing with OpenAI model: {model}")
            
            # Create input message with file
            input = self.openai.create_input_message(prompt, file_id, file_data)
            
            # Generate response
            response_dict = self.openai.generate_response(
                input=input,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
                thinking_budget=thinking_budget if enable_thinking else None,
                system_instruction=system_instruction,
                response_schema=response_schema
            )
            
            return response_dict
            
        except Exception as e:
            raise
    
    def get_token_usage_summary(self):
        """모든 API 호출에 대한 토큰 사용량 요약 가져오기"""
        return self.token_counter.create_total_usage_summary()
    
