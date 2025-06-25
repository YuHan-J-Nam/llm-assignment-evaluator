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
        for provider, config in MODEL_DICT.items():
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
        results = []
        for model, task in tasks:
            try:
                response_dict = await task
                response_dict["metadata"]["timestamp"] = datetime.now().isoformat()
                results.append(response_dict)
            except Exception as e:
                self.logger.error(f"Error processing with model {model}: {str(e)}")
                results.append({
                    "metadata": {
                        "model": model,
                        "timestamp": datetime.now().isoformat()
                    },
                    "request_error": str(e)
                })
        
        return results
    
    async def batch_process_prompts(
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
        
        Args:
            prompts: List of prompts to process
            models: List of models to use
            temperature: Temperature setting for generation
            max_tokens: Maximum tokens in response
            system_instruction: Optional system instruction
            response_schema: Optional response schema
            custom_ids: Optional custom IDs for batch requests
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
                    lambda m=model: self.anthropic.batch_process(
                        prompts=prompts,
                        model=m,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        system_instruction=system_instruction,
                        response_schema=response_schema,
                        custom_ids=custom_ids,
                        wait_for_completion=wait_for_completion,
                        poll_interval=poll_interval
                    )
                )
                tasks.append((model, task))
                
            elif provider == "OpenAI":
                task = loop.run_in_executor(
                    self.executor,
                    lambda m=model: self.openai.batch_process(
                        prompts=prompts,
                        model=m,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        system_instruction=system_instruction,
                        response_schema=response_schema,
                        custom_ids=custom_ids,
                        wait_for_completion=wait_for_completion,
                        poll_interval=poll_interval
                    )
                )
                tasks.append((model, task))
                
            else:
                # Gemini doesn't support batch processing in the same way
                continue
        
        # Gather results
        results = {}
        for model, task in tasks:
            try:
                batch_id, batch_results = await task
                results[model] = {
                    "batch_id": batch_id,
                    "results": batch_results
                }
            except Exception as e:
                self.logger.error(f"Error batch processing with model {model}: {str(e)}")
                results[model] = {"error": str(e)}
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], file_path: str) -> None:
        """
        Save the results of API calls to a JSON file.

        Args:
            results: List of dictionaries containing model responses and metadata
            file_path: Path to save the results JSON file
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Results saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving results to file {file_path}: {str(e)}")
            raise

    def process_results(self, results: List[Dict[str, Any]], custom_id_func: Optional[callable] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Process the results of API calls to extract relevant information.
        
        Args:
            results: List of dictionaries containing model responses and metadata
            custom_id_func: Optional function to extract information from custom_id
            **kwargs: Additional metadata to include in the processed results
            
        Returns:
            List of dictionaries with processed results containing metadata(model name), response text, and token usage
        """
        processed_results = []
        for result_dict in results:
            if 'error' in result_dict:
                continue
            elif 'response' not in result_dict:
                continue
            
            # Extract model name and response
            model = result_dict['metadata']['model']
            try:
                if model in MODEL_DICT['Gemini']['supported_models']:
                    text = result_dict['response'].candidates[0].content.parts[0].text
                    token_usage = json.loads(result_dict['response'].usage_metadata.model_dump_json())
                elif model in MODEL_DICT['Anthropic']['supported_models']:
                    text = result_dict['response'].content[0].text
                    token_usage = json.loads(result_dict['response'].usage.model_dump_json())
                elif model in MODEL_DICT['OpenAI']['supported_models']:
                    # OpenAI responses with reasoning can have multiple outputs, handle accordingly
                    output_idx = 0 if model not in MODEL_DICT['OpenAI']['thinking_models'] else 1
                    text = result_dict['response'].output[output_idx].content[0].text
                    token_usage = json.loads(result_dict['response'].usage.model_dump_json())
            except Exception as e:
                self.logger.error(f"Skipping model {model} due to processing error: {str(e)}")
                text = ""
                token_usage = {}

            # If custom_id_func is provided, extract additional metadata
            if custom_id_func:
                try:
                    custom_id = result_dict['metadata']['custom_id']
                    additional_info = custom_id_func(custom_id)
                    kwargs.update(additional_info)
                except Exception as e:
                    pass

            # Create metadata dictionary with additional information
            metadata = result_dict['metadata'].copy()
            metadata.update(kwargs)

            # Parse the response text using the schema manager
            parsed_text = self.schema_manager.parse_response(response_text=text, model=model)

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
                if 'raw_text' in result_dict['response']['text']:
                    continue  # JSON 파싱 실패로 인해 raw_text가 있는 경우 건너뜀

                data_dict = result_dict['metadata'].copy()
                model = data_dict.get('model', 'unknown_model')
                response = result_dict['response']['text']

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

                _extract_scores_recursive(response['evaluation'], data_dict)
                
                if 'text' not in response or 'evaluation' not in response['text']:
                    self.logger.warning(f"Skipping model {model} due to missing checklist in response")
                    continue

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
                self.logger.info(f"Processing with Gemini (with PDF): {file_path}")
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
                self.logger.info(f"Processing with Anthropic (with PDF): {file_path}")
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
                self.logger.info(f"Processing with OpenAI (with PDF): {file_path}")
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
        """Get a summary of token usage across all API calls"""
        return self.token_counter.get_usage_summary()
    
