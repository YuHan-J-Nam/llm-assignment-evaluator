import json
import logging
from typing import Dict, Any, Optional, Union, List
from datetime import datetime

class TokenCounter:
    """Module for counting tokens used in requests to different LLM APIs"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger("token_counter")
        self.usage_log = []
        
        # Token cost per 1K tokens (input/output) for reference
        self.token_costs = {
            "o4-mini": {"input": 1.1, "output": 4.4},
            "gpt-4.1": {"input": 2, "output": 8},
            "gpt-4.1-mini": {"input": 0.4, "output": 1.6},
            "claude-3-7-sonnet-20250219": {"input": 3, "output": 15},
            "claude-3-5-haiku-20241022": {"input": 0.3, "output": 1.25},
            "claude-3-opus-20240229": {"input": 15, "output": 75},
            "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
            "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
            "gemini-2.5-flash-preview-04-17": {"input": 0.15, "output": 0.60},
        }
    
    def log_token_usage(self, model_name: str, prompt_tokens: int = 0, 
                        completion_tokens: int = 0, total_tokens: int = 0) -> Dict[str, Any]:
        """Log token usage for a request/response cycle"""
        
        # Create usage entry
        usage_entry = {
            "model": model_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens or (prompt_tokens + completion_tokens),
            "timestamp": datetime.now().isoformat()
        }
        
        # Estimate cost if available for this model
        base_model = self._get_base_model_name(model_name)
        if base_model in self.token_costs:
            costs = self.token_costs[base_model]
            input_cost = (usage_entry["prompt_tokens"] / 1000000) * costs["input"]
            output_cost = (usage_entry["completion_tokens"] / 1000000) * costs["output"]
            usage_entry["estimated_cost"] = input_cost + output_cost
        
        # Add to usage log
        self.usage_log.append(usage_entry)
        
        # Log token usage to the existing logger
        self.logger.info(f"Token usage for {model_name}: {usage_entry['total_tokens']} tokens "
                         f"(prompt: {prompt_tokens}, completion: {completion_tokens})")
        if "estimated_cost" in usage_entry:
            self.logger.info(f"Estimated cost: ${usage_entry['estimated_cost']:.6f}")
        
        # Log structured token data for potential parsing
        log_data = {
            "token_usage": {
                "model": model_name,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": usage_entry["total_tokens"]
            }
        }
        if "estimated_cost" in usage_entry:
            log_data["token_usage"]["estimated_cost"] = usage_entry["estimated_cost"]
            
        self.logger.info(f"TOKEN_DATA: {json.dumps(log_data)}")
        
        return usage_entry
    
    def extract_token_usage_from_openai_response(self, response, model_name: str) -> Dict[str, int]:
        """Extract token usage data from an OpenAI API response"""
        # This is a sample usage format for OpenAI API responses
        # usage=ResponseUsage(input_tokens=614, input_tokens_details=InputTokensDetails(cached_tokens=0), output_tokens=142, output_tokens_details=OutputTokensDetails(reasoning_tokens=0), total_tokens=756)
        if hasattr(response, 'usage'):
            # OpenAI API returns input_tokens, output_tokens, and total_tokens
            prompt_tokens = getattr(response.usage, 'input_tokens', 0)
            if hasattr(response.usage, 'input_tokens_details'):
                cached_tokens = getattr(response.usage.input_tokens_details, 'cached_tokens', 0)
            completion_tokens = getattr(response.usage, 'output_tokens', 0)
            if hasattr(response.usage, 'output_tokens_details'):
                reasoning_tokens = getattr(response.usage.output_tokens_details, 'reasoning_tokens', 0)
            total_tokens = getattr(response.usage, 'total_tokens', 0) or (prompt_tokens + completion_tokens)
            
            return {
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached_tokens,
                "completion_tokens": completion_tokens,
                "reasoning_tokens": reasoning_tokens,
                "total_tokens": total_tokens
            }
    
    def extract_token_usage_from_claude_response(self, response, model_name: str) -> Dict[str, int]:
        """Extract token usage data from a Claude API response"""
        if hasattr(response, 'usage'):
            # Claude API returns input_tokens and output_tokens
            prompt_tokens = getattr(response.usage, 'input_tokens', 0)
            completion_tokens = getattr(response.usage, 'output_tokens', 0)
            total_tokens = prompt_tokens + completion_tokens
            
            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    def extract_token_usage_from_gemini_response(self, response, model_name: str) -> Dict[str, int]:
        """Extract token usage data from a Gemini API response"""
        if hasattr(response, 'usage_metadata'):
            # Gemini API returns prompt_token_count, candidates_token_count, and total_token_count
            prompt_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
            completion_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
            total_tokens = getattr(response.usage_metadata, 'total_token_count', 0) or (prompt_tokens + completion_tokens)
            
            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    def _get_base_model_name(self, full_model_name: str) -> str:
        """Extract base model name for cost calculation"""
        return full_model_name
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get a summary of token usage across all requests"""
        if not self.usage_log:
            return {"total_requests": 0}
        
        total_tokens = sum(entry["total_tokens"] for entry in self.usage_log)
        total_prompt_tokens = sum(entry["prompt_tokens"] for entry in self.usage_log)
        total_completion_tokens = sum(entry["completion_tokens"] for entry in self.usage_log)
        
        # Group by model
        models = {}
        for entry in self.usage_log:
            model = entry["model"]
            if model not in models:
                models[model] = {
                    "requests": 0,
                    "total_tokens": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "estimated_cost": 0.0
                }
            
            models[model]["requests"] += 1
            models[model]["total_tokens"] += entry["total_tokens"]
            models[model]["prompt_tokens"] += entry["prompt_tokens"]
            models[model]["completion_tokens"] += entry["completion_tokens"]
            if "estimated_cost" in entry:
                models[model]["estimated_cost"] += entry["estimated_cost"]
        
        # Calculate total estimated cost
        total_cost = sum(
            entry.get("estimated_cost", 0) 
            for entry in self.usage_log 
            if "estimated_cost" in entry
        )
        
        summary = {
            "total_requests": len(self.usage_log),
            "total_tokens": total_tokens,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_estimated_cost": total_cost,
            "models": models
        }
        
        # Log the summary to the existing log
        self.logger.info(f"Token usage summary: {len(self.usage_log)} requests, "
                        f"{total_tokens} total tokens, estimated cost ${total_cost:.6f}")
        
        return summary 