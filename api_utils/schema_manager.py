import re
import json5
import json
import logging
import copy

class ResponseSchemaManager:
    """다양한 LLM API에 대한 응답 스키마 관리"""
    
    def __init__(self):
        # 사용자 정의 가능한 공통 응답 스키마 템플릿
        self.default_schema = {
            "type": "object",
            "properties": {
                "response": {
                    "type": "string",
                    "description": "모델의 응답 텍스트"
                },
            "required": ["response"]
            }
        }
        self.logger = logging.getLogger(__name__)

    def _add_additional_properties_to_schema(self, schema):
        """스키마에서 required 필드 다음에 additionalProperties: False를 재귀적으로 추가"""
        try:
            if isinstance(schema, dict):
                # required 필드를 찾으면 동일한 수준에서 additionalProperties: False 추가
                if "required" in schema and "additionalProperties" not in schema:
                    schema["additionalProperties"] = False
                    
                # 딕셔너리의 모든 값을 재귀적으로 처리
                for key, value in schema.items():
                    schema[key] = self._add_additional_properties_to_schema(value)
            
            # 리스트인 경우 각 항목 처리
            elif isinstance(schema, list):
                schema = [self._add_additional_properties_to_schema(item) for item in schema]

        except Exception as e:
            self.logger.error(f"스키마에 additionalProperties 추가 중 오류 발생: {str(e)}")
            raise
    
        return schema
    
    def format_gemini_schema(self, custom_schema=None):
        """Gemini API용으로 포맷된 스키마 가져오기"""
        # 현재로서는 별도 요구사항이 없지만, 필요시 추가 가능
        schema = custom_schema if custom_schema else self.default_schema
        new_schema = copy.deepcopy(schema)
        return schema
    
    def format_anthropic_schema(self, custom_schema=None):
        """Anthropic API용으로 포맷된 스키마 가져오기"""
        # 현재로서는 별도 요구사항이 없지만, 필요시 추가 가능
        schema = custom_schema if custom_schema else self.default_schema
        new_schema = copy.deepcopy(schema)
        return schema
    
    def format_openai_schema(self, custom_schema=None):
        """OpenAI API용으로 포맷된 스키마 가져오기"""
        schema = custom_schema if custom_schema else self.default_schema
        new_schema = copy.deepcopy(schema)

        # 엄격한 스키마 검증을 위해 추가 속성 설정
        processed_schema = self._add_additional_properties_to_schema(new_schema)

        # OpenAI API용으로 적절하게 포맷된 스키마 생성
        return {
            "format": {
                "type": "json_schema",
                "name": "response_schema",
                "schema": processed_schema,
                "strict": True
            }
        }
    
    def parse_response(self, response_text, model: str = None):
        """응답 텍스트를 파싱하여 구조화된 데이터 추출"""
        try:
            # 텍스트가 ```json으로 시작하고 끝나는 경우 제거
            if response_text.startswith("```json") and response_text.endswith("```"):
                response_text = re.sub(r"^```json\s*|\s*```$", "", response_text, flags=re.DOTALL).strip()

            # JSON으로 파싱 시도
            return json5.loads(response_text)
        
        except json5.JSONDecodeError:
            logging.warning(f"{model}의 응답이 유효한 JSON이 아닙니다. 원시 텍스트를 반환합니다.")
            # 원시 텍스트를 구조화된 형식으로 반환
            return {
                "raw_text": response_text,
                "structured": False
            }