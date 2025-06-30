import re
import json5, jsonschema
from json_repair import repair_json
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
    
    def parse_response(self, response_text: str, schema: dict, model: str = None) -> dict:
        """
        응답 텍스트를 파싱하여 구조화된 데이터 추출
        
        Args:
            response_text: 파싱할 응답 텍스트
            schema: 검증에 사용할 JSON 스키마
            model: 모델명 (로깅용)
        
        Returns:
            파싱된 JSON 데이터 또는 에러 정보가 포함된 딕셔너리
        """
        if not response_text or not isinstance(response_text, str):
            return {"error": "잘못된 입력: response_text는 비어있지 않은 문자열이어야 합니다", "raw_text": response_text}
        
        # 코드 블록 제거
        cleaned_text = self._remove_code_blocks(response_text)
        
        # JSON 파싱 시도
        parsed_data = self._try_parse_json(cleaned_text, model)
        if "error" not in parsed_data:
            # 스키마 검증
            validation_error = self.validate_response(parsed_data, schema)
            if validation_error is None:
                return parsed_data
            else:
                self.logger.warning(f"{model}의 응답이 스키마와 맞지 않습니다: {validation_error}")
                return {"error": validation_error, "raw_text": response_text}
        
        # JSON 수리 시도
        return self._attempt_json_repair(cleaned_text, schema, model, response_text)


    def _remove_code_blocks(self, text: str) -> str:
        """코드 블록 마커 제거"""
        return re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()

    def _try_parse_json(self, text: str, model: str = None) -> dict:
        """JSON 파싱 시도"""
        try:
            return json5.loads(text)
        except ValueError as e:
            self.logger.warning(f"{model}의 응답이 유효한 JSON이 아닙니다: {str(e)}")
            return {"error": f"JSON 파싱 실패: {str(e)}"}

    def _attempt_json_repair(self, text: str, schema: dict, model: str = None, original_text: str = "") -> dict:
        """JSON 수리 시도"""
        try:
            repaired_text = repair_json(text, skip_json_loads=True)
            parsed_data = json5.loads(repaired_text)
            
            validation_error = self.validate_response(parsed_data, schema)
            if validation_error is None:
                self.logger.info(f"{model}의 응답을 성공적으로 수리했습니다")
                return parsed_data
            else:
                return {"error": validation_error, "raw_text": original_text}
                
        except ValueError as e:
            self.logger.error(f"JSON 수리 실패: {str(e)}")
            return {"error": f"JSON 수리 실패: {str(e)}", "raw_text": original_text}

    def validate_response(self, response: dict, schema: dict) -> str | None:
        """
        응답이 주어진 스키마에 맞는지 검증
        
        Args:
            response: 검증할 응답 데이터
            schema: JSON 스키마
        
        Returns:
            검증 에러 메시지 또는 None (성공시)
        """
        try:
            jsonschema.validate(instance=response, schema=schema)
            return None
        except jsonschema.ValidationError as e:
            error_msg = f"스키마 검증 실패: {e.message}"
            self.logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"스키마 검증 중 예기치 않은 오류 발생: {str(e)}"
            self.logger.error(error_msg)
            return error_msg