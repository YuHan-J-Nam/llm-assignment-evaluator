"""
Constants and configuration for the educational assessment system.
Centralizes all templates, schemas, and configuration values.
"""

# Model Configuration - Updated to match llm_api_client.py MODEL_DICT
MODEL_OPTIONS = {
    'GEMINI': [
        'gemini-2.5-pro',
        'gemini-2.5-flash',
        'gemini-2.5-flash-lite-preview-06-17',
        'gemini-2.0-flash',
        'gemini-2.0-flash-lite',
        'None'
    ],
    'CLAUDE': [
        'claude-opus-4-20250514',
        'claude-sonnet-4-20250514',
        'claude-3-7-sonnet-20250219',
        'claude-3-5-haiku-20241022',
        'claude-3-5-sonnet-20241022',
        'claude-3-opus-20240229',
        'claude-3-haiku-20240307',
        'None'
    ],
    'OPENAI': [
        'gpt-4.1',
        'gpt-4.1-mini',
        'gpt-4.1-nano',
        'gpt-4o',
        'gpt-4o-mini',
        'o4-mini',
        'o3-mini',
        'None'
    ]
}

MODEL_DEFAULTS = {
    'GEMINI': 'gemini-2.0-flash-lite',
    'CLAUDE': 'claude-3-5-haiku-20241022',
    'OPENAI': 'gpt-4.1-nano'
}

# Thinking/Reasoning Models - Models that support enhanced reasoning
THINKING_MODELS = {
    'GEMINI': [
        'gemini-2.5-pro',
        'gemini-2.5-flash',
        'gemini-2.5-flash-lite-preview-06-17'
    ],
    'CLAUDE': [
        'claude-opus-4-20250514',
        'claude-sonnet-4-20250514',
        'claude-3-7-sonnet-20250219'
    ],
    'OPENAI': [
        'o4-mini',
        'o3-mini'
    ]
}

# File Paths
DIRECTORIES = {
    'CHECKLISTS': './checklists',
    'EVALUATIONS': './evaluations',
    'TEMP': './temp',
    'SUMMARY': './summary'
}

# Assessment Types
ASSESSMENT_TYPES = ['찬성반대', '독서와 작문', '나의 위인전']
GRADE_OPTIONS = ['고등학교 1학년', '고등학교 2학년', '고등학교 3학년']

# Default Templates
DEFAULT_SYSTEM_INSTRUCTION_EVALUATION = """
너는 선생님이다. 학생이 제출한 수행평가 과제에 대하여, 각 평가항목을 기반으로 논리적으로 평가하라. 
학생의 점수와 왜 그 점수를 받았는지에 대해 서술하고, 가능하다면 학생이 작성한 과제중 관련 텍스트를 증거로 제시하라.

수행평가에 대한 세부정보는 다음과 같다:

학년: [학년]
과목: [과목]
수행평가 제목: [수행평가 제목]
수행평가 유형: [수행평가 유형]
수행평가 설명: [수행평가 설명]

최종 평가내용을 JSON 형식으로 반환하라.
"""

DEFAULT_PROMPT_EVALUATION = """
다음은 학생이 제출한 수행평가 과제이다. 수행평가 과제에 대한 세부정보와 평가 기준을 고려하여, 각 평가항목에 대하여 논리적으로 평가하라.

평가 기준은 다음과 같다:
[평가 기준]

학생의 수행평가 과제는 다음과 같다:
[학생 제출물]
"""

DEFAULT_SYSTEM_INSTRUCTION_CHECKLIST = """
너는 [과목] 과목의 교사다. 학생들에게 [수행평가 유형] 형태의 수행평가 과제를 부여했다.
사용자가 제시한 수행평가 제목과 설명에 대해서 수행평가 기준을 생성하는 역할을 수행하라.

이 수행평가 과제를 공정하고 체계적으로 평가하기 위해, 다음 조건에 맞는 평가 기준(또는 체크리스트)을 생성하라. 평가기준 체크리스트를 생성할 때, 아래 기준에 맞추어 어떤 평가 기준을 만드는 것이 좋을 지 차근차근 생각해봐라.

1. 4~6개의 평가 항목을 제시할 것
2. 각 항목에는 명확한 평가 목적을 반영할 것 (예: 논리성, 창의성, 과제 이해 등)
3. 학생과 교사가 모두 이해하기 쉬운 언어로 작성할 것
4. 가능하면 과목 및 수행평가 유형의 특성을 반영할 것
5. 체크리스트는 아래와 같은 JSON 형식의 구조로 응답할 것:
    
```json
{
    "checklist": [
    {
        "title": "표현",
        "subcategories": [
        {
            "name": "문법의 정확성",
            "description": "문법의 정확성을 평가",
            "levels": {
            "high": "글이 문법적 오류 없이 완벽함",
            "medium": "글에 문법적 오류가 일부 있음",
            "low": "글에 문법적 오류가 다소 많음"
            }
        }
        ]
    }
    ]
}
```
"""

DEFAULT_PROMPT_CHECKLIST = """
다음 자료를 바탕으로 '[수행평가 제목]' 수행평가에 대한 평가 기준(체크리스트)을 생성해주세요.
이 수행평가는 [과목] 과목의 [수행평가 유형] 유형의 수행평가 입니다.
수행평가에 대한 설명은 다음과 같습니다:
[수행평가 설명]
"""

DEFAULT_SYSTEM_INSTRUCTION_SUMMARIZE = """
너는 고등학교 [과목] 교사이며, 학생이 제출한 '[수행평가 제목]' 수행평가 과제물을 평가 기준에 따라 요약해야 한다.

요약 작성 시 다음과 같은 사고 흐름을 따라 단계적으로 판단하라:

1. 각 항목에 대해 과제물 내에 **관련 정보가 명시적으로 언급되었는지**를 먼저 확인한다.

2. 다음 항목은 **내용이 조금이라도 모호하거나 유추가 필요한 경우, 반드시 'null'로 처리한다**:
    - **탐구_주제_선정_동기**: 주제를 선택하게 된 계기나 동기를 의미함.
    - **탐구_이후의_방향성**: 해당 탐구 이후의 학습 계획이나 학문적 발전 방향을 의미함.  

3. 다음 항목은 글 전체의 흐름과 문맥을 바탕으로 **적절한 범위 내에서 요약을 허용**한다:
    - **탐구_과정**: 주제에 대해 어떤 방식으로 탐구하고 학습했는지를 의미함.
    - **탐구_과정에서_배우고_느낀_점**: 탐구 과정에서 새롭게 알게 된 사실이나 느낀 점을 의미함.
    ※ 단, **과도한 해석이나 학생 의도와 다른 의미 부여는 금지함**.

기타 유의사항:
- 과제물에 **명확히 표현된 내용만 요약** 대상이며, 이외 정보는 포함하지 않음.
- 최종 출력은 **JSON 형식**으로 작성하고, 모든 문장은 '-음', '-임'으로 끝맺음.

수행평가 정보:
- 학년: [학년]
- 과목: [과목]
- 수행평가 제목: [수행평가 제목]
- 수행평가 유형: [수행평가 유형]
- 수행평가 설명: [수행평가 설명]
"""

DEFAULT_PROMPT_SUMMARIZE = """
다음은 학생이 제출한 수행평가 과제물이다.

시스템 지침에 따라 아래 네 항목의 내용을 요약하라.  
내용이 **지침의 기준에 부합하지 않을 경우** 반드시 'null'로 처리해야 하며,  
출력은 **JSON 형식**으로 하되, 모든 문장은 '-음', '-임' 형태로 마무리할 것.

[학생 제출 과제물]:
[학생 제출물]
"""

# Schema definitions
CHECKLIST_SCHEMA = {
    "type": "object",
    "properties": {
        "checklist": {
            "type": "array",
            "description": "체크리스트 대분류 항목들",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "대분류 제목"},
                    "subcategories": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "소분류 제목"},
                                "description": {"type": "string", "description": "소분류 설명"},
                                "levels": {
                                    "type": "object",
                                    "properties": {
                                        "high": {"type": "string", "description": "상 수준 설명"},
                                        "medium": {"type": "string", "description": "중 수준 설명"},
                                        "low": {"type": "string", "description": "하 수준 설명"}
                                    },
                                    "required": ["high", "medium", "low"]
                                }
                            },
                            "required": ["name", "description", "levels"]
                        }
                    }
                },
                "required": ["title", "subcategories"]
            }
        }
    },
    "required": ["checklist"]
}

EVALUATION_SCHEMA = {
    "type": "object",
    "properties": {
        "evaluation": {
            "type": "array",
            "description": "평가 항목별 점수와 이유",
            "items": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "평가 대분류"
                    },
                    "subcategories": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "평가 소분류"
                                },
                                "score": {
                                    "type": "integer",
                                    "description": "0~3 사이의 점수"
                                },
                                "reason": {
                                    "type": "string",
                                    "description": "점수 평가 이유"
                                },
                                "evidence":{
                                    "type": "array",
                                    "description": "관련 있는 텍스트를 증거로 제시",
                                    "items": {
                                        "type": "string",
                                        "description": "증거 텍스트"
                                    }
                                }
                            },
                            "required": ["name", "score", "reason", "evidence"]
                        }
                    }
                },
                "required": ["category", "subcategories"]
            }
        },
        "overall_feedback": {
            "type": "string",
            "description": "전체적인 피드백"
        }
    },
    "required": ["evaluation", "overall_feedback"]
}

SUMMARIZE_SCHEMA = {
    "type": "object",
    "properties": {
        "탐구_주제_선정_동기": {
            "type": "object",
            "properties": {
                "요약": {
                    "type": "string",
                    "nullable": True,
                    "description": "탐구 주제를 선택하게 된 동기에 대한 간략한 요약 (null 허용)"
                }
            },
            "required": ["요약"]
        },
        "탐구_과정": {
            "type": "object",
            "properties": {
                "요약": {
                    "type": "string",
                    "nullable": True,
                    "description": "탐구를 수행한 과정에 대한 요약 (null 허용)"
                }
            },
            "required": ["요약"]
        },
        "탐구_과정에서_배우고_느낀_점": {
            "type": "object",
            "properties": {
                "요약": {
                    "type": "string",
                    "nullable": True,
                    "description": "탐구 과정 중 배운 점이나 느낀 점의 요약 (null 허용)"
                }
            },
            "required": ["요약"]
        },
        "탐구_이후의_방향성": {
            "type": "object",
            "properties": {
                "요약": {
                    "type": "string",
                    "nullable": True,
                    "description": "탐구 이후의 발전 방향이나 후속 계획에 대한 요약 (null 허용)"
                }
            },
            "required": ["요약"]
        }
    },
    "required": [
        "탐구_주제_선정_동기",
        "탐구_과정",
        "탐구_과정에서_배우고_느낀_점",
        "탐구_이후의_방향성"
    ]
}
