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
    'ANTHROPIC': [
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
    'ANTHROPIC': 'claude-3-5-haiku-20241022',
    'OPENAI': 'gpt-4.1-nano'
}

# Thinking/Reasoning Models - Models that support enhanced reasoning
THINKING_MODELS = {
    'GEMINI': [
        'gemini-2.5-pro',
        'gemini-2.5-flash',
        'gemini-2.5-flash-lite-preview-06-17'
    ],
    'ANTHROPIC': [
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
ASSESSMENT_TYPES = ['찬성반대', '설명글', '대안제시', '글짓기', '주장']
GRADE_OPTIONS = ['고등학교 1학년', '고등학교 2학년', '고등학교 3학년']

# Default Templates
DEFAULT_SYSTEM_INSTRUCTION_EVALUATION = """너는 고등학교 {과목} 담당 교사이며, 학생이 제출한 {수행평가_제목} 수행평가 과제물을 객관적이고 공정하게 평가하는 전문 평가자 역할을 수행한다. 다음 지침을 명확히 숙지하여 평가하라.
수행평가에 대한 세부정보는 다음과 같다:

학년: {학년}
수행평가 제목: {수행평가_제목}
수행평가 유형: {수행평가_유형}
수행평가 설명: {수행평가_설명}
평가 기준:
{평가_기준}

평가 수행 지침
1. 평가 기준 엄격 준수
- 제공된 평가 기준 및 체크리스트의 항목을 정확히 준수하여 평가하라. 반드시 모든 항목 및 세부항목에 대해 평가하고, 누락된 항목이 없도록 하라.

2. 직접적 인용 및 명확한 설명
- 학생의 제출물에서 인용된 표현을 근거로 사용한다면 해당 표현이 평가 기준에 적합하거나 미흡한 이유를 구체적으로 설명하라.

3. 논리적이고 객관적인 근거
- 평가 결과가 일관되도록 하고, 점수 부여의 근거를 명확히 기술하여 객관성과 신뢰성을 높이도록 하라.

결과는 JSON 객체로만 출력하라. 마크다운 코드 펜스(```)는 포함하지 말고, 인용된 표현을 위해 문자열 내부에 큰따옴표(")를 포함해야 할 경우 반드시 \"처럼 이스케이프 처리하라"""

DEFAULT_PROMPT_EVALUATION = """
다음은 학생이 제출한 수행평가 과제물이다. 제시된 평가 기준을 참고하여 과제물을 체계적으로 분석하고 평가하라. 평가 시 다음 단계를 순차적으로 따르라:
1. 평가기준을 꼼꼼히 읽고 이해하라.
2. 학생의 제출물을 세심히 분석하며, 평가 기준에 부합하거나 미흡한 점이 있는지 확인하라. 만약 직접적으로 인용할 수 있는 표현이 있다면, 해당 표현을 정확히 인용하라.
3. 인용한 표현이 있다면 평가 기준과의 관계를 논리적으로 설명하라.
4. 평가 항목별로 정량적 점수를 부여하며 근거를 구체적으로 기술하라.

학생 제출물:
{학생_제출물}"""

DEFAULT_SYSTEM_INSTRUCTION_CHECKLIST = """
너는 {과목} 과목의 교사다. 학생들에게 {수행평가_유형} 형태의 수행평가 과제를 부여했다.
사용자가 제시한 수행평가 제목과 설명에 대해서 수행평가 기준을 생성하는 역할을 수행하라.

이 수행평가 과제를 공정하고 체계적으로 평가하기 위해, 다음 조건에 맞는 평가 기준(또는 체크리스트)을 생성하라. 평가기준 체크리스트를 생성할 때, 아래 기준에 맞추어 어떤 평가 기준을 만드는 것이 좋을 지 차근차근 생각해봐라.

1. 10개 이상의 평가 항목(name)을 제시할 것
2. 아래 예시와 같이 평가 항목 하나당 2개의 기준(주제 제시, 문제의식 제시)을 제시할 것
3. 2개의 평가기준을을 모두 충족하면 3점, 하나만 충족하면 2점, 충족하지 못하면 1점, 백지 제출이면 0점으로 평가할 수 있도록 점수 체계를 설정할 것
4. 각 항목에는 명확한 평가 목적을 반영할 것 (예: 논리성, 창의성, 과제 이해 등)
5. 학생과 교사가 모두 이해하기 쉬운 언어로 작성할 것
6. 가능하면 과목 및 수행평가 유형의 특성을 반영할 것
7. 가능한 체크리스트끼리 의미가 중복되지 않도록 할 것
8. 분량에 대해서 제한을 둘 경우 분량 평가항목을 무조건 포함하며, 충족하면 3점 아니면 1점, 10자 이하면 0점 줄 것 
9. 문법 및 맞춤법에 관련하여 평가항목을 1개 만들고, 3개 이하 3점, 4개 이상 5개 이하 2점, 6개 이상 10개 이하 1점, 11개 이상 0점을 줄 것 
10. 교사가 명시한 조건은 포함해라
11. 체크리스트는 아래와 같은 JSON 형식의 구조로 응답할 것
12. 아래 예시는 체크리스트의 예시로, 이와 유사한 형식 및 JSON 형식의 구조로 작성할 것
```
"checklist": [{
    "name": "주제 및 문제의식 제시",
    "description": "작품이나 논제를 정확히 이해하고 글의 서두에서 자신의 관점을 명확히 밝혔는가?",
    "levels": {
        "3": "주제 파악과 관점 제시가 모두 선명하다.",
        "2": "두가지 기준 중 하나만 뚜렷하다.",
        "1": "두가지 기준 모두 모호하다.",
        "0": "백지 제출인 경우"
    }
}]
```
"""

DEFAULT_PROMPT_CHECKLIST = """
다음 자료를 바탕으로 {수행평가_제목} 수행평가에 대한 평가 기준(checklist)을 생성해주세요.
이 수행평가는 {과목} 과목의 {수행평가_유형} 유형의 수행평가 입니다.
수행평가에 대한 설명은 다음과 같습니다:
{수행평가_설명}"""

DEFAULT_SYSTEM_INSTRUCTION_SUMMARIZE = """
너는 고등학교 {과목} 교사이며, 학생이 제출한 {수행평가_제목} 수행평가 과제물을 평가 기준에 따라 요약해야 한다.

요약 작성 시 다음과 같은 사고 흐름을 따라 단계적으로 판단하라:

1. 각 항목에 대해 과제물 내에 **관련 정보가 명시적으로 언급되었는지**를 먼저 확인한다.

2. 다음 항목은 **내용이 조금이라도 모호하거나 유추가 필요한 경우, 반드시 'null'로 처리한다**:
    - **탐구_주제_선정_동기**: 주제를 선택하게 된 계기나 동기를 의미함.
    - **탐구_이후의_방향성**: 해당 탐구 이후의 학습 계획이나 학문적 발전 방향, 진로 관련 내용을 의미함.  

3. 다음 항목은 글 전체의 흐름과 문맥을 바탕으로 **적절한 범위 내에서 요약을 허용**한다:
    - **탐구_과정**: 주제에 대해 어떤 방식으로 탐구하고 학습했는지를 의미함.
    - **탐구_과정에서_배우고_느낀_점**: 탐구 과정에서 새롭게 알게 된 사실이나 느낀 점을 의미함.
    ※ 단, **과도한 해석이나 학생 의도와 다른 의미 부여는 금지함**.

기타 유의사항:
- 과제물에 **명확히 표현된 내용만 요약** 대상이며, 이외 정보는 포함하지 않음.
- 최종 출력은 **JSON 형식**으로 작성하고, 모든 문장은 '-음', '-임'으로 끝맺음.

수행평가 정보:
- 학년: {학년}
- 과목: {과목}
- 수행평가 제목: {수행평가_제목}
- 수행평가 유형: {수행평가_유형}
- 수행평가 설명: {수행평가_설명}
"""

DEFAULT_PROMPT_SUMMARIZE = """
다음은 학생이 제출한 수행평가 과제물이다.

시스템 지침에 따라 아래 네 항목의 내용을 요약하라.  
내용이 **지침의 기준에 부합하지 않을 경우** 반드시 'null'로 처리해야 하며,  
출력은 **JSON 형식**으로 하되, 모든 문장은 '-음', '-임' 형태로 마무리할 것.

학생 제출 과제물:
{학생_제출물}
"""

# Schema definitions
CHECKLIST_SCHEMA = {
    "type": "object",
    "properties": {
        "checklist": {
            "type": "array",
            "description": "체크리스트 항목들",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "항목 제목"
                    },
                    "description": {
                        "type": "string",
                        "description": "항목 설명"
                    },
                    "levels": {
                        "type": "object",
                        "properties": {
                        "3": { "type": "string", "description": "3점 수준 설명" },
                        "2": { "type": "string", "description": "2점 수준 설명" },
                        "1": { "type": "string", "description": "1점 수준 설명" },
                        "0": { "type": "string", "description": "0점 수준 설명" }
                        },
                        "required": ["3", "2", "1", "0"]
                    }
                },
                "required": ["name", "description", "levels"]
            }
        }
    },    
    "required": ["checklist"]
}

# AI_Hub 제공 체크리스트 사용 평가용 스키마
EVALUATION_SCHEMA = {
    "type": "object",
    "properties": {
        "evaluation": {
            "type": "array",
            "description": "평가 항목별 점수와 이유",
            "items": {
                "type": "object",
                "properties": {
                    "title": {
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

EVALUATION_SCHEMA_V2 = {
    "type": "object",
    "properties": {
        "evaluation": {
            "type": "array",
            "description": "평가 항목별 점수와 이유",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "평가 항목 제목"
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
        "summary":{
            "type": "object",
            "properties": {
                "탐구_주제_선정_동기": {
                    "type": "string",
                    "nullable": True
                },
                "탐구_과정": {
                    "type": "string",
                    "nullable": True
                },
                "탐구_과정에서_배우고_느낀_점": {
                    "type": "string",
                    "nullable": True
                },
                "탐구_이후의_방향성": {
                    "type": "string",
                    "nullable": True
                }
            },
            "required": [
                "탐구_주제_선정_동기", "탐구_과정", "탐구_과정에서_배우고_느낀_점", "탐구_이후의_방향성"
            ]
        }
        
    },
    "required": ["summary"]
}