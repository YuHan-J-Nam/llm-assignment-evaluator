import os
from datetime import datetime
from dotenv import load_dotenv
from llm_api_client import LLMAPIClient
import logging
import json

# Load environment variables
load_dotenv()

def create_system_instruction(subject, assessment_type, assessment_title, assessment_description):
    """Create a formatted system instruction with user inputs"""
    template = """
    너는 [과목] 과목의 교사다. 학생들에게 [수행평가 유형] 형태의 수행평가 과제를 부여했다.
    사용자가 제시한 수행평가 제목과 설명에 대해서 수행평가 기준을 생성하는 역할을 수행하라.

    이 수행평가 과제를 공정하고 체계적으로 평가하기 위해, 다음 조건에 맞는 평가 기준(또는 체크리스트)을 생성하라. 평가기준 체크리스트를 생성할 때, 아래 기준에 맞추어 어떤 평가 기준을 만드는 것이 좋을 지 차근차근 생각해봐라.

    1. 4~6개의 평가 항목을 제시할 것
    2. 각 항목에는 명확한 평가 목적을 반영할 것 (예: 논리성, 창의성, 과제 이해 등)
    3. 학생과 교사가 모두 이해하기 쉬운 언어로 작성할 것
    4. 가능하면 과목 및 수행평가 유형의 특성을 반영할 것
    5. 체크리스트는 아래와 같은 JSON 형식의 구조로 응답할 것:
        
        ```
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
    
    # Replace placeholders with user inputs
    formatted_instruction = template.replace('[과목]', subject)
    formatted_instruction = formatted_instruction.replace('[수행평가 유형]', assessment_type)
    formatted_instruction = formatted_instruction.replace('[수행평가 제목]', assessment_title)
    formatted_instruction = formatted_instruction.replace('[수행평가 설명]', assessment_description)
    
    return formatted_instruction

def main():
    # Initialize the unified API client
    client = LLMAPIClient(log_level=logging.INFO)
    
    # Path to a PDF file for testing
    pdf_path = None
    # pdf_path = "C:/Users/yuhan/Desktop/CSD_18기/베어러블/독후감 예시/서평_39개_pdf/7막 7장 그리고 그 후.pdf"

    # Checklist 저장 경로 생성
    checklist_path = "./checklists"
    if not os.path.exists(checklist_path):
        os.makedirs(checklist_path)
    
    
    # Get user inputs for system instruction placeholders
    subject = input("과목을 입력하세요: ")
    assessment_type = input("수행평가 유형을 입력하세요: ")
    assessment_title = input("수행평가 제목을 입력하세요: ")
    assessment_description = input("수행평가 설명을 입력하세요: ")
    
    # Create the formatted system instruction
    system_instruction = create_system_instruction(
        subject, 
        assessment_type, 
        assessment_title, 
        assessment_description
    )
    
    # Define a prompt
    prompt = f"""
    다음 자료를 바탕으로 '{assessment_title}' 수행평가에 대한 평가 기준(체크리스트)을 생성해주세요.
    이 수행평가는 {subject} 과목의 {assessment_type} 유형의 수행평가 입니다.
    수행평가에 대한 설명은 다음과 같습니다:
    {assessment_description}
    """
    
    # Define a custom response schema
    custom_schema = {
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
            },
        },
        "required": ["checklist"]
    }
    
    try:
        print("\nSystem Instruction:")
        print(system_instruction)
        print("\nPrompt:")
        print(prompt)
        
        # Process with Gemini
        print("\nProcessing with Gemini...")
        gemini_response = client.process_pdf_with_gemini(
            file_path=pdf_path,
            prompt=prompt,
            model_name="gemini-2.0-flash",
            schema=custom_schema,
            system_instruction=system_instruction
        )
        gemini_response_text = gemini_response.text
        print(f"Gemini Response: {gemini_response_text}")
        
        # Process with Claude
        print("\nProcessing with Claude...")
        claude_response = client.process_pdf_with_claude(
            file_path=pdf_path,
            prompt=prompt,
            model_name="claude-3-7-sonnet-20250219",
            # schema=custom_schema,
            system_instruction=system_instruction
        )
        claude_response_text = claude_response.content[0].text
        print(f"Claude Response: {claude_response_text}")
        
        # Process with OpenAI
        print("\nProcessing with OpenAI...")
        openai_response = client.process_pdf_with_openai(
            file_path=pdf_path, 
            prompt=prompt,
            model_name="gpt-4.1",
            schema=custom_schema,
            system_instruction=system_instruction
        )
        openai_response_text = openai_response.choices[0].message.content
        print(f"OpenAI Response: {openai_response_text}")
        
        # Save the created checklists for each LLM
        for response_text, llm_name in zip([gemini_response_text, claude_response_text, openai_response_text], ["gemini", "claude", "openai"]):
            if response_text:
                if input(f"\n{llm_name}로 생성된 체크리스트를 저장하시겠습니까? (y/n): ").lower() == 'y':
                    result_file = f"{checklist_path}/{llm_name}_평가기준_{assessment_title}_시간_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(result_file, 'w', encoding='utf-8') as f:
                        f.write(response_text)
                    print(f"{llm_name} 생성 평가기준이 {result_file}에 저장되었습니다.")

        # Print token usage summary
        print("\nToken Usage Summary:")
        usage_summary = client.get_token_usage_summary()
        print(json.dumps(usage_summary, indent=2))
        
        # Token usage is now logged directly to the log files
        print("\nToken usage information logged to API logs")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()