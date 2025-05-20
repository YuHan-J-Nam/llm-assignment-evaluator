import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv
from llm_api_client import LLMAPIClient
import logging

# Load environment variables
load_dotenv()

# Load checklists from JSON files
def load_checklist(file_path):
    """Load a checklist from a JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Load checklists
논술형_checklist = load_checklist('./checklists/논술형_checklist.json')
수필형_checklist = load_checklist('./checklists/수필형_checklist.json')

def load_student_submission(file_path):
    """Load student's submission from a text file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def create_system_instruction(grade, subject, assessment_title, assessment_type, assessment_description):
    """Create a formatted system instruction with user inputs"""
    template = """
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
    
    # Replace placeholders with user inputs
    formatted_instruction = template.replace('[학년]', grade)
    formatted_instruction = formatted_instruction.replace('[과목]', subject)
    formatted_instruction = formatted_instruction.replace('[수행평가 제목]', assessment_title)
    formatted_instruction = formatted_instruction.replace('[수행평가 유형]', assessment_type)
    formatted_instruction = formatted_instruction.replace('[수행평가 설명]', assessment_description)
    
    return formatted_instruction

def main():
    # Initialize the unified API client
    client = LLMAPIClient(log_level=logging.INFO)
    
    # Get user inputs for system instruction placeholders
    grade = input("학년을 입력하세요: ")
    subject = input("과목을 입력하세요: ")
    assessment_title = input("수행평가 제목을 입력하세요: ")
    assessment_type = input("수행평가 유형을 입력하세요: ")
    assessment_description = input("수행평가 설명을 입력하세요: ")
    
    # Choose evaluation criteria
    criteria = 논술형_checklist

    # Define evaluation directory
    evaluation_dir = "./evaluations"
    if not os.path.exists(evaluation_dir):
        os.makedirs(evaluation_dir)
        
    # Get student submission file
    submission_file = input("학생 제출물 파일 경로를 입력하세요: ")
    try:
        submission_content = load_student_submission(submission_file)
    except Exception as e:
        print(f"제출물 파일을 불러오는 중 오류가 발생했습니다: {str(e)}")
        submission_content = input("학생 제출물을 직접 입력하세요: ")
    
    # Create the formatted system instruction
    system_instruction = create_system_instruction(
        grade, 
        subject, 
        assessment_title, 
        assessment_type, 
        assessment_description
    )
    
    # Define a prompt
    prompt = f"""
    다음은 학생이 제출한 수행평가 과제이다. 수행평가 과제에 대한 세부정보와 평가 기준을 고려하여, 각 평가항목에 대하여 논리적으로 평가하라.
    
    평가 기준은 다음과 같다:
    {json.dumps(criteria, ensure_ascii=False, indent=2)}
    
    학생의 수행평가 과제는 다음과 같다:
    {submission_content}
    """
    
    # Define a custom response schema based on the criteria
    custom_schema = {
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
    
    gemini_response_text = None
    claude_response_text = None
    openai_response_text = None

    try:
        print("\nSystem Instruction:")
        print(system_instruction)
        print("\nPrompt:")
        print(prompt)
        
        # Process with Gemini
        # print("\nProcessing with Gemini...")
        # gemini_response = client.process_pdf_with_gemini(
        #     file_path = None,
        #     prompt=prompt,
        #     model_name="gemini-2.0-flash",  
        #     schema=custom_schema,
        #     system_instruction=system_instruction
        # )
        # gemini_response_text = gemini_response.text
        # print(f"Gemini Response: {gemini_response_text}")
        
        # Process with Claude
        print("\nProcessing with Claude...")
        claude_response = client.process_pdf_with_claude(
            file_path=None,
            prompt=prompt,
            model_name="claude-3-opus-20240229",
            schema=custom_schema,
            system_instruction=system_instruction
        )
        claude_response_text = claude_response.content[0].text

        # ```json과 ```` 사이에 있는 텍스트만 추출 (특정 모델에선 json 형식으로 반환되지 않음)
        if "```json" in claude_response_text:
            claude_response_text = re.search(r'```json\s*(.*?)\s*```', claude_response_text, re.DOTALL).group(1)

        print(f"Claude Response: {claude_response_text}")

        # Process with OpenAI
        # print("\nProcessing with OpenAI...")
        # openai_response = client.process_pdf_with_openai(
        #     file_path=None,
        #     prompt=prompt,
        #     model_name="gpt-4.1",
        #     schema=custom_schema,
        #     system_instruction=system_instruction
        # ) 
        # openai_response_text = openai_response.choices[0].message.content
        # print(f"OpenAI Response: {openai_response_text}")
        
        # Print token usage summary
        print("\nToken Usage Summary:")
        usage_summary = client.get_token_usage_summary()
        print(json.dumps(usage_summary, indent=2))
        
        # Save the evaluation result
        save_result = input("\n평가 결과를 저장하시겠습니까? (y/n): ").lower() == 'y'
        if save_result:
            for response_text, llm_name in zip([gemini_response_text, claude_response_text, openai_response_text], ["gemini", "claude", "openai"]):
                if response_text:
                    result_file = f"{evaluation_dir}/{llm_name}_평가결과_{assessment_title}_시간_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(result_file, 'w', encoding='utf-8') as f:
                        f.write(response_text)
                    print(f"평가 결과가 {result_file}에 저장되었습니다.")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()