import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# CLIP 토큰 제한 (참고용)
CLIP_TOKEN_LIMIT = 77
# GPT에게 개선된 프롬프트 생성 시 요청할 목표 토큰 수 (CLIP_TOKEN_LIMIT보다 약간 작게 설정)
TARGET_TOKEN_LIMIT_FOR_GPT_OUTPUT = 75

def load_context(prompt_path: str, comparison_path: str) -> dict:
    """
    원본 프롬프트와 객체 비교 분석 결과를 로드하여 컨텍스트 딕셔너리를 생성합니다.
    """
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_text = f.read().strip()

    with open(comparison_path, "r", encoding="utf-8") as f:
        comparison_data = json.load(f)

    return {
        "prompt": prompt_text,
        "missing_objects": comparison_data.get("missing_objects", []),
        "appeared_objects": comparison_data.get("appeared_objects", []),
        "detected_objects": comparison_data.get("detected_objects", []), # YOLO가 탐지한 모든 객체 리스트
        "prompt_objects": comparison_data.get("prompt_objects", [])     # 프롬프트에서 추출된 명사 키워드
    }

def create_prompt(context: dict) -> str:
    """
    GPT에게 전달할 전체 지시사항(프롬프트)을 생성합니다.
    개선된 프롬프트 생성 시 토큰 수 제한 지침을 포함합니다.
    """
    original_prompt = context["prompt"]
    prompt_objects_str = ", ".join(context["prompt_objects"]) if context["prompt_objects"] else "None"
    # detected_objects_str = ", ".join(context["detected_objects"]) if context["detected_objects"] else "None" # 이 정보는 때로 너무 길 수 있음
    missing_objects_str = ", ".join(context["missing_objects"]) if context["missing_objects"] else "None"

    # GPT에게 전달할 지시사항에 토큰 수 제한 내용을 명시적으로 추가
    token_constraint_instruction = f"IMPORTANT: The 'improved version of the prompt' (task 2) itself must be concise and result in **fewer than {TARGET_TOKEN_LIMIT_FOR_GPT_OUTPUT} tokens** when tokenized by OpenAI's CLIP model (which has a hard limit of {CLIP_TOKEN_LIMIT} tokens). Focus on critical visual details for the improved prompt and avoid unnecessary verbosity. Output only the prompt text for this part."

    # 누락된 객체와 프롬프트 내 객체 정보를 활용하여 GPT에게 더 명확한 지시 제공
    # detected_objects_str 대신 appeared_objects를 사용하는 것이 피드백에 더 적절할 수 있음
    appeared_objects_str = ", ".join(context["appeared_objects"]) if context["appeared_objects"] else "None"


    return f"""
You are an expert AI assistant specializing in improving prompts for video generation.
Based on the analysis of a previously generated video, your task is to provide feedback and a refined prompt.

Original User Prompt:
"{original_prompt}"

Analysis:
- Objects mentioned in the original prompt: {prompt_objects_str}
- Objects that appeared in the video: {appeared_objects_str}
- Objects from the prompt that were missing in the video: {missing_objects_str}

Please perform the following tasks:

1.  Provide concise, natural-language feedback to the user. Explain which key objects from their prompt were missing in the generated video and suggest potential reasons why they might not have appeared (e.g., ambiguity in the prompt, complexity, model limitations).
2.  Then, suggest an "Improved Prompt:". This improved prompt should aim to address the issues, be clearer, more descriptive for visual elements, and increase the likelihood of all desired objects appearing in the video.
    {token_constraint_instruction}

Return your response structured with clear headings for "Feedback:" and "Improved Prompt:".
""".strip()

def call_gpt(full_prompt_for_gpt: str) -> str:
    """
    OpenAI GPT 모델을 호출하여 응답을 반환합니다.
    """
    load_dotenv() # .env 파일에서 환경 변수 로드
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        error_message = "[WARNING] OpenAI API Key not found in .env file. GPT-based prompt improvement cannot proceed."
        print(error_message)
        # API 키가 없을 경우, 사용자에게 안내할 수 있는 기본 텍스트 반환
        return "Feedback:\nCould not generate feedback because the OpenAI API Key is not configured. Please check your .env file.\n\nImproved Prompt:\nN/A"

    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # 필요에 따라 "gpt-4" 등으로 변경 가능
            messages=[
                {"role": "system", "content": "You are an expert assistant for analyzing and improving video generation prompts. You provide structured feedback and a revised prompt according to user instructions, including any specified length or token constraints for the revised prompt."},
                {"role": "user", "content": full_prompt_for_gpt}
            ],
            temperature=0.6, # 약간의 창의성을 허용하되, 너무 벗어나지 않도록 조절
            max_tokens=600  # GPT 응답의 전체 최대 길이 (피드백 + 개선된 프롬프트)
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        error_message = f"[ERROR] Error calling OpenAI API: {e}"
        print(error_message)
        return f"Feedback:\nAn error occurred while communicating with the OpenAI API: {e}\n\nImproved Prompt:\nN/A"

def save_output(text_to_save: str, output_file_path: str):
    """
    주어진 텍스트를 지정된 경로에 파일로 저장합니다.
    """
    try:
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True) # 저장 경로의 디렉토리가 없으면 생성
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(text_to_save)
        # 저장 완료 메시지는 run_pipeline.py의 log_message를 사용하거나 여기서 직접 출력 가능
        # print(f"[✅] 개선된 피드백 및 프롬프트 저장 완료 → {output_file_path}") # run_pipeline.py에서 로깅하므로 중복될 수 있음
    except Exception as e:
        print(f"[ERROR] Failed to save output to {output_file_path}: {e}")

# 이 스크립트가 직접 실행될 때의 로직은 `run_pipeline.py`에서 모듈로 호출되므로 제거합니다.
# 테스트나 개별 실행이 필요하다면 별도의 테스트 스크립트에서 이 모듈의 함수들을 호출하는 방식을 권장합니다.