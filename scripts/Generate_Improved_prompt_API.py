import os
import json
from openai import OpenAI
from dotenv import load_dotenv

def load_context(prompt_path: str, comparison_path: str) -> dict:
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_text = f.read().strip()

    with open(comparison_path, "r", encoding="utf-8") as f:
        comparison = json.load(f)

    return {
        "prompt": prompt_text,
        "missing_objects": comparison.get("missing_objects", []),
        "appeared_objects": comparison.get("appeared_objects", []),
        "detected_objects": comparison.get("detected_objects", []),
        "prompt_objects": comparison.get("prompt_objects", [])
    }

def create_prompt(context: dict) -> str:
    prompt = context["prompt"]
    prompt_objects = ", ".join(context["prompt_objects"])
    detected = ", ".join(context["detected_objects"])
    missing = ", ".join(context["missing_objects"]) or "None"

    return f"""
A user entered the following prompt for AI video generation:

"{prompt}"

The objects mentioned in the prompt are: {prompt_objects}
The detected objects in the video are: {detected}
The missing objects are: {missing}

Please perform the following tasks:

1. Give natural-language feedback about which objects were missing and why they may not have appeared.
2. Then, suggest an improved version of the prompt that increases the chance of generating a better video, including clearer visual descriptions.

Return both the feedback and the improved prompt.
""".strip()

def call_gpt(full_prompt: str) -> str:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert assistant for improving video generation prompts."},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()

def save_output(text: str, path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[✅] 개선된 피드백 및 프롬프트 저장 완료 → {path}")

if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompt_path = os.path.join(base, "data", "prompts", "prompt1.txt")
    comparison_path = os.path.join(base, "data", "analysis_results", "video1_object_comparison.json")
    output_path = os.path.join(base, "data", "analysis_results", "video1_feedback_and_revised_prompt.txt")

    context = load_context(prompt_path, comparison_path)
    prompt_text = create_prompt(context)
    gpt_response = call_gpt(prompt_text)
    save_output(gpt_response, output_path)
