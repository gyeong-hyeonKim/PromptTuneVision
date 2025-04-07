import os
import json
from openai import OpenAI
from dotenv import load_dotenv

def load_context(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_prompt(context: dict) -> str:
    prompt = context.get("prompt", "Unknown prompt")
    prompt_objects = context.get("prompt_objects", [])
    detected_objects = context.get("detected_objects", [])
    missing_objects = context.get("missing_objects", [])

    prompt_text = f"""
Prompt: "{prompt}"

Objects mentioned in the prompt: {', '.join(prompt_objects)}
Objects detected in the video: {', '.join(detected_objects)}
Objects missing from the video: {', '.join(missing_objects) if missing_objects else 'None'}

Based on this information, write a feedback message in English that explains to the user which objects were missing in the generated video and suggest how to improve the prompt.
"""

    return prompt_text.strip()

def call_gpt(prompt_text: str) -> str:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    chat_response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # 또는 "gpt-4"
        messages=[
            {"role": "system", "content": "You are an assistant that helps users refine video generation prompts."},
            {"role": "user", "content": prompt_text}
        ],
        temperature=0.7,
        max_tokens=400
    )

    return chat_response.choices[0].message.content.strip()

def save_feedback(feedback_text: str, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(feedback_text)
    print(f"[✅] GPT 피드백 저장 완료 → {output_path}")

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    context_path = os.path.join(base_path, "data", "analysis_results", "video1_object_comparison.json")
    output_path = os.path.join(base_path, "data", "analysis_results", "video1_feedback_gpt.txt")

    context = load_context(context_path)
    prompt_text = generate_prompt(context)
    gpt_feedback = call_gpt(prompt_text)
    save_feedback(gpt_feedback, output_path)
