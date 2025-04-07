import os
import json
import streamlit as st

# 📁 경로 설정
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

PROMPT_PATH = os.path.join(BASE_PATH, "data", "prompts", "prompt1.txt")
CLIP_PLOT_PATH = os.path.join(BASE_PATH, "data", "analysis_results", "video1_clip_plot.png")
YOLO_JSON_PATH = os.path.join(BASE_PATH, "data", "analysis_results", "video1_yolo.json")
COMPARISON_JSON_PATH = os.path.join(BASE_PATH, "data", "analysis_results", "video1_object_comparison.json")
GPT_FEEDBACK_PATH = os.path.join(BASE_PATH, "data", "analysis_results", "video1_feedback_gpt.txt")

# 🌟 타이틀
st.set_page_config(page_title="Prompt-TuneVision Dashboard", layout="wide")
st.title("🎬 Prompt-TuneVision : Prompt Evaluation Dashboard")

# 📌 1. 프롬프트 출력
st.header("📝 Prompt")
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    prompt_text = f.read().strip()
st.code(prompt_text, language="text")

# 📊 2. CLIP 유사도 플롯
st.header("📈 CLIP Similarity per Frame")
st.image(CLIP_PLOT_PATH, caption="CLIP 유사도 분석 결과", use_column_width=700)

# 🧠 3. YOLO 탐지 객체 요약
st.header("🔍 YOLO Detected Objects")
with open(YOLO_JSON_PATH, "r", encoding="utf-8") as f:
    yolo_data = json.load(f)

# 모든 프레임 객체 모으기
all_objects = []
for item in yolo_data:
    all_objects.extend(item["objects"])
object_counts = {obj: all_objects.count(obj) for obj in set(all_objects)}

st.write(f"총 프레임 수: {len(yolo_data)}")
st.write("탐지된 객체 빈도:")
st.json(object_counts)

# 🧠 4. 객체 등장 비교 결과
st.header("📦 Object Appearance Analysis")
with open(COMPARISON_JSON_PATH, "r", encoding="utf-8") as f:
    comparison = json.load(f)

st.write("Prompt 내 언급 객체:")
st.write(comparison.get("prompt_objects", []))

st.success(f"✅ 등장한 객체: {comparison.get('appeared_objects', [])}")
st.warning(f"❗ 누락된 객체: {comparison.get('missing_objects', [])}")

# 💬 5. GPT 피드백
st.header("🧠 GPT Feedback")
with open(GPT_FEEDBACK_PATH, "r", encoding="utf-8") as f:
    feedback = f.read().strip()
st.text_area("LLM Generated Feedback", value=feedback, height=200)

# 💬 6. 개선된 GPT 피드백 + 프롬프트
st.header("🛠 Revised Prompt (by GPT)")

improved_path = os.path.join(BASE_PATH, "data", "analysis_results", "video1_feedback_and_revised_prompt.txt")

if os.path.exists(improved_path):
    with open(improved_path, "r", encoding="utf-8") as f:
        improved = f.read().strip()
    st.text_area("Feedback + Improved Prompt", value=improved, height=300)
else:
    st.warning("Improved prompt file not found.")
