import os
import json
import re
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")

def extract_keywords_from_prompt(prompt_text: str) -> list:
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(prompt_text)
    pos_tags = pos_tag(words)
    # NN, NNS, NNP, NNPS → 명사류
    nouns = [word.lower() for word, tag in pos_tags if tag.startswith("NN")]
    keywords = [word for word in nouns if word not in stop_words]
    return list(set(keywords))

def load_yolo_results(json_path: str) -> list:
    with open(json_path, "r", encoding="utf-8") as f:
        detections = json.load(f)

    all_objects = []
    for item in detections:
        all_objects.extend(item["objects"])

    return list(set(all_objects))

def compare_objects(prompt_objects: list, detected_objects: list) -> dict:
    prompt_set = set(prompt_objects)
    detected_set = set(obj.lower() for obj in detected_objects)

    appeared = sorted(prompt_set & detected_set)
    missing = sorted(prompt_set - detected_set)

    return {
        "prompt_objects": sorted(prompt_objects),
        "detected_objects": sorted(detected_objects),
        "appeared_objects": appeared,
        "missing_objects": missing
    }

def save_results(result: dict, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[✅] 객체 비교 결과 저장 완료 → {output_path}")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompt_path = os.path.join(project_root, "data", "prompts", "prompt1.txt")
    yolo_path = os.path.join(project_root, "data", "analysis_results", "video1_yolo.json")
    output_path = os.path.join(project_root, "data", "analysis_results", "video1_object_comparison.json")

    # 프롬프트 키워드 추출
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_text = f.read()
    prompt_keywords = extract_keywords_from_prompt(prompt_text)

    # YOLO 결과 불러오기
    detected_objects = load_yolo_results(yolo_path)

    # 비교 수행
    result = compare_objects(prompt_keywords, detected_objects)

    # 저장
    save_results(result, output_path)
