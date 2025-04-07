import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

def load_prompt(prompt_path: str) -> str:
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def load_frames(frame_dir: str):
    frame_paths = sorted([
        os.path.join(frame_dir, fname)
        for fname in os.listdir(frame_dir)
        if fname.endswith(".jpg")
    ])
    return frame_paths

def compute_clip_similarity(prompt: str, frame_paths: list, device: str = "cpu"):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    scores = []
    for path in tqdm(frame_paths, desc="CLIP Similarity"):
        image = Image.open(path).convert("RGB")
        inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs)
        
        # softmax 제거: logits 직접 사용
        similarity = outputs.logits_per_image[0][0].item()

        scores.append({
            "frame": os.path.basename(path),
            "score": round(similarity, 4)
        })
    return scores

def save_results(scores, out_json_path, out_plot_path):
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

    x = list(range(len(scores)))
    y = [s["score"] for s in scores]
    plt.figure(figsize=(10, 4))
    plt.plot(x, y, marker="o")
    plt.title("CLIP Similarity per Frame")
    plt.xlabel("Frame Index")
    plt.ylabel("Similarity Score")
    plt.grid(True)
    plt.savefig(out_plot_path)
    plt.close()

if __name__ == "__main__":
    prompt_path = os.path.join("data", "prompts", "prompt1.txt")
    frame_dir = os.path.join("data", "frames", "video1")
    out_json = os.path.join("data", "analysis_results", "video1_clip.json")
    out_plot = os.path.join("data", "analysis_results", "video1_clip_plot.png")

    os.makedirs("data/analysis_results", exist_ok=True)

    prompt = load_prompt(prompt_path)
    frames = load_frames(frame_dir)
    results = compute_clip_similarity(prompt, frames, device="cpu")
    save_results(results, out_json, out_plot)

    print(f"[✅] CLIP 분석 완료! → {out_json}, {out_plot}")
