import os
import json
from ultralytics import YOLO
from tqdm import tqdm

def load_frames(frame_dir: str):
    frame_paths = sorted([
        os.path.join(frame_dir, fname)
        for fname in os.listdir(frame_dir)
        if fname.lower().endswith(".jpg")
    ])
    return frame_paths

def detect_objects(model_path: str, frame_paths: list, device: str = "cuda"):
    model = YOLO(model_path)
    detections = []

    for path in tqdm(frame_paths, desc="YOLO 객체 탐지 중"):
        result = model(path, device=device, verbose=False)[0]
        class_ids = result.boxes.cls.tolist()
        names = result.names
        detected_objects = list(set([names[int(cls)] for cls in class_ids]))
        detections.append({
            "frame": os.path.basename(path),
            "objects": detected_objects
        })

    return detections

def save_results(detections: list, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(detections, f, indent=2, ensure_ascii=False)
    print(f"[✅] 객체 탐지 결과 저장 완료 → {output_path}")

if __name__ == "__main__":
    # 프로젝트 루트 경로 계산
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 입력/출력 경로 설정
    frame_dir = os.path.join(project_root, "data", "frames", "video1")
    output_path = os.path.join(project_root, "data", "analysis_results", "video1_yolo.json")
    model_path = "yolov8m.pt"  # 또는 yolov8n.pt 등

    # 프레임 로딩 및 탐지 실행
    frame_paths = load_frames(frame_dir)
    detections = detect_objects(model_path, frame_paths, device="cuda")
    save_results(detections, output_path)
