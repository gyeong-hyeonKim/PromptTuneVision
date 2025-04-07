import cv2
import os

def extract_frames(video_filename: str, frame_interval: int = 10):
    """
    data/raw_videos/{video_filename} 경로에서 영상을 불러와
    일정 프레임 간격마다 프레임 이미지를 저장합니다.

    저장 경로: data/frames/{video_name}/frame_XXXX.jpg

    Args:
        video_filename (str): 예) "video1.mp4"
        frame_interval (int): 몇 프레임마다 1장씩 추출할지 (기본: 10)
    """
    video_path = os.path.join("data", "raw_videos", video_filename)
    video_name = os.path.splitext(video_filename)[0]
    output_dir = os.path.join("data", "frames", video_name)

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 영상을 열 수 없습니다: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] 영상 정보 — 총 프레임 수: {total_frames}, FPS: {fps:.2f}")

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            filename = f"frame_{saved_idx:04d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            saved_idx += 1

        frame_idx += 1

    cap.release()
    print(f"[INFO] 총 {saved_idx}개의 프레임이 저장되었습니다 → {output_dir}")

if __name__ == "__main__":
    extract_frames("video1.mp4", frame_interval=10)
