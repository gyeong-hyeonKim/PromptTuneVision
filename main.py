
import os
import subprocess
import time
from run_pipeline import run_pipeline

def trigger_and_run():
    # 생성된 최신 프롬프트와 영상 탐색
    prompt_dir = os.path.join("data", "prompts")
    video_dir = os.path.join("data", "raw_videos")

    prompts = sorted([f for f in os.listdir(prompt_dir) if f.endswith(".txt")], key=lambda x: os.path.getmtime(os.path.join(prompt_dir, x)), reverse=True)
    videos = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")], key=lambda x: os.path.getmtime(os.path.join(video_dir, x)), reverse=True)

    if not prompts or not videos:
        print("❗ 프롬프트 또는 비디오 파일이 존재하지 않습니다.")
        return

    latest_prompt = os.path.join(prompt_dir, prompts[0])
    latest_video = os.path.join(video_dir, videos[0])

    print(f"▶ 최신 프롬프트: {latest_prompt}")
    print(f"▶ 최신 비디오: {latest_video}")

    # 분석 파이프라인 실행
    run_pipeline(latest_prompt, latest_video)

    # Streamlit 앱 실행
    print("\n🌐 Streamlit 앱 실행 중...")
    subprocess.run(["streamlit", "run", "streamlit_app.py"])

if __name__ == "__main__":
    trigger_and_run()
