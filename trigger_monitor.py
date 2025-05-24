import os
import time
import subprocess
from datetime import datetime
import sys
python_executable = sys.executable  # 현재 파이썬 실행 파일의 절대 경로

PROMPT_DIR = "data/prompts"
VIDEO_DIR = "ComfyUI/output"
POLL_INTERVAL = 2  # 초

def get_latest_files_after(start_time):
    prompt_files = {
        f: os.path.getmtime(os.path.join(PROMPT_DIR, f))
        for f in os.listdir(PROMPT_DIR)
        if f.endswith(".txt")
    }

    video_files = {
        f: os.path.getmtime(os.path.join(VIDEO_DIR, f))
        for f in os.listdir(VIDEO_DIR)
        if f.endswith(".mp4")
    }

    matched = []
    for p_file, p_time in prompt_files.items():
        if p_time < start_time:
            continue
        base_name = os.path.splitext(p_file)[0]
        candidates = [v for v in video_files if base_name in v]
        if candidates:
            latest_video = max(candidates, key=lambda v: video_files[v])
            matched.append((p_file, latest_video))

    return matched

def run_analysis_pipeline(prompt_file, video_file):
    print(f"[▶️] 분석 시작: prompt={prompt_file}, video={video_file}")
    subprocess.run([
        python_executable,
        "run_pipeline.py",
        "--prompt", os.path.join(PROMPT_DIR, prompt_file),
        "--video", os.path.join(VIDEO_DIR, video_file)
    ], check=True)

def main():
    print("[👀] 프롬프트 및 비디오 감시 시작...")
    start_time = time.time()
    processed = set()

    while True:
        try:
            pairs = get_latest_files_after(start_time)
            for prompt_file, video_file in pairs:
                if (prompt_file, video_file) in processed:
                    continue
                run_analysis_pipeline(prompt_file, video_file)
                processed.add((prompt_file, video_file))

            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            print("\n[🛑] 감시 종료")
            break
        except subprocess.CalledProcessError as e:
            print(f"[❌] 분석 중 오류 발생: {e}")
        except Exception as e:
            print(f"[⚠️] 예기치 못한 오류: {e}")

if __name__ == "__main__":
    main()
