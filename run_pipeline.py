import os
import sys
import re
import datetime # 시간 로깅을 위해 추가
import subprocess # Streamlit 실행을 위해 추가
import argparse # __main__ 블록에서 인자 파싱을 위해 argparse를 여기에도 import

# --- 디버깅 로그 함수 ---
def log_message(message):
    print(f"[{datetime.datetime.now()}] RUN_PIPELINE_DEBUG: {message}", flush=True)

log_message("스크립트 시작점")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
log_message(f"BASE_DIR: {BASE_DIR}")
SCRIPT_PATH = os.path.join(BASE_DIR, "scripts")
log_message(f"SCRIPT_PATH: {SCRIPT_PATH}")
FRAME_PATH = os.path.join(SCRIPT_PATH, "Frame")
log_message(f"FRAME_PATH: {FRAME_PATH}")

sys.path.append(SCRIPT_PATH)
log_message(f"sys.path에 SCRIPT_PATH 추가: {SCRIPT_PATH}")
sys.path.append(FRAME_PATH)
log_message(f"sys.path에 FRAME_PATH 추가: {FRAME_PATH}")

log_message("Frame_Extraction import 시도")
from Frame_Extraction import extract_frames
log_message("Frame_Extraction import 완료")

log_message("CLIP_Similarity import 시도")
from CLIP_Similarity import load_prompt, load_frames as load_clip_frames, compute_clip_similarity, save_results as save_clip_results
log_message("CLIP_Similarity import 완료")

log_message("YOLO_Detection import 시도")
from YOLO_Detection import load_frames as load_yolo_frames, detect_objects, save_results as save_yolo_results
log_message("YOLO_Detection import 완료")

log_message("Object_Comparison import 시도")
from Object_Comparison import extract_keywords_from_prompt, load_yolo_results, compare_objects, save_results as save_comparison_results
log_message("Object_Comparison import 완료")

log_message("Generate_Feedback_API import 시도")
from Generate_Feedback_API import load_context as load_context_feedback, generate_prompt, call_gpt as call_feedback_gpt, save_feedback
log_message("Generate_Feedback_API import 완료")

log_message("Generate_Improved_prompt_API import 시도")
from Generate_Improved_prompt_API import load_context as load_context_prompt, create_prompt, call_gpt as call_improved_gpt, save_output
log_message("Generate_Improved_prompt_API import 완료")

log_message("모든 모듈 import 완료")


def run_pipeline(prompt_path, video_path, model_path="yolov8m.pt"):
    log_message(f"run_pipeline 함수 시작. prompt_path='{prompt_path}', video_path='{video_path}'")

    prompt_filename = os.path.basename(prompt_path)
    log_message(f"prompt_filename: '{prompt_filename}'")

    match = re.search(r"(\d{8}_\d{6})", prompt_filename)
    if not match:
        time_based_foldername = os.path.splitext(prompt_filename)[0]
        log_message(f"[경고] 시간 정보 못 찾음. time_based_foldername 설정: '{time_based_foldername}'")
    else:
        time_based_foldername = match.group(1)
        log_message(f"시간 정보 찾음. time_based_foldername 설정: '{time_based_foldername}'")

    video_name_for_files = os.path.splitext(os.path.basename(video_path))[0] # 결과 파일명에 사용될 비디오 이름
    log_message(f"video_name_for_files: '{video_name_for_files}'")
    project_base_path = os.path.abspath(os.path.dirname(__file__)) # run_pipeline.py 위치
    log_message(f"project_base_path: '{project_base_path}'")

    main_output_dir = os.path.join(project_base_path, "data", time_based_foldername)
    log_message(f"main_output_dir 경로 정의: '{main_output_dir}'")
    os.makedirs(main_output_dir, exist_ok=True)
    log_message(f"main_output_dir 생성 완료 (또는 이미 존재).")

    frame_output_dir = os.path.join(main_output_dir, "frames", video_name_for_files)
    log_message(f"frame_output_dir 경로 정의: '{frame_output_dir}'")

    analysis_result_dir = os.path.join(main_output_dir, "analysis_results")
    log_message(f"analysis_result_dir 경로 정의: '{analysis_result_dir}'")
    os.makedirs(analysis_result_dir, exist_ok=True)
    log_message(f"analysis_result_dir 생성 완료 (또는 이미 존재).")

    print(f"\n[INFO] 모든 결과는 '{main_output_dir}' 폴더 하위에 저장됩니다.")
    log_message("기본 INFO 메시지 출력 완료")

    print("\n[1️⃣] 프레임 추출 시작")
    log_message(f"extract_frames 호출 직전. video_path='{video_path}', frame_output_dir='{frame_output_dir}'")
    extract_frames(video_path, frame_output_dir, frame_interval=10)
    log_message("extract_frames 호출 완료")

    print("\n[2️⃣] CLIP 유사도 분석 시작")
    # (이하 CLIP, YOLO, 객체 비교, GPT 피드백, GPT 개선 프롬프트 생성 로직은 이전과 동일)
    # ... (모든 분석 및 파일 저장 로직) ...
    prompt = load_prompt(prompt_path)
    clip_frame_paths = load_clip_frames(frame_output_dir)
    clip_scores = compute_clip_similarity(prompt, clip_frame_paths, device="cuda")
    clip_json_path = os.path.join(analysis_result_dir, f"{video_name_for_files}_clip.json")
    clip_plot_path = os.path.join(analysis_result_dir, f"{video_name_for_files}_clip_plot.png")
    save_clip_results(clip_scores, clip_json_path, clip_plot_path)
    log_message("CLIP 분석 및 저장 완료")

    print("\n[3️⃣] YOLO 객체 탐지 시작")
    yolo_frame_paths = load_yolo_frames(frame_output_dir)
    yolo_results = detect_objects(model_path, yolo_frame_paths, device="cuda")
    yolo_json_path = os.path.join(analysis_result_dir, f"{video_name_for_files}_yolo.json")
    save_yolo_results(yolo_results, yolo_json_path)
    log_message("YOLO 탐지 및 저장 완료")

    print("\n[4️⃣] 객체 비교 수행")
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_text = f.read()
    prompt_objects = extract_keywords_from_prompt(prompt_text)
    detected_objects = load_yolo_results(yolo_json_path)
    comparison_result = compare_objects(prompt_objects, detected_objects)
    comparison_path = os.path.join(analysis_result_dir, f"{video_name_for_files}_object_comparison.json")
    save_comparison_results(comparison_result, comparison_path)
    log_message("객체 비교 및 저장 완료")

    print("\n[5️⃣] GPT 피드백 생성")
    context = load_context_feedback(comparison_path)
    feedback_prompt_text = generate_prompt(context) # generate_prompt 함수는 feedback_prompt로 이름을 변경했었는지 확인 필요
    feedback = call_feedback_gpt(feedback_prompt_text)
    feedback_path = os.path.join(analysis_result_dir, f"{video_name_for_files}_feedback_gpt.txt")
    save_feedback(feedback, feedback_path)
    log_message("GPT 피드백 생성 및 저장 완료")

    print("\n[6️⃣] GPT 개선 프롬프트 생성")
    prompt_context = load_context_prompt(prompt_path, comparison_path)
    improved_prompt_text_for_gpt = create_prompt(prompt_context,clip_scores) # create_prompt 함수는 improved_prompt_text로 이름을 변경했었는지 확인 필요
    improved_prompt_output = call_improved_gpt(improved_prompt_text_for_gpt)
    improved_prompt_path = os.path.join(analysis_result_dir, f"{video_name_for_files}_feedback_and_revised_prompt.txt")
    save_output(improved_prompt_output, improved_prompt_path)
    log_message("GPT 개선 프롬프트 생성 및 저장 완료")
    
    print("\n[✅] 전체 분석 파이프라인 완료!")
    log_message("run_pipeline 함수 내 분석 로직 정상 종료")

    #  streamlit_app.py의 절대 경로 (run_pipeline.py와 같은 디렉토리에 있다고 가정)
    streamlit_app_script_path = os.path.join(project_base_path, "streamlit_app.py")

    # Streamlit 앱 실행에 필요한 인자들을 절대 경로로 변환
    abs_analysis_result_dir = os.path.abspath(analysis_result_dir)
    abs_prompt_path = os.path.abspath(prompt_path)
    abs_video_path = os.path.abspath(video_path)

    # Streamlit 실행 명령어 구성
    # sys.executable은 현재 파이썬 인터프리터를 사용하도록 합니다.
    streamlit_command = [
        sys.executable, "-m", "streamlit", "run", streamlit_app_script_path,
        "--", # Streamlit 자체 인자와 스크립트 인자 구분
        "--results_dir", abs_analysis_result_dir,
        "--prompt_file_path", abs_prompt_path,
        "--video_file_path", abs_video_path,
        "--video_name", video_name_for_files # 프롬프트 파일명이 아닌, 비디오 파일에서 추출한 이름 사용
    ]

    print(f"\n[🚀] Streamlit 대시보드 실행: {' '.join(streamlit_command)}")
    log_message(f"Streamlit 앱 실행 시도: {' '.join(streamlit_command)}")
    
    # Popen을 사용하여 Streamlit을 백그라운드에서 실행 (run_pipeline.py 종료 후에도 유지)
    try:
        subprocess.Popen(streamlit_command)
        log_message("Streamlit 앱 실행 요청 완료 (백그라운드 실행).")
        print("[INFO] Streamlit 대시보드가 새 창이나 탭에서 열릴 것입니다.")
    except FileNotFoundError:
        log_message("[ERROR] Streamlit을 실행할 수 없습니다. 설치되어 있는지 확인해주세요.")
        print("[ERROR] Streamlit을 실행할 수 없습니다. `pip install streamlit`으로 설치해주세요.")
    except Exception as e:
        log_message(f"[ERROR] Streamlit 실행 중 오류 발생: {e}")
        print(f"[ERROR] Streamlit 실행 중 오류 발생: {e}")


if __name__ == "__main__":
    log_message("스크립트의 __main__ 진입점 실행됨")
    # argparse는 이미 위에 import 되어 있음

    parser = argparse.ArgumentParser(description="Run the full analysis pipeline for video and prompt, then launch Streamlit dashboard.")
    parser.add_argument("--prompt", required=True, help="Path to the prompt text file.")
    parser.add_argument("--video", required=True, help="Path to the video file.")
    parser.add_argument("--model", default="yolov8m.pt", help="Path to the YOLO model (e.g., yolov8m.pt).")
    
    args = parser.parse_args()
    log_message(f"커맨드 라인 인자 파싱 완료. Args: prompt='{args.prompt}', video='{args.video}', model='{args.model}'")

    try:
        run_pipeline(args.prompt, args.video, model_path=args.model)
        log_message("run_pipeline 함수 실행 완료 (메인 블록)")
    except Exception as e:
        log_message(f"run_pipeline 함수 실행 중 예외 발생 (메인 블록): {e}")
        print(f"[CRITICAL ERROR] 파이프라인 실행 중 오류 발생: {e}") # 사용자에게 보이는 중요 오류 메시지
        # 개발 중에는 예외를 다시 발생시켜 스택 트레이스를 보는 것이 유용할 수 있습니다.
        # raise