import os
import json
import streamlit as st
import argparse # 인자 파싱을 위해 추가
import sys # sys.exit() 사용을 위해 추가

# 🌟 타이틀
st.set_page_config(page_title="Prompt-TuneVision Dashboard", layout="wide")
st.title("🎬 Prompt-TuneVision : Prompt Evaluation Dashboard")

def display_dashboard(results_dir, prompt_file_path, video_file_path, video_name_arg):
    # 📌 0. 원본 프롬프트 및 비디오 출력
    st.header("📽️ Original Prompt & Generated Video")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Prompt")
        if os.path.exists(prompt_file_path):
            with open(prompt_file_path, "r", encoding="utf-8") as f:
                prompt_text = f.read().strip()
            st.code(prompt_text, language="text")
        else:
            st.error(f"Prompt file not found: {prompt_file_path}")

    with col2:
        st.subheader("🎞️ Generated Video")
        if os.path.exists(video_file_path):
            try:
                st.video(video_file_path)
            except Exception as e:
                st.error(f"Error loading video: {e}. Ensure the video format is supported by Streamlit (e.g., MP4, WebM, Ogg).")
                st.info(f"Video path: {video_file_path}")
        else:
            st.error(f"Video file not found: {video_file_path}")

    # 📊 1. CLIP 유사도 플롯
    st.header("📈 CLIP Similarity per Frame")
    clip_plot_path = os.path.join(results_dir, f"{video_name_arg}_clip_plot.png")
    if os.path.exists(clip_plot_path):
        st.image(clip_plot_path, caption="CLIP 유사도 분석 결과", use_container_width=True) # use_container_width=True로 변경
    else:
        st.warning(f"CLIP plot not found: {clip_plot_path}")

    # 🧠 2. YOLO 탐지 객체 요약
    st.header("🔍 YOLO Detected Objects")
    yolo_json_path = os.path.join(results_dir, f"{video_name_arg}_yolo.json")
    if os.path.exists(yolo_json_path):
        with open(yolo_json_path, "r", encoding="utf-8") as f:
            yolo_data = json.load(f)

        all_objects = []
        for item in yolo_data:
            all_objects.extend(item.get("objects", [])) # "objects" 키가 없을 경우 대비
        
        if yolo_data: # yolo_data가 비어있지 않은 경우에만 계산
            object_counts = {obj: all_objects.count(obj) for obj in set(all_objects)}
            st.write(f"총 프레임 수 (YOLO 분석 대상): {len(yolo_data)}")
            st.write("탐지된 객체 빈도:")
            st.json(object_counts)
        else:
            st.info("No YOLO detection data found or data is empty.")
            
    else:
        st.warning(f"YOLO JSON data not found: {yolo_json_path}")

    # 📦 3. 객체 등장 비교 결과
    st.header("📊 Object Appearance Analysis")
    comparison_json_path = os.path.join(results_dir, f"{video_name_arg}_object_comparison.json")
    if os.path.exists(comparison_json_path):
        with open(comparison_json_path, "r", encoding="utf-8") as f:
            comparison = json.load(f)

        st.write("Prompt 내 언급 객체:")
        st.write(comparison.get("prompt_objects", []))

        st.success(f"✅ 등장한 객체: {comparison.get('appeared_objects', [])}")
        st.warning(f"❗ 누락된 객체: {comparison.get('missing_objects', [])}")
    else:
        st.warning(f"Object comparison JSON data not found: {comparison_json_path}")

    # 💬 4. 개선된 GPT 피드백 + 프롬프트
    st.header("🛠️ Revised Prompt & Feedback (by GPT)")
    # 파일명은 run_pipeline.py에서 저장하는 _feedback_and_revised_prompt.txt 사용
    revised_prompt_feedback_path = os.path.join(results_dir, f"{video_name_arg}_feedback_and_revised_prompt.txt")
    if os.path.exists(revised_prompt_feedback_path):
        with open(revised_prompt_feedback_path, "r", encoding="utf-8") as f:
            improved_text = f.read().strip()
        st.text_area("Feedback + Improved Prompt", value=improved_text, height=300)
    else:
        st.warning(f"Revised prompt and feedback file not found: {revised_prompt_feedback_path}")

if __name__ == '__main__':
    # Streamlit 앱이 `streamlit run streamlit_app.py -- --arg1 val1` 형태로 실행될 때 인자 파싱
    # 'streamlit run' 뒤의 '--'는 스크립트에 인자를 전달하기 위한 구분자입니다.
    parser = argparse.ArgumentParser(description="Prompt-TuneVision Dashboard")
    parser.add_argument("--results_dir", required=True, help="Path to the analysis results directory.")
    parser.add_argument("--prompt_file_path", required=True, help="Path to the original prompt text file.")
    parser.add_argument("--video_file_path", required=True, help="Path to the generated video file.")
    parser.add_argument("--video_name", required=True, help="Base name of the video (used for finding result files).")
    
    # Streamlit은 자체적으로 인자를 파싱하므로, 스크립트 인자만 골라내기
    # sys.argv[0]은 스크립트 이름, 그 이후부터가 인자
    # `streamlit run app.py -- --foo bar` 와 같이 전달하면 sys.argv는 ['app.py', '--foo', 'bar']가 됨
    # 하지만 `streamlit run app.py --foo bar`와 같이 전달하면 streamlit 자체 인자로 해석될 수 있음
    # `streamlit run app.py -- --foo bar`와 같이 `--` 뒤에 스크립트 인자를 전달하는 것이 안전.
    
    # 현재 스크립트의 인자만 파싱하기 위해 sys.argv를 조정할 필요는 없음.
    # argparse는 `sys.argv[1:]`를 기본으로 사용함.
    # Streamlit 실행 시 `streamlit run your_script.py -- --your_arg value` 와 같이 `--` 뒤에 인자를 전달해야 함.
    try:
        # streamlit run 실행 시 sys.argv에 streamlit 자체 인자가 포함될 수 있으므로,
        # 스크립트에 전달된 인자만 파싱하도록 함.
        # 일반적으로 `streamlit run app.py arg1 arg2`와 같이 하면 streamlit이 arg1, arg2를 먹어버림.
        # `streamlit run app.py -- --arg1 val1 --arg2 val2`와 같이 사용해야 함.
        # 이 경우 sys.argv는 ['app.py', '--arg1', 'val1', '--arg2', 'val2']가 됩니다.
        args = parser.parse_args()
        display_dashboard(args.results_dir, args.prompt_file_path, args.video_file_path, args.video_name)
    except SystemExit as e:
        # argparse가 --help 등으로 종료할 때 SystemExit 예외 발생, 정상 종료로 처리
        if e.code != 0: # 코드가 0이 아니면 실제 오류이므로 다시 발생
             st.error(f"Argument parsing error: Check command line arguments. {e}")
             raise
        sys.exit(e.code) # 정상 종료 (예: --help)
    except Exception as e:
        st.error(f"An error occurred: {e}")
        # 에러 발생 시에도 앱은 계속 실행될 수 있도록 하거나, 혹은 여기서 앱을 중단시킬 수 있음.