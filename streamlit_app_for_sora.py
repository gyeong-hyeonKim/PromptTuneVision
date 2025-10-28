# streamlit_app.py
import os, sys, subprocess, json, time, glob
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
# --- 파일 상단 import 아래 어딘가에 유틸 추가 ---
from collections import defaultdict

def summarize_yolo(y):
    """
    YOLO 결과가 dict 또는 list 어떤 형식이든 요약 정보를 반환.
    반환: {"frames": N, "labels_top": {...}}
    """
    def count_from_frames(frames):
        labels = defaultdict(int)
        for fr in frames:
            if not isinstance(fr, dict):
                continue
            dets = (fr.get("detections") or fr.get("objects") or
                    fr.get("preds") or fr.get("results") or [])
            if isinstance(dets, dict):  # 일부 포맷이 {"detections":[...]}가 아니라 dict일 수 있음
                dets = dets.get("detections", []) or dets.get("objects", []) or dets.get("results", [])
            if isinstance(dets, list):
                for d in dets:
                    if not isinstance(d, dict):
                        continue
                    label = (d.get("label") or d.get("name") or d.get("class") or
                             d.get("cls_name"))
                    # 정수 class index만 있을 때 대비
                    if label is None and isinstance(d.get("cls"), int):
                        label = str(d["cls"])
                    if label is not None:
                        labels[str(label)] += 1
        # 상위 10개만
        top = dict(sorted(labels.items(), key=lambda x: x[1], reverse=True)[:10])
        return top

    # dict 포맷
    if isinstance(y, dict):
        frames_list = (y.get("frames") or y.get("results") or y.get("preds") or [])
        if isinstance(frames_list, dict):
            # 드문 케이스: {"frames": {"0":[...], "1":[...]}}
            frames_list = sum((v if isinstance(v, list) else [] for v in frames_list.values()), [])
        frames_count = len(frames_list)
        labels_top = y.get("labels_count") or count_from_frames(frames_list)
        return {"frames": frames_count, "labels_top": labels_top}

    # list 포맷 (프레임 배열)
    if isinstance(y, list):
        return {"frames": len(y), "labels_top": count_from_frames(y)}

    # 알 수 없는 포맷
    return {"frames": 0, "labels_top": {}}


# 프로젝트 경로 고정
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PROMPT_DIR = os.path.join(BASE_DIR, "data", "prompts")
VIDEO_DIR  = os.path.join(BASE_DIR, "ComfyUI", "output")

sys.path.append(BASE_DIR)
from openai_sora_client import generate_with_sora  # Sora 생성 모듈

st.set_page_config(page_title="PromptTuneVision – Sora Edition", layout="wide")
st.title("🎬 PromptTuneVision – Sora + Streaming 분석 대시보드")

# 환경 정보 표시(문제 디버깅에 도움)
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
deployment = os.getenv("AZURE_OPENAI_VIDEO_DEPLOYMENT", "")
with st.expander("환경 설정 확인 (.env)", expanded=False):
    st.write(f"**Endpoint**: `{endpoint}`")
    st.write(f"**Video Deployment**: `{deployment}` (※ 함수 내부에서 자동 사용)")

with st.sidebar:
    st.header("Sora 옵션")
    size = st.selectbox("해상도", ["1280x720", "1920x1080"], index=0)
    fps  = st.selectbox("FPS", [None, 24, 25, 30, 60], index=3)  # None 선택 시 서버 기본값 사용
    duration = st.slider("영상 길이(초)", 1, 10, 4)
    st.caption("배포명(model)은 .env의 `AZURE_OPENAI_VIDEO_DEPLOYMENT` 값을 사용합니다.")

st.subheader("1) 프롬프트 입력")
user_prompt = st.text_area(
    "생성 프롬프트",
    height=160,
    placeholder="장면 설명, 등장 객체, 카메라 무브, 톤/라이팅 등..."
)

colA, colB = st.columns([1,1])
with colA:
    go = st.button("🎥 Sora로 영상 생성 & 분석 시작", type="primary")
with colB:
    st.caption("생성 후 자동으로 프롬프트/영상 저장 → 분석 파이프라인 실행 → 아래에 결과가 표시됩니다.")

# 실행 영역
if go:
    if not user_prompt.strip():
        st.error("프롬프트를 입력하세요.")
        st.stop()

    with st.status("Sora로 영상 생성 중...", expanded=True) as status:
        st.write("프롬프트 저장 → Sora 호출 → MP4 저장")
        try:
            # ⚠️ model 인자 제거. fps는 Optional이므로 None일 때는 넘기지 않음.
            kwargs = dict(size=size, duration=int(duration))
            if fps is not None:
                kwargs["fps"] = int(fps)

            prompt_file, video_file, time_tag = generate_with_sora(user_prompt, **kwargs)
        except Exception as e:
            st.error(f"Sora 생성 중 오류: {e}")
            st.stop()

        st.write(f"✅ 저장된 프롬프트: `{prompt_file}`")
        st.write(f"✅ 저장된 영상: `{video_file}`")

        # 분석 파이프라인 호출
        st.write("🔎 분석 파이프라인 실행… (CLIP/YOLO/Object 비교/GPT 피드백)")
        try:
            subprocess.run(
                [
                    sys.executable, os.path.join(BASE_DIR, "run_pipeline.py"),
                    "--prompt", os.path.join(PROMPT_DIR, prompt_file),
                    "--video",  os.path.join(VIDEO_DIR,  video_file)
                ],
                check=True
            )
        except subprocess.CalledProcessError as e:
            st.error(f"분석 실행 실패: {e}")
            st.stop()

        status.update(label="✅ 생성 및 분석 완료", state="complete", expanded=False)

    # === 결과 표시 섹션 ===
    st.subheader("2) 결과 시각화")

    # 원본 비디오
    video_path = os.path.join(VIDEO_DIR, video_file)
    st.markdown("**생성된 영상**")
    st.video(video_path)

    # 분석 산출물 경로 규칙: data/{time_tag}/analysis_results/...
    results_dir = os.path.join(BASE_DIR, "data", time_tag, "analysis_results")
    st.write(f"분석 결과 폴더: `{results_dir}`")

    # 2-1) YOLO 결과
    yolo_json = os.path.join(results_dir, f"{os.path.splitext(video_file)[0]}_yolo.json")
    if os.path.exists(yolo_json):
        st.markdown("**YOLO 탐지 결과 (요약)**")
        try:
            with open(yolo_json, "r", encoding="utf-8") as f:
                yolo_data = json.load(f)
            summary = summarize_yolo(yolo_data)
            st.json(summary)
        except Exception as e:
            st.warning(f"YOLO 결과 읽기 오류: {type(e).__name__}: {e}")
    else:
        st.info("YOLO 결과 파일이 없습니다.")

    # 2-2) 객체 비교
    obj_json = os.path.join(results_dir, f"{os.path.splitext(video_file)[0]}_object_comparison.json")
    if os.path.exists(obj_json):
        st.markdown("**프롬프트 vs 객체 비교 결과 (요약)**")
        try:
            with open(obj_json, "r", encoding="utf-8") as f:
                obj_data = json.load(f)
            st.json(obj_data.get("summary", obj_data))
        except Exception as e:
            st.warning(f"객체 비교 결과 읽기 오류: {e}")

    # 2-3) GPT 피드백 & 개선 프롬프트
    fb_path = os.path.join(results_dir, f"{os.path.splitext(video_file)[0]}_feedback_gpt.txt")
    ip_path_candidates = sorted(glob.glob(os.path.join(results_dir, "*improved_prompt*.txt"))) \
                         + sorted(glob.glob(os.path.join(results_dir, "*revised_prompt*.txt")))

    c1, c2 = st.columns([1,1])
    with c1:
        st.markdown("**GPT 피드백**")
        if os.path.exists(fb_path):
            st.code(open(fb_path, "r", encoding="utf-8").read())
        else:
            st.info("피드백 파일이 없습니다.")

    with c2:
        st.markdown("**개선 프롬프트**")
        if ip_path_candidates:
            ip_file = ip_path_candidates[-1]
            improved = open(ip_file, "r", encoding="utf-8").read().strip()
            st.code(improved)
            if st.button("⬅️ 개선 프롬프트를 입력창으로 가져오기"):
                st.session_state["prefill"] = improved
                st.rerun()
        else:
            st.info("개선 프롬프트 파일이 없습니다.")

# 재실행 시 개선 프롬프트 채워넣기
if "prefill" in st.session_state and not st.session_state.get("used_prefill"):
    st.session_state["used_prefill"] = True
    st.experimental_set_query_params(prefill="1")
    st.success("개선 프롬프트를 입력창으로 불러왔습니다. 내용 확인 후 다시 생성하세요.")
