# openai_sora_client.py
import os, time, requests
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# 1) .env 로드
load_dotenv()

# 2) 환경변수
AZURE_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")   # e.g. https://<resource>.openai.azure.com  또는 https://<resource>.<region>.cognitiveservices.azure.com
AZURE_KEY        = os.getenv("AZURE_OPENAI_API_KEY")
API_VER          = os.getenv("AZURE_OPENAI_API_VERSION", "preview")
VIDEO_DEPLOYMENT = os.getenv("AZURE_OPENAI_VIDEO_DEPLOYMENT", "").strip()  # 배포 이름(Deployment name), 예: sora-deploy

def _assert_env():
    if not AZURE_ENDPOINT:
        raise ValueError("AZURE_OPENAI_ENDPOINT 가 비어 있습니다.")
    # Azure AI Foundry/Managed 엔드포인트는 둘 중 하나 패턴을 사용
    if not (".openai.azure.com" in AZURE_ENDPOINT or ".cognitiveservices.azure.com" in AZURE_ENDPOINT):
        raise ValueError("AZURE_OPENAI_ENDPOINT 형식이 올바르지 않습니다. 예) https://<resource>.openai.azure.com")
    if not AZURE_KEY:
        raise ValueError("AZURE_OPENAI_API_KEY 가 비어 있습니다.")
    if not VIDEO_DEPLOYMENT:
        raise ValueError("AZURE_OPENAI_VIDEO_DEPLOYMENT (배포 이름)가 비어 있습니다. 포털의 모델 배포 화면에서 이름을 확인하세요.")

def _json_post(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: int = 60) -> requests.Response:
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if r.status_code // 100 != 2:
        raise RuntimeError(f"POST {url} -> {r.status_code} {r.text}")
    return r

def _json_get(url: str, headers: Dict[str, str], timeout: int = 60) -> Dict[str, Any]:
    r = requests.get(url, headers=headers, timeout=timeout)
    if r.status_code // 100 != 2:
        raise RuntimeError(f"GET {url} -> {r.status_code} {r.text}")
    return r.json()

def generate_with_sora(
    prompt: str,
    *,
    size: str = "1280x720",
    duration: int = 4,
    fps: Optional[int] = None,     # 프리뷰에서 거부될 수 있으므로 옵션
    prefix: str = "video",
    poll_interval: float = 5.0,
    max_wait_sec: int = 15 * 60
):
    """
    prompt: 텍스트 프롬프트
    size:   "가로x세로" (예: "1280x720")
    duration: 초 단위 길이
    VIDEO_DEPLOYMENT: Azure에 배포한 비디오 생성 모델의 '배포 이름' (.env에서 읽음)
    반환: (prompt_filename, video_filename, time_tag)
    """
    _assert_env()

    # 파일/폴더 준비
    time_tag     = time.strftime("%Y%m%d_%H%M%S")
    prompt_fname = f"{prefix}_{time_tag}.txt"
    video_fname  = f"{prefix}_{time_tag}_00001.mp4"

    os.makedirs("data/prompts", exist_ok=True)
    os.makedirs("ComfyUI/output", exist_ok=True)

    with open(os.path.join("data/prompts", prompt_fname), "w", encoding="utf-8") as f:
        f.write(prompt.strip() + "\n")

    # 해상도 파싱
    try:
        w, h = map(int, size.lower().split("x"))
    except Exception:
        raise ValueError(f'size 형식 오류: "{size}"  (예: "1280x720")')

    headers = {"api-key": AZURE_KEY, "Content-Type": "application/json"}

    # 1) Job 생성
    create_url = f"{AZURE_ENDPOINT}/openai/v1/video/generations/jobs?api-version={API_VER}"
    body = {
        "model": VIDEO_DEPLOYMENT,   # ★ 배포 이름
        "prompt": prompt,
        "width":  w,
        "height": h,
        "n_seconds": int(duration),
        # 프리뷰 스펙에 따라 거부될 수 있어 조건부로만 포함
    }
    if fps is not None:
        body["fps"] = int(fps)

    resp = _json_post(create_url, headers, body)
    job_id = resp.json()["id"]

    # 2) 상태 폴링
    status_url = f"{AZURE_ENDPOINT}/openai/v1/video/generations/jobs/{job_id}?api-version={API_VER}"
    t0 = time.time()
    status_payload, status = {}, None
    while status not in ("succeeded", "failed", "cancelled"):
        if time.time() - t0 > max_wait_sec:
            raise TimeoutError(f"영상 생성 대기 초과({max_wait_sec}s). job_id={job_id}, 마지막 상태={status}")
        time.sleep(poll_interval)
        status_payload = _json_get(status_url, headers)
        status = status_payload.get("status")

    if status != "succeeded":
        raise RuntimeError(f"Video generation failed. status={status}, detail={status_payload}")

    # 3) 결과 다운로드
    generations = status_payload.get("generations", [])
    if not generations:
        raise RuntimeError("Job 결과에 generations가 없습니다.")
    gen_id = generations[0]["id"]

    video_url = f"{AZURE_ENDPOINT}/openai/v1/video/generations/{gen_id}/content/video?api-version={API_VER}"
    video_resp = requests.get(video_url, headers=headers, timeout=300)
    if video_resp.status_code // 100 != 2:
        raise RuntimeError(f"GET {video_url} -> {video_resp.status_code} {video_resp.text}")

    out_path = os.path.join("ComfyUI", "output", video_fname)
    with open(out_path, "wb") as f:
        f.write(video_resp.content)

    return prompt_fname, video_fname, time_tag
