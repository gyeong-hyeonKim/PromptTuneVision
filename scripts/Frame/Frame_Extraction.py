import cv2
import os
import datetime # 시간 로깅을 위해 추가

# --- 디버깅 로그 함수 ---
def log_message_fe(message):
    print(f"[{datetime.datetime.now()}] FRAME_EXTRACTION_DEBUG: {message}", flush=True)

def extract_frames(full_video_path: str, output_dir_for_frames: str, frame_interval: int = 10):
    log_message_fe(f"extract_frames 함수 시작. full_video_path='{full_video_path}', output_dir_for_frames='{output_dir_for_frames}'")

    log_message_fe(f"os.makedirs 시도: '{output_dir_for_frames}'")
    os.makedirs(output_dir_for_frames, exist_ok=True)
    log_message_fe(f"os.makedirs 완료 (또는 이미 존재).")

    log_message_fe(f"cv2.VideoCapture 시도: '{full_video_path}'")
    cap = cv2.VideoCapture(full_video_path)
    log_message_fe("cv2.VideoCapture 호출 완료.") # 성공 여부는 isOpened로 확인

    if not cap.isOpened():
        log_message_fe(f"[ERROR] 영상을 열 수 없습니다 (cap.isOpened() False): '{full_video_path}'")
        print(f"[ERROR] 영상을 열 수 없습니다: {full_video_path}") # 기존 사용자 출력 유지
        return
    log_message_fe(f"영상이 성공적으로 열렸습니다 (cap.isOpened() True): '{full_video_path}'")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log_message_fe(f"총 프레임 수: {total_frames}")

    frame_idx = 0
    saved_idx = 0

    log_message_fe("프레임 추출 루프 시작")
    while True:
        ret, frame = cap.read()
        if not ret:
            log_message_fe("cap.read() 실패 또는 비디오 종료.")
            break

        if frame_idx % frame_interval == 0:
            filename = f"frame_{saved_idx:04d}.jpg"
            filepath = os.path.join(output_dir_for_frames, filename)
            # log_message_fe(f"프레임 저장 시도: '{filepath}'") # 너무 많은 로그를 생성할 수 있어 주석 처리
            cv2.imwrite(filepath, frame)
            saved_idx += 1

        frame_idx += 1
    log_message_fe("프레임 추출 루프 종료")

    cap.release()
    log_message_fe("cap.release() 완료")
    print(f"[INFO] 총 {saved_idx}개의 프레임이 '{output_dir_for_frames}'에 저장되었습니다.")
    log_message_fe("extract_frames 함수 정상 종료")