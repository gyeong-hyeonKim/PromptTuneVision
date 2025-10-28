import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os

def test_clip_similarity(prompt_text: str, image_path: str, device: str = "cuda"):
    """
    주어진 프롬프트와 이미지 간의 CLIP 유사도 점수를 계산하여 출력합니다.

    Args:
        prompt_text (str): 유사도를 측정할 텍스트 프롬프트.
        image_path (str): 유사도를 측정할 이미지 파일의 경로.
        device (str): 사용할 장치 ("cuda" 또는 "cpu").
    """
    try:
        # 1. 모델 및 프로세서 로드
        # "openai/clip-vit-base-patch32"는 가장 기본적인 CLIP 모델 중 하나입니다.
        model_id = "openai/clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(model_id).to(device)
        processor = CLIPProcessor.from_pretrained(model_id)
        print(f"CLIP 모델과 프로세서를 성공적으로 로드했습니다. (모델 ID: {model_id})")

        # 2. 이미지 로드 및 전처리
        if not os.path.exists(image_path):
            print(f"오류: 이미지 파일을 찾을 수 없습니다 - {image_path}")
            return

        image = Image.open(image_path).convert("RGB")
        print(f"이미지를 성공적으로 로드했습니다: {image_path}")

        # 3. 입력 준비 (사용자 프롬프트와 이미지)
        # 바로 여기에서 사용자 프롬프트(prompt_text)가 사용됩니다.
        inputs = processor(
            text=[prompt_text],  # 사용자가 제공한 프롬프트
            images=image,        # 로드한 이미지
            return_tensors="pt", # PyTorch 텐서로 반환
            padding=True,        # 패딩 적용
            truncation=True      # 프롬프트가 길 경우 자동 절단 (모델의 최대 토큰 길이에 맞게)
        ).to(device)
        print("모델 입력을 성공적으로 준비했습니다.")

        # 4. 유사도 계산
        # 모델은 준비된 텍스트와 이미지 입력을 받아 로짓(logits)을 출력합니다.
        with torch.no_grad(): # 추론 모드에서는 그래디언트 계산이 필요 없습니다.
            outputs = model(**inputs)
        
        # logits_per_image는 이미지 당 텍스트의 로짓 값을 의미합니다.
        # 이 값을 직접 유사도 점수로 사용하거나, 필요시 softmax를 적용할 수도 있습니다.
        # 여기서는 원본 로짓 값을 사용합니다 (이전 코드와 동일한 방식).
        similarity_score = outputs.logits_per_image[0][0].item()
        print("유사도 점수를 성공적으로 계산했습니다.")

        # 5. 결과 출력
        print("\n--- CLIP 유사도 테스트 결과 ---")
        print(f"입력 프롬프트: \"{prompt_text}\"")
        print(f"이미지 파일: {os.path.basename(image_path)}")
        print(f"계산된 유사도 점수: {similarity_score:.4f}")

    except Exception as e:
        print(f"테스트 중 오류 발생: {e}")

if __name__ == '__main__':
    # --- 테스트 설정 ---
    # 1. 테스트할 프롬프트를 입력하세요.
    test_prompt = "A panda riding a motorcycle in a busy New York city street, camera  zoom out"
    
    # 2. 테스트할 이미지 파일의 실제 경로를 입력하세요.
    #    예시: "path/to/your/image.jpg" 또는 상대 경로 "my_images/cat_image.png"
    #    실제 이미지 파일로 경로를 꼭 수정해주세요!
    test_image_file_path = "data/frames/video1/frame_0000.jpg" # <--- 이 부분을 실제 이미지 파일 경로로 변경하세요!

    # 3. 사용할 장치를 선택하세요 ("cuda" 또는 "cpu"). GPU가 없으면 "cpu"로 설정하세요.
    test_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"테스트 장치: {test_device}")
    
    # 이미지 파일 존재 여부 간단히 확인
    if not os.path.exists(test_image_file_path):
        print(f"\n[주의!] 테스트 이미지 파일 '{test_image_file_path}'을(를) 찾을 수 없습니다.")
        print("위 코드의 'test_image_file_path' 변수를 실제 이미지 파일 경로로 수정해주세요.")
        print("예시: test_image_file_path = \"C:/Users/YourName/Pictures/my_cat.jpg\"")
    else:
        # 테스트 함수 실행
        test_clip_similarity(test_prompt, test_image_file_path, test_device)