"""
Step 3: Keras 모델 → TensorFlow.js 변환
=========================================
입력: ./model/kanji_cnn_final.h5
출력: ./tfjs_model/ (model.json + *.bin 파일들)

이 폴더를 html 파일 옆에 두면 브라우저에서 직접 추론 가능

사용법:
  pip install tensorflowjs
  python 3_convert_tfjs.py
"""

import os
import subprocess
import sys

MODEL_PATH  = './model/kanji_cnn_final.h5'
OUTPUT_DIR  = './tfjs_model'

def convert():
    # tensorflowjs_converter 설치 확인
    try:
        import tensorflowjs
        print(f"tensorflowjs 버전: {tensorflowjs.__version__}")
    except ImportError:
        print("tensorflowjs 미설치 — 설치 중...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorflowjs'])

    if not os.path.exists(MODEL_PATH):
        print(f"모델 파일 없음: {MODEL_PATH}")
        print("먼저 2_train_cnn.py를 실행하세요.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"변환 중: {MODEL_PATH} → {OUTPUT_DIR}")

    # tensorflowjs_converter CLI 호출
    # quantize_float16: 모델 크기 절반으로 줄임 (정확도 거의 유지)
    cmd = [
        'tensorflowjs_converter',
        '--input_format=keras',
        '--output_format=tfjs_layers_model',
        '--quantize_float16',          # float32 → float16 (크기 50% 절감)
        MODEL_PATH,
        OUTPUT_DIR
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        # 변환된 파일 목록
        files = os.listdir(OUTPUT_DIR)
        total_size = sum(
            os.path.getsize(os.path.join(OUTPUT_DIR, f))
            for f in files
        ) / 1024 / 1024

        print(f"\n✓ 변환 완료!")
        print(f"  출력 폴더: {OUTPUT_DIR}/")
        print(f"  파일 목록: {files}")
        print(f"  총 크기: {total_size:.1f} MB")
        print(f"\n다음 단계:")
        print(f"  1. {OUTPUT_DIR}/ 폴더를 kanji-study-v3.html 옆에 복사")
        print(f"  2. html 파일에 TF.js 추론 코드 추가 (Claude가 해줄 예정)")
    else:
        print(f"변환 실패:\n{result.stderr}")

if __name__ == '__main__':
    convert()
