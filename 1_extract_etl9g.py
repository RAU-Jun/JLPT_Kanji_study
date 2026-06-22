"""
Step 1: ETL9G → N5 83자 이미지 추출
=====================================
ETL9G 구조:
  - 50개 파일 (ETL9G_01 ~ ETL9G_50)
  - 각 파일: 여러 레코드, 레코드당 1자
  - 이미지: 128x127 흑백, JIS코드로 문자 식별

사용법:
  python 1_extract_etl9g.py --etl_dir ./ETL9G --out_dir ./dataset
"""

import os
import struct
import argparse
import numpy as np
from PIL import Image

# ============================================================
# N5 한자 83자 목록 (Unicode)
# ============================================================
N5_KANJI = [
    '一','二','三','四','五','六','七','八','九','十',
    '百','千','万','日','月','火','水','木','金','土',
    '山','川','田','人','大','小','上','下','中','口',
    '手','足','目','耳','車','気','天','花','草','雨',
    '学','校','先','生','年','男','女','子','父','母',
    '何','今','来','行','見','聞','話','読','書','食',
    '飲','出','入','休','買','高','安','白','黒','赤',
    '青','右','左','東','西','南','北','国','語','友',
    '外','時','間'
]

# Unicode → JIS X 0208 변환 테이블 생성
# ETL9G는 JIS X 0208 코드로 문자를 저장함
def build_unicode_to_jis():
    """
    Python의 cp932(Shift-JIS) 인코딩을 이용해
    유니코드 → JIS 코드 매핑 테이블 생성
    ETL9G는 JIS 코드를 big-endian 2바이트로 저장
    """
    mapping = {}
    for kanji in N5_KANJI:
        try:
            # cp932로 인코딩 후 JIS 코드 추출
            sjis = kanji.encode('cp932')
            if len(sjis) == 2:
                # Shift-JIS → JIS 변환
                b1, b2 = sjis[0], sjis[1]
                # Shift-JIS to JIS 변환 공식
                if b1 >= 0xE0:
                    b1 -= 0x40
                b1 -= 0x71
                b1 = (b1 << 1) + 1
                if b2 >= 0x9F:
                    b2 -= 0x7E
                    b1 += 1
                elif b2 >= 0x40:
                    b2 -= 0x1F
                # JIS 코드 = (b1 << 8) | b2
                jis_code = (b1 << 8) | b2
                mapping[jis_code] = kanji
        except Exception as e:
            print(f"  변환 실패: {kanji} → {e}")
    return mapping

# ============================================================
# ETL9G 레코드 파싱
# ETL9G 레코드 구조 (8199 bytes):
#   offset  0: JIS코드 (2 bytes, big-endian)
#   offset  2: 시리얼번호 (2 bytes)
#   offset  4: JIS 보조코드 (2 bytes)
#   offset  6: 이미지 품질 (1 byte)
#   offset  7: 예약 (1 byte)
#   offset  8: 이미지 데이터 (128*127 / 4 bits = 8128 bytes)
#              → 4비트 그레이스케일, 픽셀당 0~15
#   offset  8136: 예약 (63 bytes)
# ============================================================
RECORD_SIZE = 8199
IMAGE_WIDTH  = 128
IMAGE_HEIGHT = 127

def read_record(f):
    """ETL9G 파일에서 레코드 하나 읽기"""
    data = f.read(RECORD_SIZE)
    if len(data) < RECORD_SIZE:
        return None, None

    # JIS 코드 추출 (big-endian 2바이트)
    jis_code = struct.unpack('>H', data[0:2])[0]

    # 이미지 데이터 추출 (4비트 그레이스케일)
    # 각 바이트가 두 픽셀을 담고 있음 (상위 4비트, 하위 4비트)
    img_data = data[8:8 + IMAGE_WIDTH * IMAGE_HEIGHT // 2]
    pixels = []
    for byte in img_data:
        # 상위 4비트 → 첫 번째 픽셀
        pixels.append((byte >> 4) & 0xF)
        # 하위 4비트 → 두 번째 픽셀
        pixels.append(byte & 0xF)

    # 0~15 범위를 0~255로 스케일
    pixels = np.array(pixels[:IMAGE_WIDTH * IMAGE_HEIGHT], dtype=np.uint8)
    pixels = (pixels * 17).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)  # 17 = 255/15

    return jis_code, pixels

def extract_etl9g(etl_dir, out_dir):
    """ETL9G 전체 파일에서 N5 한자 이미지 추출"""

    os.makedirs(out_dir, exist_ok=True)
    jis_map = build_unicode_to_jis()

    print(f"추출 대상: {len(N5_KANJI)}자")
    print(f"JIS 매핑 완료: {len(jis_map)}자")
    print(f"ETL9G 디렉토리: {etl_dir}")
    print(f"출력 디렉토리: {out_dir}")
    print("=" * 50)

    # 각 한자별 저장 폴더 생성
    for kanji in N5_KANJI:
        os.makedirs(os.path.join(out_dir, kanji), exist_ok=True)

    # 카운터
    counts = {k: 0 for k in N5_KANJI}
    total_records = 0

    # ETL9G 파일 순회 (ETL9G_01 ~ ETL9G_50)
    for file_idx in range(1, 51):
        filename = f"ETL9G_{file_idx:02d}"
        filepath = os.path.join(etl_dir, filename)

        if not os.path.exists(filepath):
            print(f"  파일 없음: {filename}")
            continue

        print(f"처리 중: {filename}")
        with open(filepath, 'rb') as f:
            while True:
                jis_code, pixels = read_record(f)
                if jis_code is None:
                    break

                total_records += 1

                # N5 한자인지 확인
                if jis_code in jis_map:
                    kanji = jis_map[jis_code]
                    count = counts[kanji]
                    save_path = os.path.join(out_dir, kanji, f"{count:04d}.png")

                    # 이미지 저장 (흑백 반전: 배경 흰색, 글자 검정)
                    img = Image.fromarray(255 - pixels)
                    img = img.resize((64, 64), Image.LANCZOS)
                    img.save(save_path)
                    counts[kanji] += 1

    # 결과 출력
    print("\n" + "=" * 50)
    print("추출 완료!")
    print(f"총 레코드 수: {total_records:,}")
    print(f"\n한자별 추출 수:")
    for kanji in N5_KANJI:
        print(f"  {kanji}: {counts[kanji]}장")

    missing = [k for k in N5_KANJI if counts[k] == 0]
    if missing:
        print(f"\n⚠ 추출 실패 ({len(missing)}자): {' '.join(missing)}")
    else:
        print(f"\n✓ 전체 {len(N5_KANJI)}자 정상 추출!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--etl_dir', default='./ETL9G',
                        help='ETL9G 폴더 경로 (ETL9G_01~50 파일들이 있는 폴더)')
    parser.add_argument('--out_dir', default='./dataset',
                        help='이미지 저장 경로')
    args = parser.parse_args()

    extract_etl9g(args.etl_dir, args.out_dir)
