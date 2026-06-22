"""
Step 2: N5 한자 CNN 학습 (Keras/TensorFlow)
============================================
입력: ./dataset/ 폴더 (한자별 PNG 이미지)
출력: ./model/kanji_cnn.h5

모델 구조:
  - 입력: 64x64 흑백 이미지
  - Conv 블록 x3 (BatchNorm + MaxPool + Dropout)
  - Dense 512 → 83 클래스 Softmax
  - 데이터 증강: 회전, 이동, 줌, 기울임

사용법:
  python 2_train_cnn.py --dataset_dir ./dataset --epochs 30
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# N5 한자 83자 (라벨 순서 고정 — 추론 시 동일 순서 필요)
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
NUM_CLASSES = len(N5_KANJI)  # 83
IMG_SIZE    = 64
BATCH_SIZE  = 64

# ============================================================
# 데이터 로드
# ============================================================
def load_dataset(dataset_dir):
    """dataset/ 폴더에서 이미지와 라벨 로드"""
    images = []
    labels = []

    for label_idx, kanji in enumerate(N5_KANJI):
        folder = os.path.join(dataset_dir, kanji)
        if not os.path.exists(folder):
            print(f"  ⚠ 폴더 없음: {kanji}")
            continue

        files = [f for f in os.listdir(folder) if f.endswith('.png')]
        for fname in files:
            img_path = os.path.join(folder, kanji, fname)
            try:
                img = tf.keras.preprocessing.image.load_img(
                    os.path.join(folder, fname),
                    color_mode='grayscale',
                    target_size=(IMG_SIZE, IMG_SIZE)
                )
                arr = tf.keras.preprocessing.image.img_to_array(img)
                images.append(arr)
                labels.append(label_idx)
            except Exception as e:
                print(f"  로드 실패: {fname} → {e}")

        print(f"  {kanji}: {len(files)}장 로드")

    images = np.array(images, dtype=np.float32) / 255.0  # 정규화 0~1
    labels = np.array(labels, dtype=np.int32)
    return images, labels

# ============================================================
# 모델 정의
# Conv 블록 3개 + Dense
# BatchNorm: 학습 안정화
# Dropout: 과적합 방지
# ============================================================
def build_model():
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    # Conv Block 1 — 저수준 특징 (획의 끝, 꺾임)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)          # 64x64 → 32x32
    x = layers.Dropout(0.25)(x)

    # Conv Block 2 — 중간 수준 특징 (부수, 변의 구조)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)          # 32x32 → 16x16
    x = layers.Dropout(0.25)(x)

    # Conv Block 3 — 고수준 특징 (전체 한자 구조)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)          # 16x16 → 8x8
    x = layers.Dropout(0.25)(x)

    # Classifier
    x = layers.GlobalAveragePooling2D()(x) # 파라미터 수 줄이기
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    return model

# ============================================================
# 데이터 증강
# 필기 특성상 좌우 반전은 의미가 달라질 수 있어서 제외
# ============================================================
def get_augmentation():
    return keras.Sequential([
        layers.RandomRotation(0.1),        # ±36도 회전
        layers.RandomTranslation(0.1, 0.1),# ±10% 이동
        layers.RandomZoom(0.1),            # ±10% 줌
    ])

# ============================================================
# 학습
# ============================================================
def train(dataset_dir, epochs, model_dir):
    # GPU 설정
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU 감지: {[g.name for g in gpus]}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("GPU 없음 — CPU로 학습 (느릴 수 있음)")

    # 데이터 로드
    print("\n데이터 로드 중...")
    images, labels = load_dataset(dataset_dir)
    print(f"\n총 이미지: {len(images)}장, 클래스: {NUM_CLASSES}개")

    # Train/Validation 분리 (80:20)
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Train: {len(X_train)}장, Val: {len(X_val)}장")

    # 라벨 원핫 인코딩
    y_train_oh = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val_oh   = keras.utils.to_categorical(y_val, NUM_CLASSES)

    # 데이터 증강 포함 Dataset 생성
    aug = get_augmentation()

    def augment(x, y):
        x = aug(x, training=True)
        return x, y

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_oh))
    train_ds = train_ds.shuffle(len(X_train)).batch(BATCH_SIZE).map(augment).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val_oh))
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # 모델 생성
    model = build_model()
    model.summary()

    # 컴파일
    # Adam + CosineDecay: 안정적이고 수렴 빠름
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3,
        decay_steps=epochs * len(train_ds)
    )
    model.compile(
        optimizer=keras.optimizers.Adam(lr_schedule),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 콜백
    os.makedirs(model_dir, exist_ok=True)
    callbacks = [
        # 최고 val_accuracy 모델 저장
        keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, 'kanji_cnn_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # val_loss 5 에폭 개선 없으면 조기 종료
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # TensorBoard 로그
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(model_dir, 'logs'),
            histogram_freq=1
        ),
    ]

    # 학습
    print("\n학습 시작!")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # 최종 모델 저장 (TF.js 변환용)
    final_path = os.path.join(model_dir, 'kanji_cnn_final.h5')
    model.save(final_path)
    print(f"\n✓ 모델 저장: {final_path}")

    # 라벨 순서 저장 (추론 시 필요)
    labels_path = os.path.join(model_dir, 'labels.txt')
    with open(labels_path, 'w', encoding='utf-8') as f:
        for kanji in N5_KANJI:
            f.write(kanji + '\n')
    print(f"✓ 라벨 저장: {labels_path}")

    # 최고 정확도 출력
    best_acc = max(history.history['val_accuracy'])
    print(f"\n최고 검증 정확도: {best_acc*100:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='./dataset')
    parser.add_argument('--model_dir',   default='./model')
    parser.add_argument('--epochs',      default=30, type=int)
    args = parser.parse_args()

    train(args.dataset_dir, args.epochs, args.model_dir)
