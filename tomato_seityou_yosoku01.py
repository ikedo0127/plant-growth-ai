import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 定数
IMG_SIZE = 128
MODEL_PATH = "plant_growth_model.h5"
image_path = "******/366.jpg"

# 成長段階の予測
def predict_growth(image_path):
    # 学習済みモデルの読み込み
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"⚠️ モデルの読み込みに失敗しました: {e}")
        return None

    # 画像の前処理
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ 画像を読み込めません: {image_path}")
        return None

    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    
    # 画像を部分的に切り取る（ここでは単純に中央部分を切り取ってpart_inputに）
    crop_size = np.random.randint(IMG_SIZE // 2, IMG_SIZE)
    x = np.random.randint(0, IMG_SIZE - crop_size)
    y = np.random.randint(0, IMG_SIZE - crop_size)
    part_img = img[y:y+crop_size, x:x+crop_size]
    part_img_resized = cv2.resize(part_img, (IMG_SIZE, IMG_SIZE)) / 255.0

    # 入力としてfull_inputとpart_inputを作成
    full_input = np.expand_dims(img_resized, axis=0)
    part_input = np.expand_dims(part_img_resized, axis=0)

    # 予測
    try:
        prediction = model.predict([full_input, part_input])  # 2つの入力を渡す
        predicted_label = np.argmax(prediction)
        return predicted_label
    except Exception as e:
        print(f"⚠️ 予測中にエラーが発生しました: {e}")
        return None

if __name__ == "__main__":
    test_image = image_path
    
    label = predict_growth(test_image)
    if label is not None:
        print(f"🌱 予測された成長段階: {label}")
