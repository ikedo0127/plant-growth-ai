from flask import Flask, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Flaskアプリの作成
app = Flask(__name__)

# 定数
IMG_SIZE = 128
MODEL_PATH = "plant_growth_model.h5"

# ✅ モデルを最初にロード（グローバル変数として保持）
try:
    model = load_model(MODEL_PATH)
    print("✅ モデルを正常にロードしました。")
except Exception as e:
    print(f"⚠️ モデルのロードに失敗しました: {e}")
    model = None

# 成長段階の予測
def predict_growth(image_path):
    if model is None:
        return None

    # 画像の前処理
    img = cv2.imread(image_path)
    if img is None:
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
        return int(predicted_label)  # int型に変換して返す
    except Exception as e:
        print(f"⚠️ 予測中にエラーが発生しました: {e}")
        return None

# ルートパスに対応するエンドポイントを追加
@app.route('/')
def home():
    return "Flaskアプリケーションが正常に動作しています！"

# /predictエンドポイント
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_path = data.get("image_path")
    if image_path is None:
        return jsonify({"error": "画像パスが指定されていません"}), 400

    label = predict_growth(image_path)
    if label is not None:
        return jsonify({"predicted_label": label})
    else:
        return jsonify({"error": "予測に失敗しました"}), 500

if __name__ == "__main__":
    app.run(debug=False, port=5001)


# from flask import Flask, request, jsonify
# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model

# # Flaskアプリの作成
# app = Flask(__name__)

# # ✅ モデルを最初にロード（グローバル変数として保持）
# try:
#     model = load_model("plant_growth_model.h5")
#     print("✅ モデルを正常にロードしました。")
# except Exception as e:
#     print(f"⚠️ モデルのロードに失敗しました: {e}")
#     model = None

# # 成長段階の予測
# def predict_growth(image_path):
#     if model is None:
#         return None

#     img = cv2.imread(image_path)
#     if img is None:
#         return None

#     img_resized = cv2.resize(img, (128, 128)) / 255.0
#     full_input = np.expand_dims(img_resized, axis=0)

#     try:
#         prediction = model.predict(full_input)
#         predicted_label = np.argmax(prediction)
#         return predicted_label
#     except Exception as e:
#         print(f"⚠️ 予測中にエラーが発生しました: {e}")
#         return None

# # ルートパスに対応するエンドポイントを追加
# @app.route('/')
# def home():
#     return "Flaskアプリケーションが正常に動作しています！"

# # /predictエンドポイント
# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json
#     image_path = data.get("image_path")
#     label = predict_growth(image_path)
#     return jsonify({"predicted_label": label})

# if __name__ == "__main__":
#     app.run(debug=False, port=5001)

# # from flask import Flask, request, jsonify
# # import cv2
# # import numpy as np
# # from tensorflow.keras.models import load_model

# # # Flaskアプリの作成
# # app = Flask(__name__)

# # # ✅ モデルを最初にロード（グローバル変数として保持）
# # try:
# #     model = load_model("plant_growth_model.h5")
# #     print("✅ モデルを正常にロードしました。")
# # except Exception as e:
# #     print(f"⚠️ モデルのロードに失敗しました: {e}")
# #     model = None

# # # 成長段階の予測
# # def predict_growth(image_path):
# #     if model is None:
# #         return None

# #     img = cv2.imread(image_path)
# #     if img is None:
# #         return None

# #     img_resized = cv2.resize(img, (128, 128)) / 255.0
# #     full_input = np.expand_dims(img_resized, axis=0)

# #     try:
# #         prediction = model.predict(full_input)
# #         predicted_label = np.argmax(prediction)
# #         return predicted_label
# #     except Exception as e:
# #         print(f"⚠️ 予測中にエラーが発生しました: {e}")
# #         return None

# # @app.route('/predict', methods=['POST'])

# # def predict():
# #     data = request.json
# #     image_path = data.get("image_path")
# #     label = predict_growth(image_path)
# #     return jsonify({"predicted_label": label})

# # if __name__ == "__main__":
# #     app.run(debug=False)
