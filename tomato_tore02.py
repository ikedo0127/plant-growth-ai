import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, concatenate
import tensorflow as tf
import matplotlib.pyplot as plt

# 定数
IMG_SIZE = 128
DATASET_LABELED = "dataset_labeled/"
MODEL_PATH = "plant_growth_model.h5"
N_CLUSTERS = 5

# 画像データのロード（部分画像と全体画像を別々に取得）

def load_images():
    full_images, part_images, labels_full, labels_part = [], [], [], []

    for label in range(N_CLUSTERS):  
        full_folder = os.path.join(DATASET_LABELED, str(label), "full")
        part_folder = os.path.join(DATASET_LABELED, str(label), "part")
        # 全体画像の取得
        if os.path.exists(full_folder):
            for filename in os.listdir(full_folder):
                img = cv2.imread(os.path.join(full_folder, filename))
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    full_images.append(img)
                    labels_full.append(label)  # ラベルも追加

                    # ランダムクロップした部分画像も同じ数だけ作る
                    crop_size = np.random.randint(IMG_SIZE // 2, IMG_SIZE)
                    x = np.random.randint(0, IMG_SIZE - crop_size)
                    y = np.random.randint(0, IMG_SIZE - crop_size)
                    part_img = img[y:y+crop_size, x:x+crop_size]
                    part_img = cv2.resize(part_img, (IMG_SIZE, IMG_SIZE))
                    part_images.append(part_img)
                    labels_part.append(label)  # 部分画像のラベルも追加
        # もともとの部分画像の取得
        if os.path.exists(part_folder):
            for filename in os.listdir(part_folder):
                img = cv2.imread(os.path.join(part_folder, filename))
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    part_images.append(img)
                    labels_part.append(label)  

    # もし full_images の数が part_images より少なければ、バランスを取る
    min_size = min(len(full_images), len(part_images), len(labels_part))
    full_images, part_images, labels_full, labels_part = (
        full_images[:min_size], part_images[:min_size], labels_full[:min_size], labels_part[:min_size]
    )

    return np.array(full_images) / 255.0, np.array(part_images) / 255.0, np.array(labels_full)

# マルチスケールCNNモデルの構築
def build_model():
    input_full = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="full_input")
    input_part = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="part_input")
    
    def cnn_block(input_layer):
        x = Conv2D(32, (3, 3), activation='relu')(input_layer)
        x = MaxPooling2D(2, 2)(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(2, 2)(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPooling2D(2, 2)(x)
        x = Flatten()(x)
        return x
    
    full_features = cnn_block(input_full)
    part_features = cnn_block(input_part)
    merged = concatenate([full_features, part_features])
    x = Dense(128, activation='relu')(merged)
    x = Dropout(0.5)(x)
    output = Dense(N_CLUSTERS, activation='softmax')(x)
    
    model = Model(inputs=[input_full, input_part], outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# CNNモデルの学習
def train_model():
    full_images, part_images, labels = load_images()
    if len(full_images) == 0:
        print("⚠️ データセットが空です。学習できません。")
        return

    X_train_full, X_test_full, X_train_part, X_test_part, y_train, y_test = train_test_split(
        full_images, part_images, labels, test_size=0.2, random_state=42)

    # データ拡張の定義
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
    ])
    
    # モデルの作成
    model = build_model()

    # データセットを作成
    train_dataset = tf.data.Dataset.from_tensor_slices(((X_train_full, X_train_part), y_train))
    
    # # データ拡張を適用
    # train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x[0]), data_augmentation(x[1]), y))
    # データ拡張を適用
    train_dataset = train_dataset.map(lambda x, y: ((data_augmentation(x[0]), data_augmentation(x[1])), y))

    # バッチ処理とプリフェッチ
    train_dataset = train_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    
    # モデルのトレーニング
    model.fit(train_dataset, epochs=20, validation_data=([X_test_full, X_test_part], y_test))
    
    # モデルの保存
    model.save(MODEL_PATH)
    print(f"✅ モデルの学習完了（{MODEL_PATH} に保存されました）")

if __name__ == "__main__":
    train_model()



# import os
# import cv2
# import numpy as np
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, concatenate
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import tensorflow.keras.backend as K

# # 定数
# IMG_SIZE = 128
# DATASET_LABELED = "dataset_labeled/"
# MODEL_PATH = "plant_growth_model.h5"
# N_CLUSTERS = 5

# # 画像データのロード（部分画像と全体画像を別々に取得）

# def load_images():
#     full_images, part_images, labels_full, labels_part = [], [], [], []

#     for label in range(N_CLUSTERS):  
#         full_folder = os.path.join(DATASET_LABELED, str(label), "full")
#         part_folder = os.path.join(DATASET_LABELED, str(label), "part")
#         # 全体画像の取得
#         if os.path.exists(full_folder):
#             for filename in os.listdir(full_folder):
#                 img = cv2.imread(os.path.join(full_folder, filename))
#                 if img is not None:
#                     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
#                     full_images.append(img)
#                     labels_full.append(label)  # ラベルも追加

#                     # ランダムクロップした部分画像も同じ数だけ作る
#                     crop_size = np.random.randint(IMG_SIZE // 2, IMG_SIZE)
#                     x = np.random.randint(0, IMG_SIZE - crop_size)
#                     y = np.random.randint(0, IMG_SIZE - crop_size)
#                     part_img = img[y:y+crop_size, x:x+crop_size]
#                     part_img = cv2.resize(part_img, (IMG_SIZE, IMG_SIZE))
#                     part_images.append(part_img)
#                     labels_part.append(label)  # 部分画像のラベルも追加
#         # もともとの部分画像の取得
#         if os.path.exists(part_folder):
#             for filename in os.listdir(part_folder):
#                 img = cv2.imread(os.path.join(part_folder, filename))
#                 if img is not None:
#                     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
#                     part_images.append(img)
#                     labels_part.append(label)  

#     # もし full_images の数が part_images より少なければ、バランスを取る
#     min_size = min(len(full_images), len(part_images), len(labels_part))
#     full_images, part_images, labels_full, labels_part = (
#         full_images[:min_size], part_images[:min_size], labels_full[:min_size], labels_part[:min_size]
#     )

#     return np.array(full_images) / 255.0, np.array(part_images) / 255.0, np.array(labels_full)

# # マルチスケールCNNモデルの構築
# def build_model():
#     input_full = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="full_input")
#     input_part = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="part_input")
    
#     def cnn_block(input_layer):
#         x = Conv2D(32, (3, 3), activation='relu')(input_layer)
#         x = MaxPooling2D(2, 2)(x)
#         x = Conv2D(64, (3, 3), activation='relu')(x)
#         x = MaxPooling2D(2, 2)(x)
#         x = Conv2D(128, (3, 3), activation='relu')(x)
#         x = MaxPooling2D(2, 2)(x)
#         x = Flatten()(x)
#         return x
    
#     full_features = cnn_block(input_full)
#     part_features = cnn_block(input_part)
#     merged = concatenate([full_features, part_features])
#     x = Dense(128, activation='relu')(merged)
#     x = Dropout(0.5)(x)
#     output = Dense(N_CLUSTERS, activation='softmax')(x)
    
#     model = Model(inputs=[input_full, input_part], outputs=output)
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     return model

# # CNNモデルの学習
# def train_model():
#     full_images, part_images, labels = load_images()
#     if len(full_images) == 0:
#         print("⚠️ データセットが空です。学習できません。")
#         return

#     X_train_full, X_test_full, X_train_part, X_test_part, y_train, y_test = train_test_split(
#         full_images, part_images, labels, test_size=0.2, random_state=42)

#     datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1,
#                                  brightness_range=[0.7, 1.3], zoom_range=0.2, horizontal_flip=True)
    
#     model = build_model()
    
#     # データセットを作成
#     train_dataset = tf.data.Dataset.from_tensor_slices(((X_train_full, X_train_part), y_train))
    
#     # Augmentation を適用
#     train_dataset = train_dataset.map(lambda x, y: (datagen.flow(x), y))
#     train_dataset = train_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    
#     # モデルのトレーニング
#     model.fit(train_dataset, epochs=20, validation_data=([X_test_full, X_test_part], y_test))
    
#     model.save(MODEL_PATH)
#     print(f"✅ モデルの学習完了（{MODEL_PATH} に保存されました）")

# if __name__ == "__main__":
#     train_model()


# # # Grad-CAM による可視化
# # def grad_cam(model, img_array, layer_name='conv2d_2'):
# #     grad_model = Model([model.inputs], [model.get_layer(layer_name).output, model.output])
# #     with tf.GradientTape() as tape:
# #         conv_output, predictions = grad_model(img_array)
# #         loss = predictions[:, np.argmax(predictions[0])]
# #     grads = tape.gradient(loss, conv_output)[0]
# #     weights = tf.reduce_mean(grads, axis=(0, 1))
# #     cam = np.dot(conv_output[0], weights)
# #     cam = np.maximum(cam, 0)
# #     cam = (cam - cam.min()) / (cam.max() - cam.min())
# #     cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
# #     return cam

# # # Grad-CAMの表示
# # def show_grad_cam(model, img):
# #     img_array = np.expand_dims(img, axis=0)
# #     cam = grad_cam(model, img_array)
# #     plt.imshow(img)
# #     plt.imshow(cam, cmap='jet', alpha=0.5)
# #     plt.axis('off')
# #     plt.show()



