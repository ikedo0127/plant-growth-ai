import os
import cv2
import numpy as np
import shutil
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model

# 定数
IMG_SIZE = 128
DATASET_RAW = "/home/ikedo/hackathon_2025/tomato-test"
DATASET_LABELED = "dataset_labeled02/"
N_CLUSTERS = 5
N_COMPONENTS = 50

# 画像特徴量抽出
def extract_features():
    base_model = ResNet50(weights=None, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.get_layer("conv5_block1_1_conv").output)

    image_files, features = [], []
    for filename in os.listdir(DATASET_RAW):
        img_path = os.path.join(DATASET_RAW, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = np.expand_dims(img / 255.0, axis=0)
        feature = feature_extractor.predict(img)[0].flatten()
        features.append(feature)
        image_files.append(img_path)

    return np.array(features), image_files

# クラスタリング & 可視化
def create_folders():
    for i in range(N_CLUSTERS):
        os.makedirs(os.path.join(DATASET_LABELED, str(i)), exist_ok=True)

def move_images(image_files, cluster_labels):
    for img_path, cluster_id in zip(image_files, cluster_labels):
        new_path = os.path.join(DATASET_LABELED, str(cluster_id), os.path.basename(img_path))
        shutil.move(img_path, new_path)

def tsne_visualization(features, cluster_labels):
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=cluster_labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label="Cluster ID")
    plt.title('t-SNE Visualization of Feature Vectors')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.savefig('output_plot.png')  # プロットを画像ファイルとして保存
   

def cluster_images():
    features, image_files = extract_features()
    pca_features = PCA(n_components=N_COMPONENTS).fit_transform(features)
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10).fit(pca_features)

    create_folders()
    move_images(image_files, kmeans.labels_)
    tsne_visualization(pca_features, kmeans.labels_)

    print("✅ クラスタリング完了")

if __name__ == "__main__":
    cluster_images()
