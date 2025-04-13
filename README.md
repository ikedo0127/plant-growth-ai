# plant-growth-ai
CNNを用いて画像から植物の成長段階をAIで判定するシステムです。

## 特徴
- CNNによる画像分類　#tomato_tore02
- 成長ステージ0〜4をラベル付け  #app.py
- Flask API化しWebアプリ連携可能　#app.py
- スクレイピングによる画像収集　#sukure_01
- 教師なし学習によるクラスタリング　#tomaro_kurasu01
## 実行方法
1. `python app.py`
2. `python sukure_01.py`
3. `python tomaro_kurasu01.py`
4. `python tomato_tore02.py`


## 使用技術
- Python, TensorFlow, Flask,cv2,shutil,matplotlib,scikit-learn
