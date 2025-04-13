import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# 画像の保存フォルダ（ベースディレクトリ）
BASE_SAVE_DIR = "/home/ikedo/images"
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

# Chrome の WebDriver を自動セットアップ
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # ヘッドレスモード（GUIなしで実行）
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920,1080")
options.add_argument("start-maximized")
options.add_argument("disable-infobars")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# 各成長段階の検索クエリとフォルダ名
growth_stages = {
    "seedling": "tomato seedling",
    "young_plant": "young tomato plant",
    "flowering": "tomato flower",
    "green_fruit": "green tomato on plant",
    "ripe_tomato": "ripe red tomato on plant"
}

def get_next_file_number(save_folder):
    """フォルダ内の既存の画像数を取得し、新しいファイルの番号を決定"""
    existing_files = [f for f in os.listdir(save_folder) if f.endswith(".jpg")]
    if existing_files:
        numbers = [int(f.split(".")[0]) for f in existing_files if f.split(".")[0].isdigit()]
        return max(numbers) + 1 if numbers else 1
    return 1

def download_google_images(query, save_folder, max_images=1300):
    """Google画像検索から指定のフォルダに画像を保存"""
    os.makedirs(save_folder, exist_ok=True)  # フォルダ作成
    search_url = f"https://www.google.com/search?tbm=isch&q={query}"
    driver.get(search_url)
    time.sleep(2)  # ページのロードを待つ

    # スクロールして画像をたくさん表示
    body = driver.find_element(By.TAG_NAME, "body")
    for _ in range(30):
        body.send_keys(Keys.END)
        time.sleep(2)  # 読み込み待機

    # 画像の取得
    image_elements = driver.find_elements(By.CSS_SELECTOR, "img")
    image_urls = [img.get_attribute("src") for img in image_elements if img.get_attribute("src") and img.get_attribute("src").startswith("http")]

    print(f"【{query}】取得した画像数: {len(image_urls)}")

    # 画像を保存（既存のファイルがある場合は続きの番号で保存）
    next_file_number = get_next_file_number(save_folder)
    count = 0

    for i, img_url in enumerate(image_urls[:max_images]):
        try:
            # 画像のサイズをチェック（1KB以上）
            response = requests.head(img_url, timeout=10)
            content_length = int(response.headers.get("Content-Length", 0))
            
            if content_length < 1024 * 5:  # 1KB未満の画像はスキップ
                print(f"【{query}】画像 {i+1} はサイズが小さすぎるためスキップします（サイズ: {content_length} bytes）")
                continue
            
            # 画像を保存
            img_data = requests.get(img_url, timeout=10).content
            file_path = os.path.join(save_folder, f"{next_file_number}.jpg")
            with open(file_path, "wb") as f:
                f.write(img_data)
            count += 1
            next_file_number += 1
            print(f"【{query}】Downloaded {count}/{max_images} ({file_path})")
        except Exception as e:
            print(f"【{query}】Failed to download image {i+1}: {e}")

# 各成長段階ごとに画像を収集（保存フォルダを空にせず蓄積）
for stage, query in growth_stages.items():
    save_path = os.path.join(BASE_SAVE_DIR, stage)
    print(f"\n--- {query} の画像を収集開始 ---")
    download_google_images(query, save_path, max_images=1300)

driver.quit()
