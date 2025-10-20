# pip install selenium beautifulsoup4 requests webdriver-manager langchain sentence-transformers

import os
import json
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# 크롤링 대상 페이지 설정
base_url = "https://www.kinship.com"
target_url = f"{base_url}/dog-behavior/basic-obedience-training"  # 맨 뒷부분만 다른 주소 변경하면 됨 

# Selenium 브라우저 설정
options = webdriver.ChromeOptions()
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
driver.get(target_url)

# article 태그 로딩 대기
WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "article")))
soup = BeautifulSoup(driver.page_source, "html.parser")
articles = soup.find_all("article")  # 전체 크롤링

# JSON 파일 실시간 저장 준비
json_file_path = "kinship_behavior_basic_obedience_training.json"
with open(json_file_path, "w", encoding="utf-8") as f:
    f.write("[")

first_item = True  # 첫 항목 구분용

# 크롤링 시작
for article in articles:
    title_tag = article.find("h3")
    link_tag = title_tag.find("a") if title_tag else None
    title = link_tag.text.strip() if link_tag else None
    url = base_url + link_tag["href"] if link_tag and link_tag.get("href") else None

    if not title or not url:
        continue

    driver.get(url)

    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "article")))
        detail_soup = BeautifulSoup(driver.page_source, "html.parser")
        main_section = detail_soup.find("section", class_="w_dX")

        if main_section:
            text_parts = []
            for tag in main_section.find_all(["h2", "p", "li"]):
                text = tag.get_text(strip=True)
                if text:
                    text_parts.append(text)
            full_text = "\n".join(text_parts)

            # Document 형식으로 생성
            doc = Document(
                page_content=full_text,
                metadata={"title": title, "source": url}
            )

            # 실시간으로 JSON에 저장
            with open(json_file_path, "a", encoding="utf-8") as f:
                if not first_item:
                    f.write(",\n")
                json.dump(doc.dict(), f, ensure_ascii=False, indent=2)
                first_item = False

            print(f"크롤링 및 저장 완료: {title}")
        else:
            print(f"본문 없음 (스킵): {title}")

    except Exception as e:
        print(f"에러 발생 ({url}): {e}")

# JSON 배열 닫기
with open(json_file_path, "a", encoding="utf-8") as f:
    f.write("\n]")

driver.quit()
print(f"모든 문서 크롤링 및 JSON 저장 완료: {json_file_path}")

