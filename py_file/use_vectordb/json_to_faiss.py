import json
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


# JSON을 벡터DB로 변환 
def json_to_faiss(json_path, faiss_dir):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        documents = [Document(**item) for item in data]

    embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-m3") #  <- 현재 가장 성능 좋은 다국어 모델    영어 최적화 모델 : "sentence-transformers/all-MiniLM-L6-v2"
   
    db = FAISS.from_documents(documents, embedding)
    db.save_local(faiss_dir)
    print(f"FAISS 벡터 DB 저장 완료 : {faiss_dir}")

# 벡터DB로 변환
json_to_faiss("C:/Users/maest/Desktop/skn11demo/data/lower_case_merged.json", "faiss_db_all_lower")
