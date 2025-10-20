import chromadb
from sentence_transformers import SentenceTransformer
import json
import uuid
import os
from llm.ksi.call_llm import call_llm

collection_name = "dog_bread"
JSON_PATH = "./data/ksi/Json/dog_documents.json"
chromadb_path = "./data/ksi/DB/chroma_db"


def create_chromadb():
    # Chromadb
    client = chromadb.PersistentClient(path=chromadb_path)

    # collection 유무 파악
    existing_collections = client.list_collections()
    if collection_name in existing_collections:
        collection = client.get_collection(name=collection_name)
    else:
        collection = client.create_collection(name=collection_name)

    # embedding 모델 로드
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # JSON 파일 불러오기
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        documents = json.load(f)

    # ChromaDB에 추가
    for doc in documents:
        doc_id = str(uuid.uuid4())
        content = doc["page_content"]
        metadata = {
            k: (v if isinstance(v, (str, int, float, bool)) else str(v))
            for k, v in doc["metadata"].items()
        }

        embedding = embedding_model.encode(content).tolist()

        collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[doc_id],
            embeddings=[embedding],
        )

    print(f"✅ {len(documents)}개 문서가 ChromaDB에 삽입되었습니다.")


def get_collection(collection_name=collection_name):
    client = chromadb.PersistentClient(path=chromadb_path)

    existing_collections = client.list_collections()
    if collection_name in existing_collections:
        collection = client.get_collection(name=collection_name)
        return collection
    else:
        return "해당 이름의 collection 이 존재하지 않습니다."


def QA_chromadb(collection):

    # 질문 입력
    query = input("질문을 입력: ")

    # 임베딩 모델 로드
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # 질문 -> 임베딩 벡터 변환
    query_vector = embedding_model.encode([query]).tolist()[0]

    # Chroma 검색
    results = collection.query(
        query_embeddings=[query_vector], n_results=2, include=["documents", "metadatas"]
    )

    # 검색된 텍스트 조합
    retrieved_text = "\n\n".join(results["documents"][0])

    print(f"유사 문구 : {retrieved_text}")

    # 프롬프트 생성
    system_prompt = ""
    user_prompt = f"""
    [질문]
    {query}

    [견종 정보]
    {retrieved_text}
    """

    # LLM
    response = call_llm(system_prompt, user_prompt)
    print(response)

    # 출력
    return f"\n답변: {response}"
