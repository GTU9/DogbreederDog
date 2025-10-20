from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# 임베딩 모델 로드 (DB 생성에 사용된 모델과 동일해야 함)
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 병합할 DB들 로드
db_paths = [
    "kinship_faiss_db_behavioral_basic_obedience_training",
    "kinship_faiss_db_behavioral_anxiety_separation_anxiety",
    "kinship_faiss_db_behavioral_body_language",
    "kinship_faiss_db_behavioral_issues"
]

# 첫 DB 로드
merged_db = FAISS.load_local(db_paths[0], embedding, allow_dangerous_deserialization=True)

# 나머지 DB들들 병합
for path in db_paths[1:]:
    other_db = FAISS.load_local(path, embedding, allow_dangerous_deserialization=True)
    merged_db.merge_from(other_db)

# 저장
merged_db.save_local("kinship_faiss_db_merged")
print("모든 DB 병합 완료 및 저장: kinship_faiss_db_merged")

# 저장된 총 문서 수 출력
print(f"총 문서 수: {len(merged_db.docstore._dict)}")
