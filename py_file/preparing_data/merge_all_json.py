import os
import json

# 경로 설정
kinship_file = './data/jsk/json_kinship/merged_kinship.json'
breed_folder = './data/json_breed'

all_documents = []

# Kinship JSON 파일 로드
with open(kinship_file, 'r', encoding='utf-8') as f:
    kinship_docs = json.load(f)
    all_documents.extend(kinship_docs)

print(f"Kinship 문서 {len(kinship_docs)}개 추가 완료")

# 견종 정보 파일 로드
for filename in os.listdir(breed_folder):
    if filename.endswith('.json'):
        file_path = os.path.join(breed_folder, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            breed_doc = json.load(f)

            # 리스트인지 확인 후 처리
            # if isinstance(breed_doc, list):
            for doc in breed_doc:
                doc['metadata']['source'] = 'No url'
                all_documents.append(doc)
            # else:
            #     breed_doc['metadata']['source'] = 'No url'
            #     all_documents.append(breed_doc)

print(f"견종 문서 {len(all_documents) - len(kinship_docs)}개 추가 완료")

# 최종 저장
with open('merged_all_json.json', 'w', encoding='utf-8') as f:
    json.dump(all_documents, f, ensure_ascii=False, indent=2)

print(f"전체 문서 총 {len(all_documents)}개 병합 및 정리 완료")
