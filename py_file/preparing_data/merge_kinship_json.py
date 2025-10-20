import os
import json

# Kinship JSON 파일들이 들어있는 폴더
kinship_folder = './data/jsk/json_kinship'
merged_kinship = []

# 모든 JSON 파일 반복
for filename in os.listdir(kinship_folder):
    if filename.endswith('.json'):
        file_path = os.path.join(kinship_folder, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            merged_kinship.extend(data)

print(f"Kinship 총 {len(merged_kinship)}개의 문서 병합 완료")

# 병합된 JSON 저장
with open('merged_kinship.json', 'w', encoding='utf-8') as f:
    json.dump(merged_kinship, f, ensure_ascii=False, indent=2)

print("저장 완료: merged_kinship.json")
