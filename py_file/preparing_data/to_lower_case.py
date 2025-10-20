import json

# JSON 파일 로드
with open("data/db/faissdb/merged_all_json.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 모든 문서의 title을 소문자로 변환
for item in data:
    if "title" in item["metadata"]:
        item["metadata"]["title"] = item["metadata"]["title"].lower()

# 저장
with open("data/db/faissdb/merged_all_json.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("title 필드 소문자 변환 완료")
