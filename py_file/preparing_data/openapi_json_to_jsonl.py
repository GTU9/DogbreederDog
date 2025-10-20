import json

# 기존 JSON 파일 경로
input_json = "test_json.json"
# 변환된 jsonl 저장 경로
output_jsonl = "test_fine_tuning_data.jsonl"

# JSON 파일 읽기
with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)

# 변환 및 저장
with open(output_jsonl, "w", encoding="utf-8") as f_out:
    for item in data:
        title = item['metadata'].get('title', '제목 없음')
        page_content = item.get('page_content', '')

        # user 질문은 title + 간단한 요청
        user_prompt = f"{title}에 대해 자세히 알려줘."
        assistant_response = page_content.strip()

        # OpenAI Fine-Tuning 형식에 맞게 메시지 쌍 구성
        json_line = {
            "messages": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response}
            ]
        }

        # 한 줄씩 저장
        f_out.write(json.dumps(json_line, ensure_ascii=False) + "\n")

print(f"변환 완료 {output_jsonl}")
