import os
import json

# 원본 JSON 파일들이 들어있는 폴더 경로
folder_path = './data/ksi/json' 
output_folder = './data/json_all'
os.makedirs(output_folder, exist_ok=True)

# 폴더 내 모든 json 파일 반복
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 새 형식으로 변환
        new_data = [
            {
                "metadata": {
                    "title": data.get('title', 'No Title')
                },
                "page_content": "\n".join(data.get('text_chunks', []))
            }
        ]

        # 저장할 경로
        output_path = os.path.join(output_folder, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)

print("변환 완료")
