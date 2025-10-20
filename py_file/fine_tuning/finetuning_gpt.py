# import json

# # 기존 JSON 로드
# with open("./data/merged_all_json.json", "r", encoding="utf-8") as f:
#     raw_data = json.load(f)

# # SFT 포맷으로 변환
# sft_data = []
# for item in raw_data:
#     prompt = f"{item['metadata'].get('title', '제목 없음')} 문서 내용을 알려줘."
#     response = item['page_content']
#     sft_data.append({
#         "prompt": prompt,
#         "response": response
#     })

# # 저장
# with open("sft_data.jsonl", "w", encoding="utf-8") as f:
#     for entry in sft_data:
#         f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# print("SFT 데이터 준비 완료")




# from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
# from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
# from datasets import load_dataset
# import torch

# # 모델 선택 (LLaMA 3 8B 예시)
# base_model_name = "gpt-4.1-mini"    # meta-llama/Llama-3-8b-hf
# tokenizer = AutoTokenizer.from_pretrained(base_model_name,token='')

# model = AutoModelForCausalLM.from_pretrained(

#     base_model_name,
#     load_in_4bit=True,
#     torch_dtype=torch.float16,
#     device_map="auto"
# )

# # PEFT DoRA Config
# lora_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,
#     r=64,
#     lora_alpha=16,
#     target_modules=["q_proj", "v_proj"],  # LLaMA 3 기준
#     lora_dropout=0.05,
#     bias="none",
#     use_dora=True  # 핵심: DoRA 활성화
# )

# # DoRA 적용
# model = prepare_model_for_kbit_training(model)
# model = get_peft_model(model, lora_config)

# # 데이터 로딩
# dataset = load_dataset("json", data_files="sft_data.jsonl", split="train")

# # 텍스트 포맷팅 함수
# def format_prompt(example):
#     return tokenizer(
#         f"<|user|>\n{example['prompt']}\n<|assistant|>\n{example['response']}",
#         truncation=True,
#         max_length=2048,
#         padding="max_length"
#     )

# tokenized_dataset = dataset.map(format_prompt, batched=True, remove_columns=dataset.column_names)

# # Training Arguments
# training_args = TrainingArguments(
#     output_dir="./dora_llama3_finetuned",
#     num_train_epochs=3,
#     per_device_train_batch_size=4,
#     gradient_accumulation_steps=8,
#     optim="paged_adamw_32bit",
#     save_strategy="epoch",
#     logging_steps=10,
#     learning_rate=2e-5,
#     lr_scheduler_type="cosine",
#     warmup_ratio=0.1,
#     fp16=True,
#     bf16=False,
#     report_to="none"
# )

# # Trainer 실행
# trainer = Trainer(
#     model=model,
#     train_dataset=tokenized_dataset,
#     args=training_args
# )

# trainer.train()

# # 최종 LoRA + DoRA 저장
# model.save_pretrained("./finetuned_model")

# print("DoRA 파인튜닝 완료")
























from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# training_file = client.files.create(
#     file=open("test_fine_tuning_data.jsonl", "rb"),
#     purpose="fine-tune"
# )

# print(training_file.id)

# fine_tuning_job = client.fine_tuning.jobs.create(
#     training_file=training_file.id,
#     model="gpt-4.1-mini-2025-04-14"
# )

# print(fine_tuning_job.id)

# import time

# while True:
#     job_status = client.fine_tuning.jobs.retrieve(fine_tuning_job.id)
#     print(job_status.status)

#     if job_status.status in ["succeeded", "failed"]:
#         break
    
#     time.sleep(5)

# if job_status.status == "succeeded":
#     print("파인튜닝 성공: 모델:", job_status.fine_tuned_model)
# else:
#     print("파인튜닝 실패.")


# completion = client.chat.completions.create(
#     model="ft:gpt-4.1-mini-2025-04-14:jskkk::BWiGHnKs",
#     messages=[
#         {"role": "user", "content": "푸들에 대해서 알려줘"}
#     ]
# )

completion = client.chat.completions.create(
    model="ft:gpt-4.1-mini-2025-04-14:jskkk::BWiGHnKs",
    messages=[
        {"role": "user", "content": "신라면에 대해서 알려줘."}
    ]
)

response = completion.choices[0].message.content
print(response)


