from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# .env 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 벡터 DB 로드 (병합된 DB)
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3"
)  # "sentence-transformers/all-MiniLM-L6-v2"
db = FAISS.load_local(
    "data/db/faissdb/faiss_db_all_lower",
    embedding,
    allow_dangerous_deserialization=True,
)

# Retriever 생성
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# GPT 모델 설정
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3, openai_api_key=OPENAI_API_KEY)

# 프롬프트 템플릿 구성
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
당신은 반려견 행동 분석 전문가입니다.
아래 문서 내용을 참고하여 사용자의 질문에 대해 정확하고 친절하게, 꼭 '한국어'로 답변해주세요.
혹시, 문서 내용 중 일반적이지 않은 내용(예를 들어, 사용자의 질문에 없는 강아지의 이름이 포함)이 있다면 일반화해서 답변을 제공해주세요. 
[문서 내용]
{context}

[질문]
{question}

[답변]
""",
)

# QA 체인 생성
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)


# 유사도 문제 해결을 위해 한글로 입력받은 질문을 영어로 변환 후 체인에 입력(벡터DB의 문서는 영어, 질문은 한글로 받기 때문에 이슈슈 발생)
def gpt_translate_ko_to_en(text):
    prompt = (
        f"Translate the following Korean sentence into English:\n\n{text}\n\nEnglish:"
    )
    response = llm.invoke(prompt)
    return response.content.strip()


def title_case_excluding_prepositions(text):
    prepositions = {
        "a",
        "an",
        "the",
        "at",
        "by",
        "for",
        "in",
        "of",
        "on",
        "to",
        "up",
        "with",
        "about",
        "above",
        "after",
        "against",
        "along",
        "among",
        "around",
        "as",
        "before",
        "behind",
        "below",
        "beneath",
        "beside",
        "between",
        "beyond",
        "but",
        "concerning",
        "considering",
        "despite",
        "down",
        "during",
        "except",
        "following",
        "from",
        "inside",
        "into",
        "like",
        "near",
        "off",
        "onto",
        "out",
        "outside",
        "over",
        "past",
        "regarding",
        "since",
        "through",
        "throughout",
        "toward",
        "under",
        "underneath",
        "until",
        "upon",
        "via",
        "within",
        "without",
    }

    words = text.lower().split()
    result = []

    for i, word in enumerate(words):
        if i == 0 or word not in prepositions:
            result.append(word.capitalize())
        else:
            result.append(word)

    return " ".join(result)


# 사용자 질문 테스트
query_ko = "왜 내 개가 바닥을 핥을까?"
query_en = gpt_translate_ko_to_en(query_ko)
query = title_case_excluding_prepositions(query_en)

# query_en = gpt_translate_ko_to_en(query_ko)
result = qa_chain(query)

# 출력
print("질문:", query)
print("답변:\n", result["result"])

# 참고 문서 URL 출력
print("\n참고 문서 URL:")
for i, doc in enumerate(result["source_documents"], 1):
    print(f"{i}. {doc.metadata.get('source', 'URL 없음')}")
