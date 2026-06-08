"""반려견 행동 분석 RAG 백엔드.

T1 개선 사항
- T1-2: 대화 메모리 — 직전 대화를 참고해 후속 질문을 독립형 질문으로 재작성(history-aware).
- T1-3: 의존성 마이그레이션 — langchain.* (deprecated) → langchain_community.* / langchain_core.*.
- T1-4: 교차언어 검색 단일화 — 영어 출처가 최상위일 때만 영어 질의로 추가 검색하고,
        답변 생성(LLM)은 단 1회만 수행해 기존의 이중 호출(검색+LLM 2배)을 제거.
- 리소스 로딩을 팩토리 함수(build_resources)로 분리 → app.py에서 st.cache_resource로 캐싱(T1-1).
"""

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

# .env 로드
load_dotenv()

# ── 설정 ──────────────────────────────────────────────────────────
FAISS_DB_PATH = "data/db/faissdb/faiss_db_all_lower"
EMBEDDING_MODEL = "BAAI/bge-m3"
LLM_MODEL = "gpt-4.1-mini"
RETRIEVE_K = 3
HISTORY_MAX_MESSAGES = 8  # 최근 4턴만 맥락으로 사용(토큰 비용 제어)
KINSHIP_PREFIX = "https://www.kinship.com"
SOURCE_MARKER = "\n\n📚 **참고 문서:**"  # app.py 표시용 출처 블록 구분자

# ── 프롬프트 ──────────────────────────────────────────────────────
# 후속 질문을 직전 대화 맥락으로 풀어 독립형 질문으로 재작성(T1-2)
CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "이전 대화 내용과 사용자의 마지막 질문이 주어집니다. "
            "마지막 질문이 이전 대화를 참조해야만 이해되는 경우, 대화 없이도 그 자체로 "
            "이해되는 독립적인 질문으로 다시 작성하세요. "
            "질문에 답하지 말고, 다시 작성한 질문만 출력하거나 이미 독립적이면 그대로 출력하세요.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# 최종 답변 생성 프롬프트(기존 프롬프트 의도 유지 + 대화 맥락 반영)
ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 반려견 행동 분석 전문가입니다.\n"
            "아래 문서 내용을 참고하여 사용자의 질문에 대해 정확하고 친절하게, 꼭 '한국어'로 "
            "답변해주세요.\n"
            "혹시, 문서 내용 중 일반적이지 않은 내용(예를 들어, 사용자의 질문에 없는 강아지의 "
            "이름이 포함)이 있다면 일반화해서 답변을 제공해주세요.\n\n"
            "[문서 내용]\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


# ── 리소스 로딩 (무거움 → app.py에서 캐싱) ─────────────────────────
def load_retriever():
    """임베딩 + FAISS 로드 후 retriever 반환."""
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.load_local(
        FAISS_DB_PATH,
        embedding,
        allow_dangerous_deserialization=True,
    )
    return db.as_retriever(
        search_type="similarity", search_kwargs={"k": RETRIEVE_K}
    )


def build_llm(streaming: bool = False):
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.3,
        streaming=streaming,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


def build_resources():
    """앱에서 한 번만 호출해 캐싱할 무거운 리소스 묶음."""
    retriever = load_retriever()
    util_llm = build_llm(streaming=False)  # 질문 재작성 / 번역용
    answer_llm = build_llm(streaming=True)  # 최종 답변 생성용(스트리밍)
    return {
        "retriever": retriever,
        "llm": util_llm,
        "contextualize_chain": CONTEXTUALIZE_PROMPT | util_llm | StrOutputParser(),
        "answer_chain": create_stuff_documents_chain(answer_llm, ANSWER_PROMPT),
    }


# ── 번역 유틸 (교차언어 검색용) ───────────────────────────────────
def gpt_translate_ko_to_en(text, llm):
    """한글 질문을 영어로 번역(벡터DB의 영어 문서와 유사도 매칭 향상용)."""
    prompt = (
        f"Translate the following Korean sentence into English:\n\n{text}\n\nEnglish:"
    )
    return llm.invoke(prompt).content.strip()


def title_case_excluding_prepositions(text):
    prepositions = {
        "a", "an", "the", "at", "by", "for", "in", "of", "on", "to", "up", "with",
        "about", "above", "after", "against", "along", "among", "around", "as",
        "before", "behind", "below", "beneath", "beside", "between", "beyond",
        "but", "concerning", "considering", "despite", "down", "during", "except",
        "following", "from", "inside", "into", "like", "near", "off", "onto", "out",
        "outside", "over", "past", "regarding", "since", "through", "throughout",
        "toward", "under", "underneath", "until", "upon", "via", "within", "without",
    }

    words = text.lower().split()
    result = []
    for i, word in enumerate(words):
        if i == 0 or word not in prepositions:
            result.append(word.capitalize())
        else:
            result.append(word)
    return " ".join(result)


# ── 핵심 파이프라인 ───────────────────────────────────────────────
def to_lc_history(messages):
    """app의 [{'role','content'}] 메시지 → langchain 메시지 리스트(최근 N개).

    role: 'user' → HumanMessage, 그 외 → AIMessage(표시용 출처 블록 제거).
    """
    history = []
    for m in messages[-HISTORY_MAX_MESSAGES:]:
        content = m.get("content", "")
        if m.get("role") == "user":
            history.append(HumanMessage(content=content))
        else:
            history.append(AIMessage(content=content.split(SOURCE_MARKER)[0]))
    return history


def contextualize_question(question, chat_history, resources):
    """대화 맥락이 있으면 독립형 질문으로 재작성(T1-2). 없으면 원문 유지."""
    if not chat_history:
        return question
    rewritten = resources["contextualize_chain"].invoke(
        {"chat_history": chat_history, "input": question}
    )
    return (rewritten or question).strip()


def _dedup_docs(docs):
    seen, out = set(), []
    for d in docs:
        key = (d.metadata.get("source", ""), d.page_content[:80])
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def adaptive_retrieve(query, resources):
    """교차언어 검색(T1-4).

    한국어 질의로 1차 검색 후, 최상위 출처가 영어(kinship)이면 영어 질의로 추가 검색한다.
    영어 질의 결과를 우선 배치(올바른 문서가 1위로 오도록)하고 한국어 결과로 보강한다.
    답변 생성(LLM)은 호출부에서 이후 단 1회만 수행 → 기존 이중 호출 제거.
    """
    retriever = resources["retriever"]
    docs = retriever.invoke(query)
    if docs and docs[0].metadata.get("source", "").startswith(KINSHIP_PREFIX):
        query_en = gpt_translate_ko_to_en(query, resources["llm"])
        query_en = title_case_excluding_prepositions(query_en)
        en_docs = retriever.invoke(query_en)
        docs = _dedup_docs(en_docs + docs)[:RETRIEVE_K]
    return docs


def answer_stream(question, chat_history, resources):
    """(출처 문서 리스트, 답변 토큰 제너레이터)를 반환.

    답변 LLM 호출은 단 1회(스트리밍). 검색은 history-aware로 재작성된 질의를 사용하고,
    답변 프롬프트에는 원문 질문 + 대화 맥락을 전달한다.
    """
    standalone_q = contextualize_question(question, chat_history, resources)
    docs = adaptive_retrieve(standalone_q, resources)
    stream = resources["answer_chain"].stream(
        {"context": docs, "chat_history": chat_history, "input": question}
    )
    return docs, stream


def answer_question(question, chat_history=None, resources=None):
    """비스트리밍 편의 함수(스크립트/테스트용). dict(answer, source_documents) 반환."""
    resources = resources or build_resources()
    chat_history = chat_history or []
    docs, stream = answer_stream(question, chat_history, resources)
    text = "".join(stream)
    return {"answer": text, "source_documents": docs}


if __name__ == "__main__":
    res = build_resources()

    # 1턴: 단독 질문
    out = answer_question("왜 내 개가 바닥을 핥을까?", resources=res)
    print("답변:\n", out["answer"])
    print("\n참고 문서:")
    for i, d in enumerate(out["source_documents"], 1):
        print(f"{i}. {d.metadata.get('source', 'URL 없음')}")

    # 2턴: 후속 질문(대화 메모리 동작 확인)
    history = [
        HumanMessage(content="왜 내 개가 바닥을 핥을까?"),
        AIMessage(content=out["answer"]),
    ]
    out2 = answer_question("그럼 어떻게 멈추게 해?", chat_history=history, resources=res)
    print("\n[후속] 답변:\n", out2["answer"])
