"""반려견 행동 분석 RAG 백엔드.

T1 개선
- T1-2: 대화 메모리(history-aware 질문 재작성)로 후속 질문 지원.
- T1-3: langchain.* (deprecated) → langchain_community.* / langchain_core.*.
- T1-4: 교차언어 검색 단일화(영어 출처일 때만 추가 검색, 답변 생성 1회).
- 리소스 로딩을 build_resources()로 분리 → app.py에서 st.cache_resource 캐싱(T1-1).

T2 개선 (RAG 품질)
- T2-1: 하이브리드 검색(BM25 + dense Ensemble) + bge-reranker 재정렬. 모두 환경변수로
        토글하며, 의존성/모델이 없으면 안전하게 단계적 폴백(dense-only)으로 동작.
- T2-3: 범위 외(개 이외) 질문 가드레일 + 건강 의심 시 수의사 상담 면책 고지.
"""

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import logging
import os

# .env 로드
load_dotenv()
logger = logging.getLogger("dogbreeder.qa")

# ── 설정 (환경변수로 조정 가능) ───────────────────────────────────
FAISS_DB_PATH = os.getenv("FAISS_DB_PATH", "data/db/faissdb/faiss_db_all_lower")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "3"))         # 최종 LLM에 넘길 문서 수
FETCH_K = int(os.getenv("FETCH_K", "10"))              # 재정렬 전 후보 풀 크기
HISTORY_MAX_MESSAGES = int(os.getenv("HISTORY_MAX_MESSAGES", "8"))
KINSHIP_PREFIX = "https://www.kinship.com"
SOURCE_MARKER = "\n\n📚 **참고 문서:**"  # 구버전 호환용 구분자

# T2-1 토글
USE_HYBRID = os.getenv("USE_HYBRID", "1").lower() not in ("", "0", "false", "off", "none")
USE_RERANKER = os.getenv("USE_RERANKER", "1").lower() not in ("", "0", "false", "off", "none")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.4"))   # 나머지는 dense 가중치

# T2-3 토글
ENABLE_OOS_GUARD = os.getenv("ENABLE_OOS_GUARD", "1").lower() not in ("", "0", "false", "off", "none")

# T2-3: 건강 의심 키워드(이 단어가 보이면 수의사 상담 면책 고지 추가)
HEALTH_KEYWORDS = (
    "구토", "토하", "토함", "설사", "혈변", "출혈", "피가", "피를", "절뚝", "절뚜",
    "발작", "경련", "식욕", "안 먹", "안먹", "못 먹", "못먹", "사료를 거부", "거부",
    "떨어", "떨고", "떨림", "호흡", "숨을", "숨이", "기침", "재채기", "열이", "발열",
    "무기력", "처지", "탈수", "쓰러", "아파", "아픈", "통증", "콧물", "눈곱", "종양",
    "혹이", "체중", "살이 빠", "황달", "경기", "마비", "쇼크",
)
HEALTH_DISCLAIMER = (
    "\n\n⚠️ 증상이 지속되거나 심해지면 반드시 수의사와 상담하세요. "
    "본 답변은 일반적인 정보 제공용이며 수의학적 진단을 대체하지 않습니다."
)
OOS_RESPONSE = (
    "죄송해요, 저는 '반려견(개)'의 행동·견종 상담에 특화된 챗봇이에요. 🐶\n"
    "해당 질문은 제가 학습한 자료 범위를 벗어나 정확한 답변을 드리기 어렵습니다.\n"
    "반려견 행동이나 견종에 대해 궁금한 점이 있으면 편하게 물어봐 주세요!"
)

# ── 프롬프트 ──────────────────────────────────────────────────────
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

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 반려견 행동 분석 전문가입니다.\n"
            "아래 문서 내용을 참고하여 사용자의 질문에 대해 정확하고 친절하게, 꼭 '한국어'로 "
            "답변해주세요.\n"
            "혹시, 문서 내용 중 일반적이지 않은 내용(예를 들어, 사용자의 질문에 없는 강아지의 "
            "이름이 포함)이 있다면 일반화해서 답변을 제공해주세요.\n"
            "질병·건강 이상이 의심되는 경우, 행동적 조언과 함께 수의사 진료를 받아보도록 "
            "안내해주세요.\n\n"
            "[문서 내용]\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

SCOPE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "다음 질문이 '개(반려견)'의 행동, 훈련, 견종, 돌봄에 관한 것인지 판단하세요. "
            "개와 관련되면 'YES', 그렇지 않으면(다른 동물, 일상 잡담 등) 'NO'만 한 단어로 "
            "출력하세요. 인사나 사용법 문의도 'YES'로 봅니다.",
        ),
        ("human", "{input}"),
    ]
)


# ── 리소스 로딩 (무거움 → app.py에서 캐싱) ─────────────────────────
def _load_embedding():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _build_retriever(db):
    """설정에 따라 dense / 하이브리드 / 리랭커 retriever를 구성(단계적 폴백)."""
    fetch_k = FETCH_K if (USE_HYBRID or USE_RERANKER) else RETRIEVE_K
    dense = db.as_retriever(search_type="similarity", search_kwargs={"k": fetch_k})
    base = dense

    # 하이브리드(BM25 + dense) — rank_bm25 미설치/실패 시 dense로 폴백
    if USE_HYBRID:
        try:
            from langchain_community.retrievers import BM25Retriever
            from langchain.retrievers import EnsembleRetriever

            all_docs = list(db.docstore._dict.values())
            if all_docs:
                bm25 = BM25Retriever.from_documents(all_docs)
                bm25.k = fetch_k
                base = EnsembleRetriever(
                    retrievers=[bm25, dense],
                    weights=[BM25_WEIGHT, 1.0 - BM25_WEIGHT],
                )
                logger.info("하이브리드 검색 활성화(BM25+dense, docs=%d)", len(all_docs))
        except Exception as e:  # noqa: BLE001
            logger.warning("하이브리드 검색 비활성화(폴백 dense): %s", e)

    # 리랭커(cross-encoder) — 모델 로드 실패 시 base로 폴백
    if USE_RERANKER:
        try:
            from langchain.retrievers import ContextualCompressionRetriever
            from langchain.retrievers.document_compressors import CrossEncoderReranker
            from langchain_community.cross_encoders import HuggingFaceCrossEncoder

            encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
            compressor = CrossEncoderReranker(model=encoder, top_n=RETRIEVE_K)
            base = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=base
            )
            logger.info("리랭커 활성화(%s, top_n=%d)", RERANKER_MODEL, RETRIEVE_K)
        except Exception as e:  # noqa: BLE001
            logger.warning("리랭커 비활성화(폴백): %s", e)

    return base


def build_llm(streaming: bool = False):
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.3,
        streaming=streaming,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


def build_resources():
    """앱에서 한 번만 호출해 캐싱할 무거운 리소스 묶음."""
    embedding = _load_embedding()
    db = FAISS.load_local(
        FAISS_DB_PATH, embedding, allow_dangerous_deserialization=True
    )
    retriever = _build_retriever(db)
    util_llm = build_llm(streaming=False)   # 질문 재작성 / 번역 / 범위판단용
    answer_llm = build_llm(streaming=True)   # 최종 답변 생성(스트리밍)
    return {
        "db": db,
        "retriever": retriever,
        "llm": util_llm,
        "contextualize_chain": CONTEXTUALIZE_PROMPT | util_llm | StrOutputParser(),
        "scope_chain": SCOPE_PROMPT | util_llm | StrOutputParser(),
        "answer_chain": create_stuff_documents_chain(answer_llm, ANSWER_PROMPT),
    }


# ── 번역 유틸 (교차언어 검색용) ───────────────────────────────────
def gpt_translate_ko_to_en(text, llm):
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


# ── T2-3: 가드레일 / 면책 ─────────────────────────────────────────
def needs_health_disclaimer(text):
    """질문에 건강 의심 키워드가 있으면 True → 수의사 상담 면책 고지 추가."""
    if not text:
        return False
    return any(kw in text for kw in HEALTH_KEYWORDS)


def is_in_scope(question, resources):
    """질문이 '개' 관련 범위인지 LLM으로 판단(범위 외면 False). 실패 시 통과(True)."""
    if not ENABLE_OOS_GUARD:
        return True
    try:
        verdict = resources["scope_chain"].invoke({"input": question})
        return "no" not in verdict.strip().lower()[:4]
    except Exception as e:  # noqa: BLE001
        logger.warning("범위 판단 실패(통과 처리): %s", e)
        return True


# ── 핵심 파이프라인 ───────────────────────────────────────────────
def to_lc_history(messages):
    """app의 [{'role','content'}] → langchain 메시지 리스트(최근 N개)."""
    history = []
    for m in messages[-HISTORY_MAX_MESSAGES:]:
        content = m.get("content", "")
        if m.get("role") == "user":
            history.append(HumanMessage(content=content))
        else:
            history.append(AIMessage(content=content.split(SOURCE_MARKER)[0]))
    return history


def contextualize_question(question, chat_history, resources):
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
    """교차언어 검색(T1-4). 한국어 1차 검색 후 최상위 출처가 영어(kinship)이면
    영어 질의로 추가 검색해 병합. 답변 생성은 호출부에서 1회만."""
    retriever = resources["retriever"]
    docs = retriever.invoke(query)
    if docs and docs[0].metadata.get("source", "").startswith(KINSHIP_PREFIX):
        query_en = gpt_translate_ko_to_en(query, resources["llm"])
        query_en = title_case_excluding_prepositions(query_en)
        en_docs = retriever.invoke(query_en)
        docs = _dedup_docs(en_docs + docs)
    return docs[:RETRIEVE_K]


def _static_stream(text):
    """범위 외 등 고정 응답을 스트림과 동일한 제너레이터 형태로 반환."""
    yield text


def answer_stream(question, chat_history, resources):
    """(출처 문서 리스트, 답변 토큰 제너레이터)를 반환.

    범위 외(개 이외) 질문은 검색/LLM 없이 고정 안내를 반환(T2-3).
    그 외에는 history-aware 검색 후 답변 LLM 1회(스트리밍)로 생성한다.
    """
    if not is_in_scope(question, resources):
        return [], _static_stream(OOS_RESPONSE)

    standalone_q = contextualize_question(question, chat_history, resources)
    docs = adaptive_retrieve(standalone_q, resources)
    stream = resources["answer_chain"].stream(
        {"context": docs, "chat_history": chat_history, "input": question}
    )
    return docs, stream


def answer_question(question, chat_history=None, resources=None):
    """비스트리밍 편의 함수(스크립트/평가용). dict(answer, source_documents) 반환."""
    resources = resources or build_resources()
    chat_history = chat_history or []
    docs, stream = answer_stream(question, chat_history, resources)
    text = "".join(stream)
    if needs_health_disclaimer(question):
        text += HEALTH_DISCLAIMER
    return {"answer": text, "source_documents": docs}


if __name__ == "__main__":
    res = build_resources()

    out = answer_question("왜 내 개가 바닥을 핥을까?", resources=res)
    print("답변:\n", out["answer"])
    print("\n참고 문서:")
    for i, d in enumerate(out["source_documents"], 1):
        print(f"{i}. {d.metadata.get('title') or d.metadata.get('source', 'URL 없음')}")

    out2 = answer_question("페럿이 우울해 보여요", resources=res)
    print("\n[범위 외] 답변:\n", out2["answer"])
