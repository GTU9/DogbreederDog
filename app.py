# 🐶 반려견 챗봇 UI with st.chat_input() + 강아지 말풍선 유지
import streamlit as st
import streamlit.components.v1 as components
from py_file.QA_bot import (
    build_resources,
    answer_stream,
    to_lc_history,
)
import base64
import random
import logging
import os
import re
import html as html_mod

# 로거 설정 (우리 앱 로그만)
os.makedirs("data/logs", exist_ok=True)
logger = logging.getLogger("dogbreeder")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.FileHandler("data/logs/chat.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(handler)

# 페이지 설정
st.set_page_config(page_title="🐾개잘키우개🐾", layout="wide")

# 스타일
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nanum+Gothic:wght@400;700&display=swap');

    /* ── 전체 배경 ── */
    .stApp {
        background-color: #fff8ed;
        color: #333333;
    }

    /* ── 채팅 콘텐츠 최대 너비 제한 (가독성) ── */
    [data-testid="stMainBlockContainer"] {
        max-width: 860px !important;
        margin: 0 auto !important;
        padding-left: 24px !important;
        padding-right: 24px !important;
    }

    /* ── 채팅 타이틀 ── */
    h1 {
        font-family: 'Nanum Gothic', sans-serif !important;
        font-size: 1.6rem !important;
        color: #5c3d2e !important;
        padding-bottom: 12px !important;
        border-bottom: 2px solid #e8d8c4 !important;
        margin-bottom: 24px !important;
    }

    /* ── 말풍선 공통 ── */
    .bubble {
        display: inline-block;
        padding: 13px 18px;
        margin-bottom: 6px;
        border-radius: 18px;
        font-size: 15px;
        line-height: 1.7;
        max-width: 72%;
        white-space: pre-wrap;
        word-break: break-word;
        font-family: 'Nanum Gothic', sans-serif;
        position: relative;
        box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    }

    /* ── 봇 말풍선 (좌) ── */
    .left {
        background-color: #f5f0eb;
        color: #2d2d2d;
        margin-right: auto;
        border-bottom-left-radius: 4px;
        border: 1px solid #e8ddd4;
    }

    /* ── 유저 말풍선 (우) ── */
    .right {
        background-color: #a88f7f;
        color: #ffffff;
        margin-left: auto;
        border-bottom-right-radius: 4px;
        box-shadow: 0 2px 10px rgba(168,143,127,0.35);
    }

    /* ── 강아지 오버레이 캐릭터 ── */
    .character {
        width: 46px;
        position: absolute;
        top: -28px;
        left: -8px;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.15));
    }
    .character.user {
        left: auto;
        right: -8px;
    }

    /* ── 채팅 행 래퍼 ── */
    .chat-line {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        position: relative;
        margin-bottom: 36px;
    }
    .chat-line.user {
        align-items: flex-end;
    }

    /* ── 링크 색상 (참고 문서) ── */
    .bubble a {
        color: #7c5c4a;
        text-decoration: underline;
    }
    .right a {
        color: #f0e0d6;
    }

    /* ── 입력창 ── */
    [data-testid="stChatInput"] textarea {
        background-color: #fffaf3 !important;
        border-radius: 24px !important;
        border: 1.5px solid #d9c4b0 !important;
        font-family: 'Nanum Gothic', sans-serif !important;
        font-size: 15px !important;
        color: #333 !important;
        padding: 12px 20px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
        transition: border-color 0.2s !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #a88f7f !important;
        outline: none !important;
    }

    /* ── 사이드바 ── */
    [data-testid="stSidebar"] {
        background-color: #fdf3e3 !important;
        border-right: 1px solid #e8d8c4 !important;
    }
    [data-testid="stSidebar"] h2 {
        color: #5c3d2e !important;
        font-family: 'Nanum Gothic', sans-serif !important;
    }
    [data-testid="stSidebar"] p {
        color: #7a5c48 !important;
        font-size: 13px !important;
        line-height: 1.6 !important;
    }

    /* ── 스크롤바 ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #fff8ed; }
    ::-webkit-scrollbar-thumb { background: #d9c4b0; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #a88f7f; }
    </style>
""",
    unsafe_allow_html=True,
)


# 이미지 인코딩 함수
def img_to_b64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return ""


# 이미지 준비
ai_dog_b64 = img_to_b64(".streamlit/data/png/보잉.png")
user_dog_b64 = img_to_b64(".streamlit/data/png/오앙.png")
user_profile_b64 = img_to_b64(".streamlit/data/png/user.png")
bot_profile_b64s = [
    img_to_b64(".streamlit/data/png/bot4.png"),
    img_to_b64(".streamlit/data/png/bot5.png"),
]


# T1-1: 무거운 임베딩/FAISS/LLM 리소스를 1회만 로드하고 캐싱
# 캐시 TTL: 유휴 시 메모리 해제(NAS RAM 절약). 기본 1시간, 환경변수 RESOURCE_CACHE_TTL로 조정.
#   예) "1h", "30m", "2h30m" / "none"·"0"·빈값 → 만료 없이 영구 상주(반응성 우선).
def _resolve_cache_ttl():
    raw = os.getenv("RESOURCE_CACHE_TTL", "1h").strip().lower()
    if raw in ("", "0", "none", "off", "false"):
        return None
    return raw


@st.cache_resource(ttl=_resolve_cache_ttl(), show_spinner="🐶 모델을 불러오는 중...")
def get_resources():
    return build_resources()


resources = get_resources()


# ── 렌더링 헬퍼 ───────────────────────────────────────────────────
def _clean(content):
    """surrogate 유니코드 제거 후 HTML 이스케이프."""
    try:
        content = content.encode("utf-16", "surrogatepass").decode("utf-16")
    except Exception:
        content = re.sub(r"[\ud800-\udfff]", "", content)
    return html_mod.escape(content)


def escape_single_tilde(text):
    if not isinstance(text, str):
        return text
    # ~텍스트~ => &#126;텍스트&#126;
    return re.sub(r"~(.*?)~", r"&#126;\1&#126;", text)


def user_bubble_html(content):
    esc = _clean(content)
    return f"""
    <div style="display: flex; justify-content: flex-end; align-items: flex-start; margin-bottom: 30px;">
        <div class="chat-line user">
            <img src="data:image/png;base64,{user_dog_b64}" class="character user">
            <div class="bubble right">{esc}</div>
        </div>
        <img src="data:image/png;base64,{user_profile_b64}"
             style="width: 50px; height: 50px; border-radius: 50%; object-fit: cover; margin-left: 10px; flex-shrink: 0;">
    </div>
    """


def bot_bubble_html(content, profile_b64=None):
    esc = _clean(content)
    profile_b64 = profile_b64 if profile_b64 is not None else random.choice(bot_profile_b64s)
    return f"""
    <div style="display: flex; align-items: flex-start; margin-bottom: 30px;">
        <img src="data:image/png;base64,{profile_b64}"
             style="width: 50px; height: 50px; border-radius: 50%; object-fit: cover; margin-right: 10px; flex-shrink: 0;">
        <div class="chat-line">
            <img src="data:image/png;base64,{ai_dog_b64}" class="character">
            <div class="bubble left">{esc}</div>
        </div>
    </div>
    """


def format_sources(docs):
    """출처 문서를 중복 제거해 표시용 마크다운 블록으로 구성."""
    unique_sources, source_list = set(), []
    for doc in docs:
        src = doc.metadata.get("source", "").strip()
        if src and src.lower() != "none" and src not in unique_sources:
            unique_sources.add(src)
            source_list.append(src)

    if not source_list:
        return ""
    block = "\n\n📚 **참고 문서:**\n"
    for i, src in enumerate(source_list, 1):
        block += f"{i}. {src}\n"
    return block


# 세션 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 사이드바
st.sidebar.image(".streamlit/data/png/Logo.png", use_container_width=True)
st.sidebar.title("🐶 반려견 채팅")
st.sidebar.write("반려견 전문가 채팅입니다. 궁금한 점을 입력해보세요!")

st.title("🐶💬개 잘키우개")

# 메시지 출력
with st.container():
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(user_bubble_html(msg["content"]), unsafe_allow_html=True)
        else:
            st.markdown(bot_bubble_html(msg["content"]), unsafe_allow_html=True)


# 입력 받기
user_input = st.chat_input("질문을 입력하세요...")

if user_input:
    logger.info(f"질문: {user_input}")

    # 대화 맥락(현재 입력은 제외하고 직전까지) → langchain 메시지
    chat_history = to_lc_history(st.session_state.messages)

    # 사용자 메시지 저장 + 즉시 렌더(스트리밍 중에도 보이도록)
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(user_bubble_html(user_input), unsafe_allow_html=True)

    # 1) 검색(필요 시 교차언어) — 답변 LLM 호출은 이후 1회뿐
    with st.spinner("관련 문서를 찾는 중..."):
        docs, stream = answer_stream(user_input, chat_history, resources)

    # 2) 답변 스트리밍(커스텀 말풍선 유지, T1-5)
    placeholder = st.empty()
    bot_profile = random.choice(bot_profile_b64s)
    acc = ""
    for chunk in stream:
        acc += chunk
        placeholder.markdown(
            bot_bubble_html(escape_single_tilde(acc), profile_b64=bot_profile),
            unsafe_allow_html=True,
        )

    answer = escape_single_tilde(acc)
    logger.info(f"답변: {answer[:100]}{'...' if len(answer) > 100 else ''}")

    # 3) 출처 정리 후 최종 메시지 저장
    answer_with_sources = answer + format_sources(docs)
    st.session_state.messages.append({"role": "bot", "content": answer_with_sources})
    st.rerun()
