# 🐶 반려견 챗봇 UI with st.chat_input() + 강아지 말풍선 유지
import streamlit as st
import streamlit.components.v1 as components
from py_file.QA_bot import (
    build_resources,
    answer_stream,
    to_lc_history,
    needs_health_disclaimer,
    HEALTH_DISCLAIMER,
)
import base64
import random
import logging
import os
import time
from streamlit_js_eval import streamlit_js_eval
import re
import html as html_mod
from urllib.parse import urlparse

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
# 축소본(_s) 사용 — 원본(1~1.7MB)을 base64로 매 말풍선에 내장하면 전송량이 커져 느려짐
bot_profile_b64s = [
    img_to_b64(".streamlit/data/png/bot4_s.png"),
    img_to_b64(".streamlit/data/png/bot5_s.png"),
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
    """surrogate 유니코드 제거 후 HTML 이스케이프 + 말풍선용 마크다운 경량 변환.

    Streamlit 마크다운 파서가 빈 줄에서 HTML 블록을 끊고 목록에 자체 여백을 더해
    말풍선 안 여백이 이중으로 벌어지므로, 여기서 직접 변환한다:
    굵게(**x**) → <b>, 과한 빈 줄 압축, 줄바꿈 → <br> (HTML 블록 유지).
    """
    try:
        content = content.encode("utf-16", "surrogatepass").decode("utf-16")
    except Exception:
        content = re.sub(r"[\ud800-\udfff]", "", content)
    esc = html_mod.escape(content)
    esc = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", esc, flags=re.S)  # 굵게
    esc = re.sub(r"\n{2,}", "\n", esc)  # 목록 사이 빈 줄 압축(이중 여백 방지)
    return esc.replace("\n", "<br>")


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


def build_sources(docs):
    """문서 메타데이터에서 (제목, URL) 출처 목록을 구성(T2-4).

    같은 제목은 1개로 묶는다(동일 견종 유튜브 영상 등 제목 중복 방지).
    URL 없는 항목이 먼저 왔어도 뒤에 URL 있는 동일 제목이 오면 URL을 채워준다.
    병합 후에도 URL이 없는 출처는 표시하지 않는다(링크 불가 문서 제외).
    """
    by_label = {}  # 표시 제목(소문자) -> [원래 제목, url]
    for doc in docs:
        src = (doc.metadata.get("source") or "").strip()
        title = (doc.metadata.get("title") or "").strip()
        has_url = bool(src) and src.lower() != "none"
        label = title or (src if has_url else "")
        if not label:
            continue
        key = label.lower()
        if key not in by_label:
            by_label[key] = [label, src if has_url else ""]
        elif has_url and not by_label[key][1]:
            by_label[key][1] = src
    return [(label, url) for label, url in by_label.values() if url]


def _source_kind(url):
    """URL 도메인으로 출처 유형 라벨 결정."""
    if not url:
        return "내부 문서"
    host = urlparse(url).netloc.lower()
    if "youtube.com" in host or "youtu.be" in host:
        return "YouTube 견종백과"
    if "kinship.com" in host:
        return "kinship.com"
    return host or "외부 링크"


def render_sources(sources):
    """출처를 '제목 (유형)' 클릭 링크로 렌더(T2-4). URL 없으면 텍스트로 표시."""
    if not sources:
        return
    parts = []
    for title, url in sources:
        label = (title or "출처").replace("[", "(").replace("]", ")")
        text = f"{label} ({_source_kind(url)})"
        parts.append(f"[{text}]({url})" if url else text)
    st.markdown("📚 **참고 문서:** " + "  ·  ".join(parts))


# 대화 기록 저장소 — 서버 프로세스 메모리에만 보관(DB/파일 기록 없음).
# 세션 ID는 브라우저 sessionStorage에 둔다 → F5 새로고침에는 유지되고,
# 탭/브라우저를 닫으면 sessionStorage가 비워져 기록도 완전 휘발된다.
@st.cache_resource
def _conversation_store():
    return {}


store = _conversation_store()
sid = streamlit_js_eval(
    js_expressions=(
        "sessionStorage.getItem('db_sid') || (function(){"
        "var v=Math.random().toString(36).slice(2)+Date.now().toString(36);"
        "sessionStorage.setItem('db_sid', v); return v;})()"
    ),
    key="sid_eval",
)
if not sid:
    st.stop()  # 최초 1회 JS 왕복 대기(응답 오면 자동 재실행)
if sid not in store:
    # 오래된 세션 정리(메모리 보호): 100개 초과 시 가장 오래된 것부터 제거
    while len(store) >= 100:
        store.pop(next(iter(store)))
    store[sid] = {"conversations": [{"title": "새 대화", "messages": []}], "current": 0}

state = store[sid]
conversations = state["conversations"]
current = state["current"]
messages = conversations[current]["messages"]

# 사이드바
st.sidebar.image(".streamlit/data/png/Logo.png", use_container_width=True)
st.sidebar.title("🐶 반려견 채팅")
st.sidebar.write("반려견 전문가 채팅입니다. 궁금한 점을 입력해보세요!")

if st.sidebar.button("새 대화", use_container_width=True):
    # 현재 대화가 비어 있으면 그대로 재사용(빈 대화 중복 생성 방지)
    if conversations[current]["messages"]:
        conversations.append({"title": "새 대화", "messages": []})
        state["current"] = len(conversations) - 1
    st.rerun()

# 질문이 시작된 대화만 기록 목록에 표시(빈 새 대화는 숨김)
history = [(i, c) for i, c in enumerate(conversations) if c["messages"]]
if history:
    st.sidebar.markdown("#### 대화 기록")
    for i, conv in history:
        # 현재 보고 있는 대화는 비활성(회색)으로 구분
        if st.sidebar.button(
            conv["title"], key=f"conv-{i}", use_container_width=True,
            disabled=(i == current),
        ):
            state["current"] = i
            st.rerun()

st.title("🐶💬개 잘키우개")

# 메시지 출력 (현재 선택된 대화)
with st.container():
    for msg in messages:
        if msg["role"] == "user":
            st.markdown(user_bubble_html(msg["content"]), unsafe_allow_html=True)
        else:
            st.markdown(bot_bubble_html(msg["content"]), unsafe_allow_html=True)
            render_sources(msg.get("sources"))

# 답변 완료(rerun) 후 마지막 말풍선으로 자동 스크롤.
# 스크립트에 메시지 수를 넣어 내용을 매번 다르게 함(동일 HTML이면 iframe이 재실행되지 않음).
if messages:
    components.html(
        f"""
        <script>
        /* n={len(messages)} */
        const lines = window.parent.document.querySelectorAll('.chat-line');
        if (lines.length) {{
            lines[lines.length - 1].scrollIntoView({{behavior: 'smooth', block: 'end'}});
        }}
        </script>
        """,
        height=0,
    )


# 입력 받기
user_input = st.chat_input("질문을 입력하세요...")

if user_input:
    logger.info(f"질문: {user_input}")

    # 대화 맥락(현재 입력은 제외하고 직전까지) → langchain 메시지
    chat_history = to_lc_history(messages)

    # 사용자 메시지 저장 + 즉시 렌더(스트리밍 중에도 보이도록)
    messages.append({"role": "user", "content": user_input})
    # 첫 질문이면 사이드바 대화 제목으로 사용
    if conversations[current]["title"] == "새 대화":
        conversations[current]["title"] = user_input[:18] + ("…" if len(user_input) > 18 else "")
    st.markdown(user_bubble_html(user_input), unsafe_allow_html=True)

    # 1) 검색(필요 시 교차언어) — 답변 LLM 호출은 이후 1회뿐
    with st.spinner("관련 문서를 찾는 중..."):
        docs, stream = answer_stream(user_input, chat_history, resources)

    # 2) 답변 스트리밍(T1-5) — 매 청크 풀 HTML(이미지 포함) 재전송은 느리므로
    #    스트리밍 중에는 이미지 없는 경량 말풍선 + 0.1초 스로틀로 갱신하고,
    #    완료 후 한 번만 프로필/캐릭터가 포함된 풀 말풍선으로 렌더한다.
    placeholder = st.empty()
    bot_profile = random.choice(bot_profile_b64s)
    acc = ""
    last_draw = 0.0
    for chunk in stream:
        acc += chunk
        now = time.time()
        if now - last_draw >= 0.1:
            light = _clean(escape_single_tilde(acc))
            placeholder.markdown(
                f'<div class="chat-line"><div class="bubble left">{light}</div></div>',
                unsafe_allow_html=True,
            )
            last_draw = now

    answer = escape_single_tilde(acc)

    # T2-3: 건강 의심 질문이면 수의사 상담 면책 고지 추가
    if needs_health_disclaimer(user_input):
        answer += HEALTH_DISCLAIMER

    # 스트리밍 종료 → 프로필/캐릭터 포함 풀 말풍선으로 1회 최종 렌더
    placeholder.markdown(
        bot_bubble_html(answer, profile_b64=bot_profile), unsafe_allow_html=True
    )

    logger.info(f"답변: {answer[:100]}{'...' if len(answer) > 100 else ''}")

    # 3) 출처(제목 링크, T2-4) 렌더 후 최종 메시지 저장
    sources = build_sources(docs)
    render_sources(sources)
    messages.append({"role": "bot", "content": answer, "sources": sources})
    st.rerun()
