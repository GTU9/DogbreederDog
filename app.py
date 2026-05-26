# 🐶 반려견 챗봇 UI with st.chat_input() + 강아지 말풍선 유지
import streamlit as st
import streamlit.components.v1 as components
from py_file.QA_bot import (
    qa_chain,
    gpt_translate_ko_to_en,
    title_case_excluding_prepositions,
)
import base64
import random
import logging
import os

# 로거 설정 (우리 앱 로그만)
os.makedirs("data/logs", exist_ok=True)
logger = logging.getLogger("dogbreeder")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.FileHandler("data/logs/chat.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(handler)
import re
import html as html_mod

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
        role = msg["role"]
        content = msg["content"]

        # surrogate 유니코드 제거
        try:
            content = content.encode("utf-16", "surrogatepass").decode("utf-16")
        except:
            import re

            content = re.sub(r"[\ud800-\udfff]", "", content)

        escaped_content = html_mod.escape(content)

        if role == "user":
            profile_b64 = user_profile_b64
            overlay_b64 = user_dog_b64
            bubble_class = "right"
            chat_line_class = "chat-line user"
            character_position = "character user"
            html_block = f"""
            <div style="display: flex; justify-content: flex-end; align-items: flex-start; margin-bottom: 30px;">
                <div class="{chat_line_class}">
                    <img src="data:image/png;base64,{overlay_b64}" class="{character_position}">
                    <div class="bubble {bubble_class}">{escaped_content}</div>
                </div>
                <img src="data:image/png;base64,{profile_b64}"
                     style="width: 50px; height: 50px; border-radius: 50%; object-fit: cover; margin-left: 10px; flex-shrink: 0;">
            </div>
            """
        else:
            profile_b64 = random.choice(bot_profile_b64s)
            overlay_b64 = ai_dog_b64
            bubble_class = "left"
            chat_line_class = "chat-line"
            character_position = "character"
            html_block = f"""
            <div style="display: flex; align-items: flex-start; margin-bottom: 30px;">
                <img src="data:image/png;base64,{profile_b64}"
                     style="width: 50px; height: 50px; border-radius: 50%; object-fit: cover; margin-right: 10px; flex-shrink: 0;">
                <div class="{chat_line_class}">
                    <img src="data:image/png;base64,{overlay_b64}" class="{character_position}">
                    <div class="bubble {bubble_class}">{escaped_content}</div>
                </div>
            </div>
            """

        st.markdown(html_block, unsafe_allow_html=True)


# 입력 받기
def escape_single_tilde(text):
    if not isinstance(text, str):
        return text
    # ~텍스트~ => &#126;텍스트&#126;
    return re.sub(r"~(.*?)~", r"&#126;\1&#126;", text)


user_input = st.chat_input("질문을 입력하세요...")

if user_input:
    logger.info(f"질문: {user_input}")
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("답변 생성 중..."):
        result_temp_raw = qa_chain(user_input)

        sources = result_temp_raw.get("source_documents", [])
        first_source = sources[0].metadata.get("source", "") if sources else ""

        if first_source.startswith("https://www.kinship.com"):
            query_en = gpt_translate_ko_to_en(user_input)
            query_title_case = title_case_excluding_prepositions(query_en)
            result = qa_chain(query_title_case)
        else:
            result = result_temp_raw

        # ✅ 여기에서 result["result"]에 대해 escape 적용
        if "result" in result:
            result["result"] = escape_single_tilde(result["result"])

        answer = result["result"]
        logger.info(f"답변: {answer[:100]}{'...' if len(answer) > 100 else ''}")

        # 참고 문서 정리
        source_info = ""
        unique_sources = set()
        source_list = []

        for doc in result.get("source_documents", []):
            src = doc.metadata.get("source", "").strip()
            if src and src.lower() != "none" and src not in unique_sources:
                unique_sources.add(src)
                source_list.append(src)

        if source_list:
            source_info += "\n\n📚 **참고 문서:**\n"
            for i, src in enumerate(source_list, 1):
                source_info += f"{i}. {src}\n"

        answer_with_sources = answer + source_info

    st.session_state.messages.append({"role": "bot", "content": answer_with_sources})
    st.rerun()
