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
import re

# 페이지 설정
st.set_page_config(page_title="🐾개잘키우개🐾", layout="wide")

# 스타일
st.markdown(
    """
    <style>
    .stApp {
        background-color: #fff8ed;
        color: #333333;
    }
    .bubble {
        display: inline-block;
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 20px;
        font-size: 16px;
        max-width: fit-content;
        white-space: pre-wrap;
        word-break: break-word;
        font-family: 'Nanum Gothic', sans-serif;
        position: relative;
    }
    .left {
        background-color: #f5f0eb;
        color: #000000;
        margin-right: auto;
    }
    .right {
        background-color: #a88f7f;
        color: white;
        margin-left: auto;
    }
    .character {
        width: 50px;
        position: absolute;
        top: -30px;
        left: -10px;
    }
    .character.user {
        left: auto;
        right: -10px;
    }
    .chat-line {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        position: relative;
        margin-bottom: 40px;
    }
    .chat-line.user {
        align-items: flex-end;
    }
    input[type="text"] {
        background-color: #fffaf3;
        color: #333;
        border-radius: 18px;
        padding: 10px;
        font-size: 16px;
    }
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

# 배경음악
file_path = ".streamlit/data/mp3/bgm.mp3"


def get_audio_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


st.markdown(
    f"""
    <audio autoplay loop>
        <source src="data:audio/mp3;base64,{get_audio_base64(file_path)}" type="audio/mp3">
    </audio>
""",
    unsafe_allow_html=True,
)

# 세션 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 사이드바
st.sidebar.image(".streamlit/data/png/logo.png", use_container_width=True)
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

        if role == "user":
            profile_b64 = user_profile_b64
            overlay_b64 = user_dog_b64
            bubble_class = "right"
            chat_line_class = "chat-line user"
            profile_margin = "margin-left: 10px;"
            character_position = "character user"
            direction = "flex-direction: row-reverse;"
        else:
            profile_b64 = random.choice(bot_profile_b64s)
            overlay_b64 = ai_dog_b64
            bubble_class = "left"
            chat_line_class = "chat-line"
            profile_margin = "margin-right: 10px;"
            character_position = "character"
            direction = ""

        st.markdown(
            f"""
            <div style="display: flex; align-items: flex-start; margin-bottom: 30px; {direction}">
                <img src="data:image/png;base64,{profile_b64}"
                     style="width: 50px; height: 50px; border-radius: 50%; object-fit: cover; {profile_margin}">
                <div class="{chat_line_class}">
                    <img src="data:image/png;base64,{overlay_b64}" class="{character_position}">
                    <div class="bubble {bubble_class}">{content}</div>
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )


# 입력 받기
def escape_single_tilde(text):
    if not isinstance(text, str):
        return text
    # ~텍스트~ => &#126;텍스트&#126;
    return re.sub(r"~(.*?)~", r"&#126;\1&#126;", text)


user_input = st.chat_input("질문을 입력하세요...")

if user_input:
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
