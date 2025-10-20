import os
import yt_dlp
import whisper
from llm.ksi.call_llm import call_llm
import json
import re

# 유튜브 플레이 리스트
youtube_list_url = "https://youtube.com/playlist?list=PLVh3TM0B0WtksY4ZQVNdD0aTG1Qm1mQNM&si=D9uDiJKwaQQb-GDi"
ffmpeg_path = "C:\\prg\\ffmpeg\\ffmpeg-master-latest-win64-gpl-shared\\bin"

AUDIO_DIR = "./data/ksi/Audio"  # 오디오 추출 저장 딕렉토리
INFO_JSON_PATH = os.path.join(AUDIO_DIR, "audio_info.json")
TEXT_DIR = "./data/ksi/Text"  # 자막 저장 디렉토리
JSON_PATH = "./data/ksi/Json/dog_documents.json"  # json 저장 디렉토리


# 기존 audio_info.json 불러오기
def load_existing_info():
    if os.path.exists(INFO_JSON_PATH):
        with open(INFO_JSON_PATH, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []


# audio_info.json에 저장
def save_info(info_list):
    with open(INFO_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(info_list, f, ensure_ascii=False, indent=2)


# mp3 오디오 추출 메소드
def preparing_mp3():

    # 디렉토리 확인후 없으면 생성
    os.makedirs(AUDIO_DIR, exist_ok=True)

    # 기존 정보 불러오기
    existing_entries = load_existing_info()
    existing_urls = {entry["url"] for entry in existing_entries if entry.get("url")}

    # ydl_opts 설정
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(AUDIO_DIR, "%(title)s.%(ext)s"),
        "ffmpeg_location": ffmpeg_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }
    # 재생목록 오디오 다운로드
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_list_url, download=True)

    new_entries = []
    for entry in info.get("entries", []):
        if entry is None:
            continue

        url = entry.get("webpage_url")
        title = entry.get("title")
        file_path = os.path.join(AUDIO_DIR, f"{title}.mp3").replace("\\", "/")
        print(f"\nMP3 오디오 추출 완료 → 저장 위치: {AUDIO_DIR}")

        if url in existing_urls:
            continue  # 이미 저장된 항목 넘기기

        new_entries.append({"title": title, "url": url, "file_path": file_path})

    # 누적해서 저장
    all_entries = existing_entries + new_entries
    save_info(all_entries)


# mp3 오디어 whisper txt 텍스트 변환
def preparing_txt():

    # 디렉토리 확인후 없으면 생성성
    os.makedirs(TEXT_DIR, exist_ok=True)

    # ffmpeg 경로 명시
    os.environ["PATH"] += os.pathsep + rf"{ffmpeg_path}"

    # whisper 모델 설정
    model = whisper.load_model("small").to("cuda")

    for file in os.listdir(AUDIO_DIR):
        if file.endswith(".mp3"):
            audio_path = os.path.join(AUDIO_DIR, file)
            output = os.path.join(TEXT_DIR, file.replace(".mp3", ".txt"))

            # 중복 여부 확인
            if os.path.exists(output):
                print(f"이미 존재함, 건너뜀: {file}")
                continue

            print(f"처리 시작: {file}")

            try:
                # 오디오 -> 텍스트 처리
                result = model.transcribe(audio_path, language="ko", fp16=True)
                text = result["text"]

                # 자막 TEXT 저장
                with open(output, "w", encoding="utf-8") as f:
                    f.write(text)

                print(f"처리 완료: {file}")

            except Exception as e:
                print(f"에러 : {file}")


def text_append_url():
    # audio_info.json 로드

    with open(INFO_JSON_PATH, "r", encoding="utf-8") as f:
        audio_info = json.load(f)

    # title → url 매핑 딕셔너리 생성
    title_to_url = {
        entry["title"]: entry["url"]
        for entry in audio_info
        if "title" in entry and "url" in entry
    }

    updated_count = 0
    skipped_count = 0

    for filename in os.listdir(TEXT_DIR):
        if not filename.endswith(".txt"):
            continue

        title = os.path.splitext(filename)[0]
        url = title_to_url.get(title)

        if not url:
            skipped_count += 1
            continue

        file_path = os.path.join(TEXT_DIR, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if f"유튜브 URL : {url}" in content:
            continue  # 이미 있음 → 건너뜀

        with open(file_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n유튜브 URL : {url}")

        updated_count += 1

    print(f" URL 추가 완료: {updated_count}개 파일")
    print(f" 매칭되는 title이 없어 건너뛴 파일: {skipped_count}개")


# txt 텍스트 기반 openapi 활용 json 형식 변환
def create_json_with_llm(dog_text):
    system_prompt = """
    당신은 견종 분석 유튜브 영상 자막 분석기입니다. 아래 자막 내용을 바탕으로, 해당 영상에 대한 정보를 구조화하는 데 도움을 줍니다.
    
    0. 존재하는 견종인지 검증하고, 존재하지 않는 견종으로 판단되면 해당 글자와 유사하며 특징이 비슷한 견종 텍스트로 대체한다. 문장이 자연스럽게 만들어지게 오타나 자연스럽지 않은 문자를 자연스럽게 수정한다. 예시) 그레이트덴 -> 그레이트 데인
    1. title 에는 해당 자막이 어떤 견종에 대해 이야기하는 것인지 입력한다. 예시) title : 리트리버
    2. source에는 텍스트 파일에 명시되어있는 유튜브 주소를 입력합니다. 명시되어 있지 않다면 "None"을 입력합니다.
    3. page_content는 유튜브 링크를 제외한 자막 텍스트의 내용을 견종의 정보를 뽑아내 자세하고 상세하게 출력하며, 너무 짧게 압축하거나 생략하면 안된다. 오타를 수정하고, 필요없는 특수문자를 제거하고, 견종과 관계없는 내용은 제거한 견종 관련 텍스트만 추출합니다. 견종과 관련된 내용은 부분 생략이 아니라 모두 포함해야한다.
    4. "type": "Document" 는 반드시 포함해야 합니다.

    밑의 json 형식으로 출력하고 반드시 지켜져야 한다.
    {
      "metadata": {
        "title": "",
        "source": ""
      },
      "page_content": "",
      "type": "Document"
    }

    ##중요## 밑의 형식처럼 출력되지 않게 주의 한다.
    {
      "metadata": {
        "title": "",
        "source": ""
      },
      "page_content": "" 
      },
      "type": "Document"
    }

    그래도 문제가 생기면 '{', '}' 문자가 각각 2개만 나온다고 생각해라

    """

    user_prompt = f"""
    아래는 해당 자막에 대한 텍스트입니다:
    {dog_text}
    """

    response = call_llm(system_prompt, user_prompt)
    print("\n" + response)

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            return json.loads(match.group())
        else:
            raise ValueError("GPT 응답에서 JSON을 추출하지 못했습니다.")


def preparing_json():
    os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)
    results = []

    for file in os.listdir(TEXT_DIR):
        if file.endswith(".txt"):
            file_path = os.path.join(TEXT_DIR, file)

            with open(file_path, "r", encoding="utf-8") as f:
                dog_text = f.read()

            try:
                print(f"처리 시작: {file}")
                result = create_json_with_llm(dog_text)

                results.append(result)
                print(f"성공: {file}")
            except Exception as e:
                print(f"처리 실패: {file} → {e}")
                continue  # 실패했으면 이 파일은 건너뜀

    # 모든 result를 저장
    if results:
        with open(JSON_PATH, "w", encoding="utf-8") as out:
            json.dump(results, out, ensure_ascii=False, indent=2)
        print(f"\n전체 처리 완료: {len(results)}개 문서 저장됨 → {JSON_PATH}")
    else:
        print("\n저장할 문서가 없습니다.")
