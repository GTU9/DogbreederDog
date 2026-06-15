"""QA_bot 파이프라인 회귀 테스트 (모델/OpenAI 불필요 — 가짜 체인 사용).

실행:
    testvenv/Scripts/python.exe -m pytest tests/test_qa_pipeline.py
    또는 testvenv/Scripts/python.exe tests/test_qa_pipeline.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import py_file.QA_bot as q
from langchain_core.messages import HumanMessage, AIMessage


class _FakeContextualize:
    """후속 질문을 '개 맥락이 들어간 독립형 질문'으로 재작성한다고 가정."""

    def invoke(self, d):
        return "강아지 초콜릿 중독 증상 중 가장 위험한 것은 무엇인가?"


class _ScopeByKeyword:
    """범위 판단: 텍스트에 개/강아지 단서가 있으면 YES, 없으면 NO."""

    def invoke(self, d):
        t = d["input"]
        return "YES" if ("개" in t or "강아지" in t) else "NO"


class _FakeRetriever:
    def invoke(self, _):
        return []


class _FakeAnswer:
    def stream(self, _):
        yield "위험 증상은 발작과 부정맥입니다."


def _resources():
    return {
        "contextualize_chain": _FakeContextualize(),
        "scope_chain": _ScopeByKeyword(),
        "retriever": _FakeRetriever(),
        "answer_chain": _FakeAnswer(),
        "llm": None,
    }


def test_followup_not_rejected_by_oos_guard():
    """회귀: OOS 가드가 대화 메모리를 깨면 안 된다.

    후속 질문 '그 중에 제일 위험한 건?'은 단독으로 보면 개 단서가 없지만,
    contextualize → is_in_scope 순서이므로 거절되지 않아야 한다.
    """
    q.ENABLE_OOS_GUARD = True
    hist = [
        HumanMessage(content="강아지가 초콜릿을 먹으면 어떤 증상이 나타나?"),
        AIMessage(content="구토, 발작 등이 나타납니다."),
    ]
    _, stream = q.answer_stream("그 중에 제일 위험한 건 뭐야?", hist, _resources())
    answer = "".join(stream)
    assert not answer.startswith(q.OOS_RESPONSE[:15]), "후속 질문이 OOS로 잘못 거절됨"


def test_true_out_of_scope_still_rejected():
    """맥락 없는 진짜 범위 외 질문은 여전히 거절되어야 한다(과잉 허용 방지)."""
    q.ENABLE_OOS_GUARD = True
    _, stream = q.answer_stream("페럿 키우는 법", [], _resources())
    assert "".join(stream).startswith(q.OOS_RESPONSE[:15])


def test_health_disclaimer_keywords():
    """건강/독성 키워드 감지 (초콜릿 등 독성음식 포함)."""
    assert q.needs_health_disclaimer("강아지가 초콜릿을 먹었어요")
    assert q.needs_health_disclaimer("포도를 삼켰어요")
    assert not q.needs_health_disclaimer("말티즈가 신나서 짖어요")


if __name__ == "__main__":
    test_followup_not_rejected_by_oos_guard()
    test_true_out_of_scope_still_rejected()
    test_health_disclaimer_keywords()
    print("ALL PASS")
