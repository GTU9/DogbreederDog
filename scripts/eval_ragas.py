"""T2-2: RAG 자동 평가 하니스 (ragas).

README의 테스트 케이스(TC01~05)를 자동 평가한다. 각 질문을 실제 챗봇 파이프라인
(QA_bot.answer_question)에 통과시켜 생성 답변과 검색 문서를 모은 뒤, ragas로 지표를 계산한다.

측정 지표
- faithfulness        : 답변이 검색 문서에 근거하는가(환각 여부)
- answer_relevancy    : 답변이 질문에 실제로 답하는가
- context_precision   : 검색된 문서가 질문/정답과 관련 있는가(retriever 품질)

사전 준비
    pip install ragas datasets
    .env 에 OPENAI_API_KEY 설정 (챗봇 + ragas 평가 LLM 호출에 사용 → 비용 발생)

실행
    python scripts/eval_ragas.py
    python scripts/eval_ragas.py --cases data/eval/testcases.json --out data/eval/ragas_result.csv

주의: 임베딩 모델(bge-m3) 및 리랭커 모델 다운로드/로딩, OpenAI API 호출이 발생하므로
시간과 비용이 듭니다. CI에서 돌릴 때는 케이스 수를 줄이거나 모델을 캐시해 두세요.
"""

import argparse
import json
import os
import sys

# 프로젝트 루트를 import 경로에 추가(py_file 패키지 사용)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _load_cases(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _generate_samples(cases):
    """각 질문을 챗봇에 통과시켜 (answer, contexts)를 수집."""
    from py_file.QA_bot import build_resources, answer_question

    print(f"[1/3] 리소스 로딩 중... (임베딩/FAISS/리랭커)")
    resources = build_resources()

    rows = []
    for i, c in enumerate(cases, 1):
        q = c["question"]
        print(f"  - ({i}/{len(cases)}) {c.get('id','')}: {q}")
        out = answer_question(q, resources=resources)
        contexts = [d.page_content for d in out["source_documents"]]
        if not contexts:  # 범위 외 등으로 문서가 없으면 ragas 오류 방지용 placeholder
            contexts = ["(검색된 문서 없음 - 범위 외 또는 무응답)"]
        rows.append(
            {
                "question": q,
                "answer": out["answer"],
                "contexts": contexts,
                "ground_truth": c.get("ground_truth", ""),
            }
        )
    return rows


def _evaluate(rows):
    """ragas로 지표 계산. ragas/datasets 미설치 시 안내 후 종료."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
        )
    except ImportError:
        print(
            "\n[오류] ragas/datasets 가 설치되지 않았습니다.\n"
            "       pip install ragas datasets  후 다시 실행하세요.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("[2/3] ragas 평가 실행 중... (평가용 LLM 호출 → 비용 발생)")
    dataset = Dataset.from_dict(
        {
            "question": [r["question"] for r in rows],
            "answer": [r["answer"] for r in rows],
            "contexts": [r["contexts"] for r in rows],
            "ground_truth": [r["ground_truth"] for r in rows],
        }
    )
    return evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )


def main():
    parser = argparse.ArgumentParser(description="RAG 자동 평가(ragas)")
    parser.add_argument("--cases", default="data/eval/testcases.json")
    parser.add_argument("--out", default="data/eval/ragas_result.csv")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("[경고] OPENAI_API_KEY 가 설정되지 않았습니다(.env 확인).", file=sys.stderr)

    cases = _load_cases(args.cases)
    rows = _generate_samples(cases)
    result = _evaluate(rows)

    print("\n[3/3] 결과")
    print(result)

    # 상세 결과 CSV 저장
    try:
        df = result.to_pandas()
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        df.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"\n상세 결과 저장: {args.out}")
    except Exception as e:  # noqa: BLE001
        print(f"[경고] CSV 저장 실패: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
