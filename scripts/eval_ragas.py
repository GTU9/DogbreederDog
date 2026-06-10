"""T2-2: RAG 자동 평가 하니스 (ragas).

ragas 가 앱의 langchain 스택을 업그레이드해 충돌시킬 수 있으므로 2단계로 분리한다.

  [1단계] 샘플 생성 (앱 스택 필요, OpenAI 호출):
      python scripts/eval_ragas.py --generate --samples data/eval/ragas_samples.json
      # USE_RERANKER 환경변수로 리랭커 ON/OFF 비교 샘플 생성 가능

  [2단계] 평가 (ragas 스택만 필요, OpenAI 호출):
      python scripts/eval_ragas.py --evaluate --samples data/eval/ragas_samples.json \
          --out data/eval/ragas_result.csv

  기본(둘 다, 단일 프로세스):
      python scripts/eval_ragas.py

측정 지표: faithfulness / answer_relevancy / context_precision
사전 준비: pip install "ragas<0.2" datasets  |  .env 의 OPENAI_API_KEY (비용 발생)
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --evaluate 단독 실행 시에도 .env(OPENAI_API_KEY)를 로드 (ragas 자체 LLM/임베딩이 사용)
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(ROOT, ".env"))
except Exception:  # noqa: BLE001
    pass


def _load_cases(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_samples(cases, out_path):
    """각 질문을 챗봇에 통과시켜 (question, answer, contexts, ground_truth) 수집 후 JSON 저장."""
    from py_file.QA_bot import build_resources, answer_question

    use_rer = os.getenv("USE_RERANKER", "1")
    print(f"[generate] 리소스 로딩... (USE_RERANKER={use_rer})")
    resources = build_resources()

    rows = []
    for i, c in enumerate(cases, 1):
        q = c["question"]
        print(f"  - ({i}/{len(cases)}) {c.get('id','')}: {q}")
        for attempt in (1, 2, 3):
            try:
                out = answer_question(q, resources=resources)
                break
            except Exception as e:  # noqa: BLE001  (전체 중단 방지: 재시도 후 기록)
                print(f"    재시도 {attempt}/3 ({type(e).__name__}: {e})")
                if attempt == 3:
                    out = {"answer": f"(생성 실패: {type(e).__name__})", "source_documents": []}
        contexts = [d.page_content for d in out["source_documents"]]
        if not contexts:
            contexts = ["(검색된 문서 없음 - 범위 외 또는 무응답)"]
        rows.append(
            {
                "id": c.get("id", ""),
                "question": q,
                "answer": out["answer"],
                "contexts": contexts,
                "ground_truth": c.get("ground_truth", ""),
            }
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"[generate] 샘플 {len(rows)}건 저장: {out_path}")
    return rows


def evaluate_samples(rows, out_csv):
    """ragas 로 지표 계산. ragas/datasets 미설치 시 안내 후 종료."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
    except ImportError as e:
        print(
            f"\n[오류] ragas/datasets 미설치({e}).\n"
            '       pip install "ragas<0.2" datasets  후 다시 실행하세요.',
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[evaluate] ragas 평가 실행({len(rows)}건)... (평가용 LLM/임베딩 호출 → 비용)")
    dataset = Dataset.from_dict(
        {
            "question": [r["question"] for r in rows],
            "answer": [r["answer"] for r in rows],
            "contexts": [r["contexts"] for r in rows],
            "ground_truth": [r["ground_truth"] for r in rows],
        }
    )
    result = evaluate(
        dataset, metrics=[faithfulness, answer_relevancy, context_precision]
    )
    print("\n[결과]")
    print(result)
    try:
        df = result.to_pandas()
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"\n상세 결과 저장: {out_csv}")
    except Exception as e:  # noqa: BLE001
        print(f"[경고] CSV 저장 실패: {e}", file=sys.stderr)
    return result


def main():
    p = argparse.ArgumentParser(description="RAG 자동 평가(ragas)")
    p.add_argument("--cases", default="data/eval/testcases.json")
    p.add_argument("--samples", default="data/eval/ragas_samples.json")
    p.add_argument("--out", default="data/eval/ragas_result.csv")
    p.add_argument("--generate", action="store_true", help="샘플 생성만(앱 스택)")
    p.add_argument("--evaluate", action="store_true", help="평가만(ragas 스택)")
    args = p.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("[경고] OPENAI_API_KEY 미설정(.env 확인).", file=sys.stderr)

    do_both = not (args.generate or args.evaluate)

    if args.generate or do_both:
        rows = generate_samples(_load_cases(args.cases), args.samples)
    if args.evaluate or do_both:
        if not (args.generate or do_both):  # evaluate-only: 파일에서 로드
            with open(args.samples, "r", encoding="utf-8") as f:
                rows = json.load(f)
        evaluate_samples(rows, args.out)


if __name__ == "__main__":
    main()
