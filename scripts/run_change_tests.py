"""T1/T2 변경점 실측 테스트 러너.

--local : 모델만 사용(OpenAI 불필요). 임베딩/FAISS/리랭커 로드 + dense/하이브리드/리랭커 검색 비교.
--e2e   : OpenAI 호출 포함. 범위 가드/면책/대화 메모리/교차언어 단일화 검증.

결과는 UTF-8 파일(data/eval/*_results.txt)로 직접 기록한다(콘솔/파이프 인코딩 깨짐 방지).
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_BUF = []


def log(*a):
    line = " ".join(str(x) for x in a)
    _BUF.append(line)
    sys.stdout.write(line.encode("ascii", "replace").decode() + "\n")


def flush(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(_BUF) + "\n")
    print(f"[written] {path}")


QUERIES = [
    "말티즈가 자꾸 짖어요",
    "강아지가 분리불안이 있어요",
    "강아지가 벽지를 긁어요",
]


def _top(retriever, query, k=3):
    docs = retriever.invoke(query)[:k]
    return [
        (d.metadata.get("title", "") or "", d.metadata.get("source", "") or "")
        for d in docs
    ]


def local_tests():
    import py_file.QA_bot as q
    from langchain_community.vectorstores import FAISS

    log("== TP-A: 리소스 로드(임베딩+FAISS) ==")
    t0 = time.time()
    emb = q._load_embedding()
    t_emb = time.time() - t0
    t1 = time.time()
    db = FAISS.load_local(q.FAISS_DB_PATH, emb, allow_dangerous_deserialization=True)
    t_db = time.time() - t1
    ndocs = len(db.docstore._dict)
    log(f"임베딩 로드: {t_emb:.1f}s | FAISS 로드: {t_db:.1f}s | 문서수: {ndocs}")

    modes = [
        ("dense-only", False, False),
        ("hybrid", True, False),
        ("hybrid+reranker", True, True),
    ]
    for name, hyb, rer in modes:
        log(f"\n== TP-B/{name} ==")
        q.USE_HYBRID, q.USE_RERANKER = hyb, rer
        t = time.time()
        try:
            retr = q._build_retriever(db)
            log(f"[{name}] 구성 {time.time()-t:.1f}s")
            for query in QUERIES:
                log(f"  Q: {query}")
                for i, (title, src) in enumerate(_top(retr, query), 1):
                    log(f"    {i}. title={title!r} src={src[:60]}")
        except Exception as e:  # noqa: BLE001
            log(f"[{name}] 실패: {type(e).__name__}: {e}")

    log("\n== TP-D: 출처 메타데이터 ==")
    q.USE_HYBRID, q.USE_RERANKER = False, False
    retr = q._build_retriever(db)
    for i, d in enumerate(retr.invoke(QUERIES[0])[:3], 1):
        log(f"  {i}. keys={sorted(d.metadata.keys())} "
            f"title={d.metadata.get('title','')!r} src?={bool(d.metadata.get('source'))}")
    log("\nLOCAL DONE")
    flush("data/eval/local_results.txt")


def e2e_tests():
    import py_file.QA_bot as q
    from langchain_core.messages import HumanMessage, AIMessage

    log("== build_resources (리랭커 포함) ==")
    t = time.time()
    res = q.build_resources()
    log(f"build_resources: {time.time()-t:.1f}s")

    log("\n== TP-E: 범위 외(페럿) ==")
    out = q.answer_question("페럿이 우울해 보여요", resources=res)
    log(f"docs={len(out['source_documents'])}")
    log(f"answer={out['answer'][:240]!r}")
    log("OOS_HIT:", out["answer"].startswith(q.OOS_RESPONSE[:20]))

    log("\n== TP-F: 건강 면책(구토) ==")
    out = q.answer_question("강아지가 자꾸 구토해요", resources=res)
    log("DISCLAIMER_PRESENT:", q.HEALTH_DISCLAIMER.strip()[:20] in out["answer"])
    log(f"answer_tail={out['answer'][-160:]!r}")

    log("\n== TP-G: 대화 메모리 ==")
    o1 = q.answer_question("말티즈가 자꾸 짖어요", resources=res)
    log(f"turn1={o1['answer'][:160]!r}")
    hist = [HumanMessage(content="말티즈가 자꾸 짖어요"), AIMessage(content=o1["answer"])]
    o2 = q.answer_question("그럼 어떻게 훈련해?", chat_history=hist, resources=res)
    ans2 = o2["answer"]
    log(f"turn2={ans2[:240]!r}")
    log("CONTEXT_KEPT(말티즈/짖 언급):", ("말티즈" in ans2) or ("짖" in ans2))

    log("\n== TP-H: 교차언어 단일화(영어 출처 질의) ==")
    oh = q.answer_question("강아지가 사람 손을 자꾸 깨물어요", resources=res)
    srcs = [d.metadata.get("source", "") for d in oh["source_documents"]]
    en_hit = any(s.startswith(q.KINSHIP_PREFIX) for s in srcs)
    log(f"top_sources={[s[:55] for s in srcs]}")
    log(f"english_source_branch={en_hit} | single_answer_len={len(oh['answer'])}")
    log(f"answer={oh['answer'][:200]!r}")
    log("\nE2E DONE")
    flush("data/eval/e2e_results.txt")


# 카테고리별 쇼케이스 질문(질문 + 실제 답변/출처 확인용)
SHOWCASE = [
    ("견종", "푸들은 어떤 성격을 가지고 있어?", None),
    ("견종", "골든리트리버는 훈련을 어떻게 시켜야 해?", None),
    ("행동", "강아지가 산책할 때 자꾸 줄을 당겨요", None),
    ("행동", "강아지가 낯선 사람만 보면 짖어요", None),
    ("건강(면책)", "강아지가 어제부터 설사를 해요", None),
    ("건강(면책)", "강아지가 다리를 절뚝거려요", None),
    ("범위 외", "고양이가 밥을 안 먹어요", None),
    ("범위 외", "오늘 서울 날씨 어때?", None),
    ("교차언어", "강아지가 자기 꼬리를 빙빙 돌며 쫓아요", None),
    ("대화메모리", "비글은 어떤 견종이야?", "그럼 운동은 얼마나 시켜야 해?"),
]


def showcase_tests(max_answer=500):
    import py_file.QA_bot as q
    from langchain_core.messages import HumanMessage, AIMessage

    log("== SHOWCASE: 질문별 실제 답변/출처 ==")
    t = time.time()
    res = q.build_resources()
    log(f"(build_resources {time.time()-t:.1f}s)\n")

    for idx, (cat, question, followup) in enumerate(SHOWCASE, 1):
        log(f"### [{idx}] ({cat}) {question}")
        out = q.answer_question(question, resources=res)
        ans = out["answer"]
        oos = out["source_documents"] == [] and ans.startswith(q.OOS_RESPONSE[:15])
        disc = q.HEALTH_DISCLAIMER.strip()[:18] in ans
        titles = []
        for d in out["source_documents"]:
            ti = (d.metadata.get("title") or d.metadata.get("source") or "").strip()
            if ti and ti not in titles:
                titles.append(ti)
        flags = []
        if oos:
            flags.append("범위외-가드")
        if disc:
            flags.append("건강-면책")
        log(f"- 플래그: {flags or '없음'}")
        log(f"- 출처: {titles or '없음'}")
        body = ans if len(ans) <= max_answer else ans[:max_answer] + " …(생략)"
        log(f"- 답변:\n{body}")

        if followup:
            log(f"\n  ↳ 후속질문: {followup}")
            hist = [HumanMessage(content=question), AIMessage(content=ans)]
            o2 = q.answer_question(followup, chat_history=hist, resources=res)
            a2 = o2["answer"]
            body2 = a2 if len(a2) <= max_answer else a2[:max_answer] + " …(생략)"
            log(f"  ↳ 답변:\n{body2}")
        log("")

    log("SHOWCASE DONE")
    flush("data/eval/showcase_results.txt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--local", action="store_true")
    p.add_argument("--e2e", action="store_true")
    p.add_argument("--showcase", action="store_true")
    args = p.parse_args()
    if args.local:
        local_tests()
    if args.e2e:
        _BUF.clear()
        e2e_tests()
    if args.showcase:
        _BUF.clear()
        showcase_tests()
    if not (args.local or args.e2e or args.showcase):
        print("use --local and/or --e2e and/or --showcase")


if __name__ == "__main__":
    main()
