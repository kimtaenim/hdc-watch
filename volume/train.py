"""거래량 모델 학습: 최근 60일(야후 5분봉) 봉을 모두 더해 프로토타입을 만들고, 그 기간을 다시 채점한다.

실행: python -m volume.train [--stock 045510.KQ] [--end YYYY-MM-DD] [--raw 저장된원자료.json]
결과: docs/data/volume/<종목>/model.json, bars.json, raw.json, traces/train-*.json, docs/data/runs.json
"""

import argparse
import datetime as dt
import json
import os
import traceback

import numpy as np

from hdc import D, bind, hamming, nearest, noise_level, similarity
from hdc.trace import REPO_ROOT, Trace, check, source_of, vec_info, why
from volume import data, model

CONFIG = os.path.join(REPO_ROOT, "config", "watch.json")


def stock_dir(stock_id: str) -> str:
    return os.path.join(REPO_ROOT, "docs", "data", "volume", stock_id)


def rel(path: str) -> str:
    return os.path.relpath(path, os.path.join(REPO_ROOT, "docs")).replace(os.sep, "/")


def append_run(entry: dict) -> None:
    path = os.path.join(REPO_ROOT, "docs", "data", "runs.json")
    runs = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else []
    runs.append(entry)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(runs, f, ensure_ascii=False, indent=1)


def histogram(values, width=0.005):
    lo = np.floor(min(values) / width) * width
    hi = np.ceil(max(values) / width) * width + width
    counts, edges = np.histogram(values, bins=np.arange(lo, hi + 1e-12, width))
    return [[round(float(edges[i]), 4), int(counts[i])] for i in range(len(counts))]


def run(stock_id: str, end: str | None, raw_path: str | None):
    cfg = json.load(open(CONFIG, encoding="utf-8"))
    stock = next(s for s in cfg["stocks"] if s["id"] == stock_id)
    vc = cfg["volume"]
    sessions, levels = vc["sessions"], vc["levels"]
    now = dt.datetime.now(data.KST)
    stamp = now.strftime("%Y%m%d-%H%M%S")
    out = stock_dir(stock_id)
    trace_path = os.path.join(out, "traces", f"train-{stamp}.json")

    t = Trace("volume.train", subject={"type": "stock", "id": stock_id, "code": stock["code"],
                                       "name": stock["name"]},
              params={"D": D, "levels": levels, "sessions": sessions, "source": "야후 5분봉 최근 60일",
                      "ratio_edge_exponents": vc["ratio_edge_exponents"]})

    t.section("공통: 자료와 거래량 단계", "HDC 연산 없이 숫자를 이름표로 바꾸는 앞단.")
    # ① 원자료 받기
    try:
        if raw_path:
            raw = json.load(open(raw_path, encoding="utf-8"))
            fetched_at = raw.get("fetched_at")
            t.note(f"저장된 원자료 사용: {raw_path}")
        else:
            raw = data.fetch(stock_id)
            fetched_at = now.isoformat(timespec="seconds")
        fivemin, daily = raw["fivemin"], raw["daily"]
    except Exception as e:
        t.failure("저장된 원자료 읽기" if raw_path else "yfinance 수집", f"{type(e).__name__}: {e}")
        t.save(trace_path)
        append_run({"at": now.isoformat(timespec="seconds"), "kind": "volume.train", "subject": stock_id,
                    "fetched": {}, "failures": [f"수집 실패: {e}"], "trace": rel(trace_path)})
        raise
    slotted = data.slots_from_5m(fivemin, 30)
    t.fetched("5분봉", len(fivemin))
    t.fetched("일봉", len(daily))
    days = data.complete_days(slotted, daily, now)
    if end:
        days = [d for d in days if d <= end]
    dates = days[1:]  # 60일의 첫날은 중간부터 잘려 있을 수 있어 뺀다
    t.step("원자료 받기 (5분봉)", "일반 계산", [
        why("설계 선택", "yfinance(야후 금융의 비공식 경로)로 5분봉을 받는다. 야후가 5분봉을 주는 최대 기간인 최근 60일이 학습 기간이다. "
                       "받은 원자료는 raw.json에 그대로 남긴다."),
        why("설계 선택", "60일의 첫날은 중간부터 잘려 있을 수 있어 빼고, 오늘은 장 마감 뒤(16시 이후)에만 쓴다."),
    ], outputs=[{"type": "value", "name": "받은 5분봉", "value": len(fivemin)},
                {"type": "value", "name": "학습 기간", "value": f"{dates[0]} - {dates[-1]}"},
                {"type": "value", "name": "학습 거래일 수", "value": len(dates)},
                {"type": "value", "name": "받은 시각", "value": fetched_at}],
       source=data.fetch)

    # 5분봉으로 만든 30분 봉과 야후 30분봉 대조
    yh = {(r["date"], r["slot"]): r["volume"] for r in raw.get("yahoo_30m", [])}
    pairs = [(h, yh.get((h["date"], h["slot"]))) for h in slotted if h["date"] in set(dates) and h["slot"] in sessions]
    pairs = [(h, y) for h, y in pairs if y is not None]
    same = sum(1 for h, y in pairs if h["volume"] == y)
    rel_diff = [abs(h["volume"] - y) / y for h, y in pairs if y > 0]
    rel_med, rel_95 = (float(np.median(rel_diff)), float(np.percentile(rel_diff, 95))) if rel_diff else (0.0, 0.0)
    zero_y = [(h, y) for h, y in pairs if y == 0 and h["volume"] > 0]
    # ② 하루 봉 만들기
    bars = data.build_bars(slotted, dates, sessions)
    span = f"{sessions[0]}-{sessions[-1]}"
    for b in bars:
        if b["missing"]:
            t.failure("봉 만들기", f"{b['date']} {b['slot']}: {b['missing']}")
    n_empty = sum(1 for b in bars if b["n5"] == 0)
    t.step(f"5분봉을 30분씩 합쳐 하루 {len(sessions)}개 봉 만들기", "일반 계산", [
        why("설계 선택", f"5분봉을 30분 칸으로 합쳐 {span} 봉 {len(sessions)}개를 만든다. 15:00 이후는 야후 5분봉에 없고, "
                       "'일별 - 5분봉 합'으로 역산하면 음수가 나오는 날이 있어 뺀다(2026-11-11 결정)."),
        why("이 데이터에서 관찰", f"5분봉이 하나도 없는(거래가 없던) 30분 칸은 {n_empty}개이고, 거래량 0인 봉으로 둔다."),
        why("이 데이터에서 관찰", f"5분봉 합과 야후 30분봉이 정확히 같은 봉은 {len(pairs)}개 중 {same}개다."
                               + ("" if same == len(pairs) else
                                  f" 상대 차이 중앙값은 {rel_med:.1%}, 95% 지점은 {rel_95:.1%}다. 원인은 확인하지 못했다. "
                                  f"야후 30분봉이 0인데 5분봉에는 거래가 있는 칸이 {len(zero_y)}개다.")),
    ], outputs=[{"type": "value", "name": "만든 봉", "value": len(bars)},
                {"type": "value", "name": "거래 없던 칸(0주)", "value": n_empty},
                {"type": "value", "name": "야후 30분봉과 같은 봉", "value": f"{same} / {len(pairs)}"},
                {"type": "table", "name": "야후 30분봉이 0인데 5분봉엔 거래가 있던 칸", "columns": ["날짜", "시간", "5분봉 합", "야후 30분봉"],
                 "rows": [[h["date"], h["slot"], h["volume"], y] for h, y in zero_y]},
                {"type": "table", "name": "최근 하루", "columns": ["날짜", "요일", "시간", "거래량", "5분봉 수"],
                 "rows": [[b["date"], b["weekday"], b["slot"], b["volume"], b.get("n5")] for b in bars[-len(sessions):]]}],
       check=check("만든 봉 수 = 거래일 수 × 시간대 수", len(dates) * len(sessions), len(bars)),
       source=data.slots_from_5m)

    # ③ 시간대 중앙값
    meds = data.session_medians(bars, sessions)
    t.step("시간대별 평소 거래량(중앙값)", "일반 계산", [
        why("설계 선택", "거래량은 절대량이 아니라 '그 시간대 중앙값의 몇 배인가'로 본다. 장 시작과 마감 무렵은 원래 많고 점심 무렵은 원래 적기 때문이다."),
        why("수학적 사실", "중앙값은 한두 번의 폭증에 끌려가지 않는다. 평균은 끌려간다."),
    ], outputs=[{"type": "table", "name": "시간대별 중앙값", "columns": ["시간", "중앙값(주)", "봉 수"],
                 "rows": [[s, meds[s]["median"], meds[s]["n"]] for s in sessions]}],
       source=data.session_medians)

    # ④ 단계 나누기
    edges = 2.0 ** np.array(vc["ratio_edge_exponents"], dtype=float)
    data.assign_levels(bars, meds, edges)
    regular = [b["level"] for b in bars if b["level"]]
    dist_reg = np.bincount(regular, minlength=levels + 1)[1:].tolist()
    bounds_txt = ["0"] + [f"{e:g}" for e in edges] + ["∞"]
    t.step("거래량 단계 나누기", "일반 계산", [
        why("설계 선택", f"배율의 경계를 2의 {vc['ratio_edge_exponents'][0]}제곱부터 {vc['ratio_edge_exponents'][-1]}제곱까지 2배 간격으로 두어 "
                       f"{levels}단계로 나눈다. 6단계가 평소(1배-2배)다. 경계값은 위 단계에 넣는다."),
        why("설계 선택", "2배 간격은 '평소의 2배'와 '평소의 4배'를 같은 거리로 본다는 뜻이다(로그 눈금)."),
    ], outputs=[{"type": "table", "name": "단계별 배율 구간과 봉 수",
                 "columns": ["단계", "배율 구간", "봉 수"],
                 "rows": [[k + 1, f"{bounds_txt[k]} - {bounds_txt[k + 1]}", dist_reg[k]]
                          for k in range(levels)]}],
       check=check("봉 수 = 단계별 봉 수 합", sum(1 for b in bars if b["level"]),
                   sum(dist_reg)),
       source=data.level_of)

    # ⑤ 기본 벡터 꺼내기
    t.section("시간대 모델", "바이너리 스패터 코드. 봉 하나 = 요일 ⊕ 시간대 ⊕ 거래량 단계. "
                         "프로토타입 = 봉 벡터들을 더한 것. '이 요일·시간대에 이 거래량이 흔했나'를 묻는 프로토타입 기반 이상탐지.")
    cb = model.Codebook(sessions, levels, D)
    base = list(cb.days.values()) + list(cb.sessions.values())
    sims_ds = [similarity(x, y) for i, x in enumerate(base) for j, y in enumerate(base) if i < j]
    n_base, n_pairs = len(base), len(sims_ds)
    t.step(f"요일 벡터 {len(cb.days)}개, 시간대 벡터 {len(cb.sessions)}개 꺼내기", "벡터 생성", [
        why("설계 선택", "요일과 시간대는 이름의 SHA-256에서 만든 서로 무관한 벡터다. 월요일과 화요일이 가깝다는 식의 순서 정보는 넣지 않았다."),
        why("수학적 사실", f"무작위 벡터 두 개의 유사도는 평균 0, 표준편차 1/√D = {noise_level(D):.2f}인 분포를 따른다(근사). "
                       "SHA-256 출력을 무작위로 볼 수 있다는 것은 증명된 정리가 아니라 널리 받아들이는 가정이다."),
        why("이 데이터에서 관찰", f"{n_base}개 벡터의 {n_pairs}쌍 유사도는 {min(sims_ds):.4f}부터 {max(sims_ds):.4f}까지였다."),
    ], outputs=[vec_info(model.day_name(k), v, full=True) for k, v in cb.days.items()]
               + [vec_info(model.session_name(k), v, full=True) for k, v in cb.sessions.items()]
               + [{"type": "value", "name": f"{n_pairs}쌍 유사도 범위",
                   "value": [round(min(sims_ds), 4), round(max(sims_ds), 4)]}],
       check=check(f"{n_pairs}쌍 모두 |유사도| ≤ 0.03인가", True, bool(max(abs(x) for x in sims_ds) <= 0.03)),
       source=model.Codebook)

    lv_sims = [round(similarity(cb.level(1), cb.level(k)), 4) for k in range(1, levels + 1)]
    t.step(f"수준 벡터 {levels}단계 만들기", "뒤집기", [
        why("설계 선택", f"1단계는 '{model.LADDER_NAME}'에서 꺼낸 벡터. 한 단계 오를 때마다 아직 안 뒤집은 칸을 약 {D // 2 // (levels - 1)}개씩 뒤집는다. "
                       "HDC 문헌의 수준 벡터와 같은 방식이다."),
        why("수학적 사실", f"뒤집은 칸이 겹치지 않으므로 두 단계의 해밍 거리는 그 사이에 뒤집은 칸 수와 정확히 같다. "
                       f"그래서 유사도는 단계 차이에 따라 일정하게 줄고, 1단계와 {levels}단계는 정확히 절반이 달라 유사도 0이다."),
    ], outputs=[vec_info(f"{k}단계", cb.level(k), full=True) for k in range(1, levels + 1)]
               + [{"type": "table", "name": "단계마다 뒤집은 칸 수", "columns": ["구간", "칸 수"],
                   "rows": [[f"{k + 1}→{k + 2}", len(f)] for k, f in enumerate(cb.flips)]},
                  {"type": "table", "name": "1단계와의 유사도", "columns": ["단계", "유사도"],
                   "rows": [[k, x] for k, x in zip(range(1, levels + 1), lv_sims)]}],
       check=check(f"1단계와 {levels}단계의 다른 칸 수", D // 2, hamming(cb.level(1), cb.level(levels))),
       source=model.Codebook)

    used = [b for b in bars if b["level"] is not None]
    mem = model.train(cb, used)
    mv = mem.to_vector()
    vecs = [model.bar_vector(cb, b["weekday"], b["slot"], b["level"]) for b in used]

    # 봉 하나 묶기 예시는 채점 뒤에 고른다(가장 드문 봉)
    for b, v in zip(used, vecs):
        b["score"] = round(model.counter_score_leave_one_out(mem, v), 4)
        b["score_bin"] = round(model.score_leave_one_out(mem, v), 4)
        cl, same, n, cl_sum = model.count_closeness(used, b, levels)
        b["count_closeness"] = round(cl, 4) if cl is not None else None
        b["count_sum"] = round(cl_sum, 3)
        b["count_same"], b["count_peers"] = same, n
    scores = np.array([b["score"] for b in used])
    bins = np.array([b["score_bin"] for b in used])
    alert = float(np.percentile(scores, vc["alert_percentile"]))
    watch = float(np.percentile(scores, vc["watch_percentile"]))
    alert_b = float(np.percentile(bins, vc["alert_percentile"]))
    watch_b = float(np.percentile(bins, vc["watch_percentile"]))
    ranked = sorted(used, key=lambda b: b["score"])
    for i, b in enumerate(ranked):
        b["rank"] = i + 1
        b["verdict"] = "이상" if b["score"] <= alert else "주의" if b["score"] <= watch else "평소"
    for i, b in enumerate(sorted(used, key=lambda b: b["score_bin"])):
        b["rank_bin"] = i + 1

    ex = ranked[0]
    dv, sv, lv = cb.days[ex["weekday"]], cb.sessions[ex["slot"]], cb.level(ex["level"])
    ds = bind(dv, sv)
    bv = bind(ds, lv)
    back = nearest(bind(bv, ds), {f"{k}단계": cb.level(k) for k in range(1, levels + 1)})
    t.step("봉 하나를 벡터 하나로 묶기", "묶기", [
        why("설계 선택", "봉 하나 = 요일 ⊕ 시간대 ⊕ 단계. 세 벡터를 XOR로 묶어 1만 비트 하나로 만든다."),
        why("수학적 사실", "XOR로 묶은 벡터는 무작위 재료와 무관해진다(유사도 0 근처). 같은 열쇠(요일 ⊕ 시간대)로 묶은 두 봉의 해밍 거리는 "
                       "두 단계 벡터의 해밍 거리와 정확히 같다. 그래서 같은 요일·시간대 안에서는 단계끼리 닮은 만큼만 닮는다."),
        why("수학적 사실", "묶은 벡터를 (요일 ⊕ 시간대)로 다시 XOR하면 단계 벡터가 그대로 나온다."),
        why("수학적 사실", "요일 ⊕ 시간대 열쇠들은 쌍마다 거의 무관하지만 서로 독립은 아니다. 예: (월 ⊕ 10시) ⊕ (월 ⊕ 11시) ⊕ (화 ⊕ 10시) = 화 ⊕ 11시. "
                       "그래서 다른 칸 봉들이 섞어 넣는 잡음은 무작위가 아니라 구조를 가진다."),
    ], inputs=[{"type": "value", "name": "예시 봉(가장 드문 봉)",
                "value": f"{ex['date']} ({ex['weekday']}) {ex['slot']} {ex['volume']:,}주, 중앙값의 {ex['ratio']:.2f}배, {ex['level']}단계"},
               vec_info(model.day_name(ex["weekday"]), dv, full=True),
               vec_info(model.session_name(ex["slot"]), sv, full=True),
               vec_info(f"{ex['level']}단계", lv, full=True)],
       outputs=[vec_info(f"{model.day_name(ex['weekday'])} ⊕ {model.session_name(ex['slot'])}", ds, full=True),
                vec_info("봉 벡터", bv, full=True),
                {"type": "value", "name": "봉 벡터와 요일 벡터의 유사도", "value": round(similarity(bv, dv), 4)},
                {"type": "value", "name": "봉 벡터와 단계 벡터의 유사도", "value": round(similarity(bv, lv), 4)}],
       check=check("봉 벡터 ⊕ (요일 ⊕ 시간대)에 가장 가까운 단계", f"{ex['level']}단계", back[0][0]),
       source=model.bar_vector)

    ctx = {(w, h): sum(1 for b in used if b["weekday"] == w and b["slot"] == h) for w in "월화수목금" for h in sessions}
    empty = [f"{w} {h}" for (w, h), n in ctx.items() if n == 0]
    nonempty = [n for n in ctx.values() if n]
    loo_ties = float(np.mean([int((model.without(mem, v).counts == 0).sum()) for v in vecs]))
    hist_counts = np.bincount(np.abs(mem.counts))
    t.step("모든 봉을 프로토타입에 더하기", "더하기", [
        why("설계 선택", f"학습 기간의 봉 {mem.n}개(결측 제외)를 칸마다 +1/-1로 누적한다. 이것이 정수 프로토타입이다. "
                       "다수결로 0/1을 정하면 이진 프로토타입이 된다."),
        why("수학적 사실", "정수 프로토타입은 새 봉의 +1/-1을 더하기만 하면 갱신되고, 더하는 순서와 무관하다. "
                       "다만 중앙값, 단계 경계, 판정 기준은 자료가 늘면 다시 계산해야 한다."),
        why("설계 선택", "다수결에서 0:0으로 비긴 칸은 고정된 동점깨기 벡터를 따른다. 모든 이진 프로토타입이 이 벡터 쪽으로 조금 치우치는 대가가 있다."),
        why("이 데이터에서 관찰", f"요일·시간대 칸마다 봉 수는 {min(nonempty)}-{max(nonempty)}개다. "
                               + (f"{', '.join(empty)}에는 봉이 하나도 없다. "
                                  "이 칸의 새 봉은 비교할 같은 칸 봉이 없다." if empty else "")),
    ], inputs=[{"type": "value", "name": "더한 봉 수", "value": mem.n}],
       outputs=[{"type": "value", "name": "정수 프로토타입 범위", "value": [int(mem.counts.min()), int(mem.counts.max())]},
                {"type": "value", "name": "모든 봉이 같은 값이던 칸 수(|정수 프로토타입| = 봉 수)", "value": int((np.abs(mem.counts) == mem.n).sum())},
                {"type": "table", "name": "정수 프로토타입 절댓값 분포", "columns": ["|정수 프로토타입|", "칸 수"],
                 "rows": [[i, int(c)] for i, c in enumerate(hist_counts) if c]},
                {"type": "value", "name": "전체 프로토타입의 동점(0) 칸 수", "value": mem.tie_count()},
                {"type": "value", "name": "자기 제외 프로토타입의 동점 칸 수(평균)", "value": round(loo_ties, 1)},
                {"type": "table", "name": "요일·시간대 칸별 봉 수", "columns": ["요일"] + list(sessions),
                 "rows": [[w] + [ctx[(w, h)] for h in sessions] for w in "월화수목금"]},
                vec_info("이진 프로토타입", mv, full=True)],
       check=check("더한 봉 수 = 결측 아닌 봉 수", len(used), mem.n),
       source=model.train)

    noise = noise_level(D)
    rho_c = model.spearman(scores, [b["count_sum"] for b in used])
    rho_b = model.spearman(bins, [b["count_sum"] for b in used])
    rho_cb = model.spearman(scores, bins)
    t.step("학습 기간을 다시 채점하기: 정수 점수와 이진 점수", "비교", [
        why("설계 선택", "본 점수 = 봉 벡터(0→-1, 1→+1)와 정수 프로토타입의 코사인. 비교용 점수 = 봉 벡터와 이진 프로토타입의 해밍 유사도(바이너리 스패터 코드 본래 방식). "
                       "둘 다 그 봉 자신을 정수 프로토타입에서 뺀 뒤 잰다."),
        why("수학적 사실", "봉 벡터를 무작위로 뽑는다고 보면, 정수 점수의 기댓값은 같은 요일·시간대 다른 봉들의 '단계 가까움' 합에 비례한다. "
                       "다수결을 거친 이진 점수는 이 합의 비선형 함수라 순서가 달라진다."),
        why("이 데이터에서 관찰", f"단순 카운트(가까움 합)와의 순위 상관: 정수 점수 {rho_c:.2f}, 이진 점수 {rho_b:.2f}. "
                               f"두 점수끼리의 순위 상관은 {rho_cb:.2f}다. 다수결에서 잃는 정보가 크다."),
        why("이 데이터에서 관찰", f"이진 점수는 평균 {bins.mean():.3f}, 양수인 봉 {(bins > 0).mean():.0%}다. 정수 점수는 평균 {scores.mean():.3f}다. "
                               "이진 점수가 한쪽(음수)으로 쏠리는 원인은 확인하지 못했다. 후보 하나: 세 벡터의 XOR(a ⊕ b ⊕ c)과 그 셋의 다수결은 "
                               "8가지 경우 중 2가지만 같아 유사도가 -0.5다(수학적 사실). 요일 ⊕ 시간대 열쇠들이 XOR로 얽혀 있어 이런 관계가 다수결 안에 섞일 수 있다(추측)."),
        why("수학적 사실", f"점수에 섞이는 잡음의 크기는 약 1/√D = {noise:.3f}(근사)다. 이 잡음은 실행마다 바뀌지 않는다. "
                       "벡터 이름이 정해지면 고정되는 치우침이라, 같은 조합은 늘 같은 방향으로 틀린다."),
    ], outputs=[{"type": "table", "name": "정수 점수 분포", "columns": ["구간 시작", "봉 수"], "rows": histogram(scores)},
                {"type": "value", "name": "정수 점수와 카운트 순위 상관", "value": round(rho_c, 3)},
                {"type": "value", "name": "이진 점수와 카운트 순위 상관", "value": round(rho_b, 3)},
                {"type": "value", "name": "두 점수의 순위 상관", "value": round(rho_cb, 3)},
                {"type": "table", "name": "봉별 점수",
                 "columns": ["날짜", "요일", "시간대", "거래량", "배율", "단계", "점수", "이진 점수",
                             "카운트 가까움 합", "같은 단계 수", "같은 칸 봉 수", "순위", "이진 순위", "판정"],
                 "rows": [[b["date"], b["weekday"], b["slot"], b["volume"], round(b["ratio"], 4),
                           b["level"], b["score"], b["score_bin"], b["count_sum"], b["count_same"],
                           b["count_peers"], b["rank"], b["rank_bin"], b["verdict"]] for b in used]}],
       check=check("카운트 하위 30개(가장 드문 30개) 중 정수 점수 하위 30개에 든 수", 30,
                   len({id(b) for b in sorted(used, key=lambda b: b["count_sum"])[:30]} & {id(b) for b in ranked[:30]})),
       source=model.counter_score_leave_one_out)

    # 이름을 바꿔 다시 만들면?
    robust = []
    for salt in ("다른이름1|", "다른이름2|"):
        cb2 = model.Codebook(sessions, levels, D, salt)
        mem2 = model.train(cb2, used)
        s2 = [model.counter_score_leave_one_out(mem2, model.bar_vector(cb2, b["weekday"], b["slot"], b["level"])) for b in used]
        top2 = {id(used[i]) for i in np.argsort(s2)[:30]}
        robust.append([salt, round(model.spearman(scores, s2), 3), len(top2 & {id(b) for b in ranked[:30]})])
    t.step("이름을 바꿔 다시 만들어 보기", "비교", [
        why("설계 선택", "요일·시간대·단계 벡터 이름 앞에 글자를 붙여 전혀 다른 벡터로 같은 계산을 다시 한다. 점수가 벡터 선택에 얼마나 기대는지 본다."),
        why("이 데이터에서 관찰", f"다시 만든 점수와의 순위 상관은 {robust[0][1]:.2f}, {robust[1][1]:.2f}이고, 하위 30개 중 겹친 수는 "
                               f"{robust[0][2]}, {robust[1][2]}개다. D = {D:,}에서도 순위의 세부는 벡터 선택에 따라 달라진다. "
                               "'이상' 판정 하나하나를 확정된 답으로 읽으면 안 된다."),
    ], outputs=[{"type": "table", "name": "이름을 바꾼 결과", "columns": ["붙인 글자", "순위 상관", "하위 30 겹침"], "rows": robust}],
       source=model.Codebook)

    n_alert = sum(b["verdict"] == "이상" for b in used)
    n_watch = sum(b["verdict"] == "주의" for b in used)
    t.step("판정 기준 정하기", "일반 계산", [
        why("설계 선택", f"학습 봉 정수 점수의 하위 {vc['alert_percentile']}% 지점을 '이상', 하위 {vc['watch_percentile']}% 지점을 '주의' 경계로 삼는다. "
                       "기준이 이 종목의 자기 데이터에서 나오므로 종목마다 다르다."),
        why("이 데이터에서 관찰", f"같은 점수인 봉이 있어 정확히 {vc['alert_percentile']}%가 되지는 않는다. 이번에는 '이상' {n_alert}개, '주의' {n_watch}개다."),
    ], outputs=[{"type": "value", "name": "이상 경계(점수 이하)", "value": round(alert, 4)},
                {"type": "value", "name": "주의 경계(점수 이하)", "value": round(watch, 4)},
                {"type": "value", "name": "이진 점수의 이상 경계(참고)", "value": round(alert_b, 4)},
                {"type": "table", "name": "판정별 봉 수", "columns": ["판정", "봉 수"],
                 "rows": [[v, sum(b["verdict"] == v for b in used)] for v in ("이상", "주의", "평소")]}],
       source=run)

    big = [b for b in used if b["ratio"] >= 16]
    count_rank = {id(b): i + 1 for i, b in enumerate(sorted(used, key=lambda b: b["count_sum"]))}
    t.step("가장 드문 봉 순위", "일반 계산", [
        why("설계 선택", "정수 점수가 낮은 순으로 줄 세운다. 옆에 이진 점수 순위와 단순 카운트 순위를 둔다."),
        why("이 데이터에서 관찰", "점수가 묻는 것은 '이 요일·시간대에 이 단계가 흔했나'다. '중앙값의 몇 배인가'와 같은 질문이 아니다. "
                               "그래서 거래가 거의 없던 봉(1-2단계)도 폭증한 봉만큼 낮은 점수를 받는다(1단계와 12단계가 평소에서 비슷하게 멀다)."),
    ], outputs=[{"type": "table", "name": "하위 30",
                 "columns": ["순위", "날짜", "요일", "시간대", "거래량", "배율", "단계", "점수", "판정", "이진 순위", "카운트 순위"],
                 "rows": [[b["rank"], b["date"], b["weekday"], b["slot"], b["volume"], round(b["ratio"], 2),
                           b["level"], b["score"], b["verdict"], b["rank_bin"], count_rank[id(b)]] for b in ranked[:30]]},
                {"type": "table", "name": f"평소의 16배 이상이던 {span} 봉",
                 "columns": ["순위", "날짜", "요일", "시간대", "배율", "단계", "점수", "이진 순위", "카운트 순위"],
                 "rows": [[b["rank"], b["date"], b["weekday"], b["slot"], round(b["ratio"], 1), b["level"],
                           b["score"], b["rank_bin"], count_rank[id(b)]] for b in sorted(big, key=lambda b: b["rank"])]}],
       check=check("16배 이상 봉 중 하위 30위 안에 든 수", len(big), sum(1 for b in big if b["rank"] <= 30)),
       source=model.counter_score_leave_one_out)

    # 저장
    os.makedirs(out, exist_ok=True)
    window_set = set(dates)
    with open(os.path.join(out, "raw.json"), "w", encoding="utf-8") as f:
        json.dump({"fetched_at": fetched_at, "stock": stock_id, "source": "yahoo 5m 60d",
                   "window": [dates[0], dates[-1]],
                   "fivemin": fivemin, "slots": slotted, "yahoo_30m": raw.get("yahoo_30m", []),
                   "daily": daily}, f, ensure_ascii=False)
    with open(os.path.join(out, "bars.json"), "w", encoding="utf-8") as f:
        json.dump({"stock": stock_id, "bars": bars}, f, ensure_ascii=False)
    model_json = {
        "stock": stock_id, "name": stock["name"], "dim": D, "levels": levels, "sessions": sessions,
        "ladder": model.LADDER_NAME, "edges": edges.tolist(),
        "medians": {str(s): meds[s] for s in sessions},
        "score": "정수 프로토타입 코사인(자기 제외). 이진 점수는 비교용",
        "thresholds": {"alert": round(alert, 4), "watch": round(watch, 4),
                       "alert_bin": round(alert_b, 4), "watch_bin": round(watch_b, 4),
                       "alert_percentile": vc["alert_percentile"], "watch_percentile": vc["watch_percentile"]},
        "trained": {"from": dates[0], "end": dates[-1], "days": len(dates), "bars": mem.n,
                    "at": now.isoformat(timespec="seconds")},
        "dates": dates,
        "history": [{"at": now.isoformat(timespec="seconds"), "kind": "학습", "added": mem.n, "n_after": mem.n}],
        "memory": mem.to_json(),
        "memory_train": mem.to_json(),  # 학습 때 그대로. 마감 갱신은 memory에만 더한다(화면의 학습 기간 대조용)
    }
    with open(os.path.join(out, "model.json"), "w", encoding="utf-8") as f:
        json.dump(model_json, f, ensure_ascii=False, separators=(",", ":"))
    t.save(trace_path)
    append_run({"at": now.isoformat(timespec="seconds"), "kind": "volume.train", "subject": stock_id,
                "fetched": t.data["log"]["fetched"],
                "failures": [f"{x['what']}: {x['detail']}" for x in t.data["log"]["failures"]],
                "trace": rel(trace_path)})
    return t.data, ranked, big


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--raw", default=None)
    a = ap.parse_args()
    cfg = json.load(open(CONFIG, encoding="utf-8"))
    ids = [a.stock] if a.stock else [s["id"] for s in cfg["stocks"]]
    for sid in ids:
        try:
            tr, ranked, big = run(sid, a.end, a.raw)
        except Exception:
            traceback.print_exc()
            continue
        for s in tr["steps"]:
            c = s["check"]
            print(f"{s['index']:>2}. {s['name']} [{s['op']}]" +
                  (f"  정답={c['count_answer']} HDC={c['hdc_answer']}" if c else ""))
        print("상위 10:")
        for b in ranked[:10]:
            print(f"  {b['rank']:>3} {b['date']} {b['weekday']} {b['slot']} {b['volume']:>8,}주 "
                  f"{b['ratio']:7.2f}배 {b['level']:>2}단계 점수 {b['score']:.4f} {b['verdict']}")
        print("16배 이상 봉 순위:", sorted(b["rank"] for b in big))


if __name__ == "__main__":
    main()
