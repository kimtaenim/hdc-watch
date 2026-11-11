"""거래량 모델 학습: 최근 반년 봉을 모두 더해 기억을 만들고, 그 반년을 다시 채점한다.

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
    sessions, levels, dsess = vc["sessions"], vc["levels"], vc["derived_session"]
    now = dt.datetime.now(data.KST)
    stamp = now.strftime("%Y%m%d-%H%M%S")
    out = stock_dir(stock_id)
    trace_path = os.path.join(out, "traces", f"train-{stamp}.json")

    t = Trace("volume.train", subject={"type": "stock", "id": stock_id, "code": stock["code"],
                                       "name": stock["name"]},
              params={"D": D, "levels": levels, "sessions": sessions, "train_months": vc["train_months"],
                      "ratio_edge_exponents": vc["ratio_edge_exponents"]})

    t.section("공통: 자료와 거래량 단계", "HDC 연산 없이 숫자를 이름표로 바꾸는 앞단.")
    # ① 원자료 받기
    try:
        if raw_path:
            raw = json.load(open(raw_path, encoding="utf-8"))
            hourly, daily, fetched_at = raw["hourly"], raw["daily"], raw.get("fetched_at")
            closes = raw.get("closes", {})
            t.note(f"저장된 원자료 사용: {raw_path}")
        else:
            hourly, daily, closes = data.fetch(stock_id)
            fetched_at = now.isoformat(timespec="seconds")
    except Exception as e:
        t.failure("저장된 원자료 읽기" if raw_path else "yfinance 수집", f"{type(e).__name__}: {e}")
        t.save(trace_path)
        append_run({"at": now.isoformat(timespec="seconds"), "kind": "volume.train", "subject": stock_id,
                    "fetched": {}, "failures": [f"수집 실패: {e}"], "trace": rel(trace_path)})
        raise
    t.fetched("시간봉", len(hourly))
    t.fetched("일봉", len(daily))
    days = data.complete_days(hourly, daily, now)
    end = end or days[-1]
    dates, start = data.window(days, end, vc["train_months"])
    t.step("원자료 받기", "일반 계산", [
        why("설계 선택", "yfinance(야후 금융의 비공식 경로)로 1시간봉과 일봉을 받는다. 비공식이라 실패하거나 값이 바뀔 수 있어, "
                       "받은 원자료를 raw.json에 그대로 남긴다."),
        why("설계 선택", f"학습 기간은 {start} 다음 날부터 {end}까지 {vc['train_months']}개월. 일별 거래량이 확정된 날만 쓴다."),
    ], outputs=[{"type": "value", "name": "받은 시간봉", "value": len(hourly)},
                {"type": "value", "name": "받은 일봉", "value": len(daily)},
                {"type": "value", "name": "학습 기간", "value": f"{dates[0]} - {dates[-1]}"},
                {"type": "value", "name": "학습 거래일 수", "value": len(dates)},
                {"type": "value", "name": "받은 시각", "value": fetched_at}],
       source=data.fetch)

    # ② 하루 봉 만들기 (15시대 역산)
    zero_missing = vc.get("zero_is_missing_sessions", [])
    hsess = vc.get("hourly_sessions") or [s for s in sessions if s != dsess]
    regular_s = [s for s in sessions if s != dsess]
    span = f"{regular_s[0]}-{regular_s[-1]}시대"
    hspan = f"{hsess[0]}-{hsess[-1]}시대"
    skipped = [s for s in hsess if s not in sessions]
    bars, problems = data.build_bars(hourly, daily, dates, sessions, dsess, zero_missing, hsess)
    derived = [b for b in bars if b["source"] == "역산"]
    ok = [b for b in derived if b["missing"] is None]
    sums_ok = sum(1 for b in ok if b["volume"] + b["hourly_sum"] == b["daily"])
    for p in problems:
        t.failure("봉 만들기", f"{p['date']}: {p['reason']}")
    zero_days = sorted({p["date"] for p in problems if "0주" in p["reason"]})
    n_zero = len(zero_days)
    after_break = sum(1 for d in zero_days if data.after_break(d, days))
    n_neg = sum(1 for b in derived if b["missing"] and "음수" in b["missing"])
    t.step(f"하루 {len(sessions)}개 봉 만들기", "일반 계산", [
        why("설계 선택", f"{span} {len(regular_s)}개는 시간봉 그대로 쓰고, {dsess}시대(마감 동시호가) 봉은 시간봉에 없어 "
                       f"'일별 거래량 - {hspan} 시간봉 {len(hsess)}개 합'으로 역산한다. 역산값이 음수면 그 봉은 결측으로 둔다."),
        why("설계 선택", f"{', '.join(map(str, skipped))}시대는 봉으로 만들지 않는다(2026-11-11 의장님 결정). "
                       "다만 역산할 때는 빼야 그 거래량이 역산 봉에 섞이지 않는다.") if skipped else
        why("설계 선택", "모든 시간봉 시간대를 봉으로 만든다."),
        why("이 데이터에서 관찰",
            f"{len(dates)}일 중 {n_zero}일은 {', '.join(map(str, zero_missing))}시대 시간봉이 0주였다. 그중 {after_break}일이 쉬는 날 바로 다음 거래일이고, "
            "같은 날 30분봉·5분봉에는 9시대 거래가 있다(2026-11-11 최근 60일 대조). 그래서 그날 역산 봉을 결측으로 둔다."),
        why("이 데이터에서 관찰",
            f"{n_neg}일은 시간봉 {len(hsess)}개 합이 일별 거래량보다 커서 역산할 수 없었다. 원인은 확인하지 못했다."),
    ], outputs=[{"type": "value", "name": "만든 봉", "value": len(bars)},
                {"type": "value", "name": f"역산한 {dsess}시대 봉", "value": len(ok)},
                {"type": "table", "name": "결측으로 둔 봉", "columns": ["날짜", "시간대", "이유"],
                 "rows": [[b["date"], b["hour"], b["missing"]] for b in bars if b["missing"]]},
                {"type": "table", "name": "역산 예시(최근 5일)", "columns": ["날짜", "일별", f"시간봉 {len(hsess)}개 합", "역산"],
                 "rows": [[b["date"], b["daily"], b["hourly_sum"], b["volume"]] for b in ok[-5:]]}],
       check=check(f"역산 {dsess}시대 + 시간봉 {len(hsess)}개 = 일별 거래량인 날 수", len(ok), sums_ok),
       source=data.build_bars)

    # ③ 시간대 중앙값
    meds = data.session_medians(bars, sessions)
    t.step("시간대별 평소 거래량(중앙값)", "일반 계산", [
        why("설계 선택", "거래량은 절대량이 아니라 '그 시간대 중앙값의 몇 배인가'로 본다. 장 시작과 마감 무렵은 원래 많고 점심 무렵은 원래 적기 때문이다."),
        why("수학적 사실", "중앙값은 한두 번의 폭증에 끌려가지 않는다. 평균은 끌려간다."),
    ], outputs=[{"type": "table", "name": "시간대별 중앙값", "columns": ["시간대", "중앙값(주)", "봉 수"],
                 "rows": [[s, meds[s]["median"], meds[s]["n"]] for s in sessions]}],
       source=data.session_medians)

    # ④ 단계 나누기
    edges = 2.0 ** np.array(vc["ratio_edge_exponents"], dtype=float)
    data.assign_levels(bars, meds, edges)
    regular = [b["level"] for b in bars if b["hour"] != dsess and b["level"]]
    last = [b["level"] for b in bars if b["hour"] == dsess and b["level"]]
    dist_reg = np.bincount(regular, minlength=levels + 1)[1:].tolist()
    dist_last = np.bincount(last, minlength=levels + 1)[1:].tolist()
    bounds_txt = ["0"] + [f"{e:g}" for e in edges] + ["∞"]
    t.step("거래량 단계 나누기", "일반 계산", [
        why("설계 선택", f"배율의 경계를 2의 {vc['ratio_edge_exponents'][0]}제곱부터 {vc['ratio_edge_exponents'][-1]}제곱까지 2배 간격으로 두어 "
                       f"{levels}단계로 나눈다. 6단계가 평소(1배-2배)다. 경계값은 위 단계에 넣는다."),
        why("설계 선택", "2배 간격은 '평소의 2배'와 '평소의 4배'를 같은 거리로 본다는 뜻이다(로그 눈금)."),
    ], outputs=[{"type": "table", "name": "단계별 배율 구간과 봉 수",
                 "columns": ["단계", "배율 구간", f"{span} 봉", f"{dsess}시대 봉"],
                 "rows": [[k + 1, f"{bounds_txt[k]} - {bounds_txt[k + 1]}", dist_reg[k], dist_last[k]]
                          for k in range(levels)]}],
       check=check(f"{span} 봉 수 = 단계별 봉 수 합", sum(1 for b in bars if b["hour"] != dsess and b["level"]),
                   sum(dist_reg)),
       source=data.level_of)

    # ⑤ 기본 벡터 꺼내기
    t.section("시간대 모델", "봉 하나 = 요일 ⊕ 시간대 ⊕ 거래량 단계. '이 요일·시간대에 이 거래량이 흔했나'를 묻는다.")
    cb = model.Codebook(sessions, levels, D)
    sims_ds = [similarity(a, b) for i, a in enumerate(list(cb.days.values()) + list(cb.sessions.values()))
               for j, b in enumerate(list(cb.days.values()) + list(cb.sessions.values())) if i < j]
    n_base = len(cb.days) + len(cb.sessions)
    n_pairs = n_base * (n_base - 1) // 2
    t.step(f"요일 벡터 {len(cb.days)}개, 시간대 벡터 {len(cb.sessions)}개 꺼내기", "벡터 생성", [
        why("설계 선택", "요일과 시간대는 서로 독립인 무작위 벡터다. 월요일과 화요일이 가깝다는 식의 순서 정보는 넣지 않았다."),
        why("수학적 사실", f"이름이 다르면 벡터는 서로 무관하다. {n_base}개 벡터의 {n_pairs}쌍 유사도는 0 근처이고 흔들림은 약 {noise_level(D):.2f}이다."),
    ], outputs=[vec_info(model.day_name(k), v, full=True) for k, v in cb.days.items()]
               + [vec_info(model.session_name(k), v, full=True) for k, v in cb.sessions.items()]
               + [{"type": "value", "name": f"{n_pairs}쌍 유사도 범위",
                   "value": [round(min(sims_ds), 4), round(max(sims_ds), 4)]}],
       check=check(f"{n_pairs}쌍 모두 |유사도| ≤ 0.03인가", True, bool(max(abs(s) for s in sims_ds) <= 0.03)),
       source=model.Codebook)

    lv_sims = [round(similarity(cb.level(1), cb.level(k)), 4) for k in range(1, levels + 1)]
    t.step("수준 벡터 12단계 만들기", "뒤집기", [
        why("설계 선택", f"1단계는 '{model.LADDER_NAME}'에서 꺼낸 벡터. 한 단계 오를 때마다 아직 안 뒤집은 칸을 약 {D // 2 // (levels - 1)}개씩 뒤집는다."),
        why("수학적 사실", "이웃 단계는 비슷하고 먼 단계는 무관하다. 1단계와 12단계는 정확히 절반이 달라 유사도 0이다. "
                       "이 성질 때문에 '7단계 봉'은 기억 속 '6단계 봉'과도 꽤 닮은 것으로 계산된다."),
    ], outputs=[vec_info(f"{k}단계", cb.level(k), full=True) for k in range(1, levels + 1)]
               + [{"type": "table", "name": "단계마다 뒤집은 칸 수", "columns": ["구간", "칸 수"],
                   "rows": [[f"{k + 1}→{k + 2}", len(f)] for k, f in enumerate(cb.flips)]},
                  {"type": "table", "name": "1단계와의 유사도", "columns": ["단계", "유사도"],
                   "rows": [[k, s] for k, s in zip(range(1, levels + 1), lv_sims)]}],
       check=check("1단계와 12단계의 다른 칸 수", D // 2, hamming(cb.level(1), cb.level(levels))),
       source=model.Codebook)

    # ⑥ 기억에 더하기
    used = [b for b in bars if b["level"] is not None]
    mem = model.train(cb, used)
    mv = mem.to_vector()

    # ⑦ 다시 채점 (자기 자신 빼고)
    vecs = [model.bar_vector(cb, b["weekday"], b["hour"], b["level"]) for b in used]
    for b, v in zip(used, vecs):
        b["score"] = round(model.score_leave_one_out(mem, v), 4)
        b["score_in"] = round(model.score(mv, v), 4)
        cl, same, n = model.count_closeness(used, b, levels)
        b["count_closeness"] = round(cl, 4) if cl is not None else None
        b["count_same"], b["count_peers"] = same, n
    scores = np.array([b["score"] for b in used])
    alert = float(np.percentile(scores, vc["alert_percentile"]))
    watch = float(np.percentile(scores, vc["watch_percentile"]))
    ranked = sorted(used, key=lambda b: b["score"])
    for i, b in enumerate(ranked):
        b["rank"] = i + 1
        b["verdict"] = "이상" if b["score"] <= alert else "주의" if b["score"] <= watch else "평소"

    # 따라가기 예시: 가장 이상한 봉
    ex = ranked[0]
    dv, sv, lv = cb.days[ex["weekday"]], cb.sessions[ex["hour"]], cb.level(ex["level"])
    ds = bind(dv, sv)
    bv = bind(ds, lv)
    back = nearest(bind(bv, ds), {f"{k}단계": cb.level(k) for k in range(1, levels + 1)})
    t.step("봉 하나를 벡터 하나로 묶기", "묶기", [
        why("설계 선택", "봉 하나 = 요일 ⊕ 시간대 ⊕ 단계. 세 이름표를 XOR로 묶어 1만 비트 하나로 만든다."),
        why("수학적 사실", "묶은 벡터는 재료 셋 중 어느 것과도 닮지 않는다(유사도 0 근처). "
                       "그래서 '화요일 13시대 12단계'와 '화요일 13시대 6단계'는 단계 벡터끼리 닮은 만큼만 닮는다."),
        why("수학적 사실", "묶은 벡터를 (요일 ⊕ 시간대)로 다시 XOR하면 단계 벡터가 그대로 나온다."),
    ], inputs=[{"type": "value", "name": "예시 봉",
                "value": f"{ex['date']} ({ex['weekday']}) {ex['hour']}시대 {ex['volume']:,}주, 중앙값의 {ex['ratio']:.1f}배, {ex['level']}단계"},
               vec_info(model.day_name(ex["weekday"]), dv, full=True),
               vec_info(model.session_name(ex["hour"]), sv, full=True),
               vec_info(f"{ex['level']}단계", lv, full=True)],
       outputs=[vec_info(f"{model.day_name(ex['weekday'])} ⊕ {model.session_name(ex['hour'])}", ds, full=True),
                vec_info("봉 벡터", bv, full=True),
                {"type": "value", "name": "봉 벡터와 요일 벡터의 유사도", "value": round(similarity(bv, dv), 4)},
                {"type": "value", "name": "봉 벡터와 단계 벡터의 유사도", "value": round(similarity(bv, lv), 4)}],
       check=check("봉 벡터 ⊕ (요일 ⊕ 시간대)에 가장 가까운 단계", f"{ex['level']}단계", back[0][0]),
       source=model.bar_vector)

    hist_counts = np.bincount(np.abs(mem.counts))
    t.step("모든 봉을 기억에 더하기", "더하기", [
        why("설계 선택", f"학습 기간의 봉 {mem.n}개(결측 제외)를 칸마다 +1/-1로 누적한다. 이 카운터 1만 개가 모델의 전부다."),
        why("수학적 사실", "새 봉이 오면 카운터에 그 봉의 +1/-1을 더하기만 하면 된다. 다시 학습할 필요가 없다. 순서를 바꿔 더해도 결과가 같다."),
        why("설계 선택", "비교할 때는 카운터를 다수결로 0/1 벡터로 만든다. 0:0인 칸은 고정된 동점깨기 벡터를 따른다."),
    ], inputs=[{"type": "value", "name": "더한 봉 수", "value": mem.n}],
       outputs=[{"type": "value", "name": "카운터 범위", "value": [int(mem.counts.min()), int(mem.counts.max())]},
                {"type": "table", "name": "카운터 절댓값 분포", "columns": ["|카운터|", "칸 수"],
                 "rows": [[i, int(c)] for i, c in enumerate(hist_counts) if c]},
                {"type": "value", "name": "동점(0) 칸 수", "value": mem.tie_count()},
                vec_info("기억(다수결)", mv, full=True)],
       check=check("더한 봉 수 = 결측 아닌 봉 수", len(used), mem.n),
       source=model.train)

    noise = noise_level(D)
    t.step("반년을 다시 채점하기", "비교", [
        why("설계 선택", "점수 = 봉 벡터와 기억의 유사도. 높으면 그 요일·시간대에 흔하던 단계, 낮으면 드물던 단계다."),
        why("설계 선택", "채점할 때는 그 봉 자신을 기억에서 빼고(카운터에서 그 봉의 +1/-1을 되돌리고) 비교한다. "
                       "앞으로 들어올 새 봉은 채점 시점에 아직 기억에 없으므로, 같은 조건을 맞춘 것이다."),
        why("수학적 사실", f"이 유사도에는 다른 요일·시간대 봉들에서 오는 잡음이 섞이며, 그 흔들림은 약 1/√D = {noise:.3f}이다."),
        why("이 데이터에서 관찰", f"점수는 {scores.min():.4f}부터 {scores.max():.4f}까지 퍼져 있다. "
                               f"잡음 폭 {noise:.3f}은 이 퍼짐(5%-95% 구간 {np.percentile(scores, 5):.4f}-{np.percentile(scores, 95):.4f})에 비해 작지 않다."),
    ], outputs=[{"type": "table", "name": "점수 분포", "columns": ["구간 시작", "봉 수"], "rows": histogram(scores)},
                {"type": "value", "name": "잡음 수준(1/√D)", "value": noise},
                {"type": "table", "name": "봉별 점수",
                 "columns": ["날짜", "요일", "시간대", "거래량", "출처", "배율", "단계", "점수", "자기포함 점수",
                             "카운트 가까움", "같은 단계 수", "같은 칸 봉 수", "순위", "판정"],
                 "rows": [[b["date"], b["weekday"], b["hour"], b["volume"], b["source"], round(b["ratio"], 4),
                           b["level"], b["score"], b["score_in"], b["count_closeness"], b["count_same"],
                           b["count_peers"], b["rank"], b["verdict"]] for b in used]}],
       check=check("점수 순서와 단순 카운트(같은 요일·시간대 봉들과의 단계 가까움) 순서의 순위 상관",
                   1.0, round(model.spearman(scores, [b["count_closeness"] for b in used]), 3)),
       source=model.score_leave_one_out)

    t.step("판정 기준 정하기", "일반 계산", [
        why("설계 선택", f"학습 봉 점수의 하위 {vc['alert_percentile']}%를 '이상', 하위 {vc['watch_percentile']}%를 '주의' 경계로 삼는다. "
                       "기준이 이 종목의 자기 데이터에서 나오므로 종목마다 다르다."),
        why("수학적 사실", f"정의상 학습 봉 {len(used)}개 중 약 {vc['alert_percentile']}%가 '이상'으로, 약 {vc['watch_percentile']}%가 '주의' 이하로 분류된다."),
    ], outputs=[{"type": "value", "name": "이상 경계(점수 이하)", "value": round(alert, 4)},
                {"type": "value", "name": "주의 경계(점수 이하)", "value": round(watch, 4)},
                {"type": "table", "name": "판정별 봉 수", "columns": ["판정", "봉 수"],
                 "rows": [[v, sum(b["verdict"] == v for b in used)] for v in ("이상", "주의", "평소")]}],
       source=run)

    big = [b for b in used if b["hour"] != dsess and b["ratio"] >= 16]
    count_rank = {id(b): i + 1 for i, b in enumerate(sorted(used, key=lambda b: b["count_closeness"]))}
    t.step("가장 이상한 봉 순위", "일반 계산", [
        why("설계 선택", "점수가 낮은 순으로 줄 세운다. 옆에 단순 카운트로 매긴 순위를 함께 둔다."),
        why("이 데이터에서 관찰", "HDC 점수가 묻는 것은 '이 요일·시간대에 이 단계가 흔했나'다. '중앙값의 몇 배인가'와 같은 질문이 아니다. "
                               "그래서 거래량이 0에 가까운 봉(1단계)도 폭증한 봉만큼 낮은 점수를 받고, "
                               "같은 요일·시간대에 큰 봉이 여러 번 있었다면 폭증 봉도 점수가 덜 낮다."),
    ], outputs=[{"type": "table", "name": "상위 30",
                 "columns": ["순위", "날짜", "요일", "시간대", "거래량", "배율", "단계", "점수", "판정", "카운트 순위"],
                 "rows": [[b["rank"], b["date"], b["weekday"], b["hour"], b["volume"], round(b["ratio"], 2),
                           b["level"], b["score"], b["verdict"], count_rank[id(b)]] for b in ranked[:30]]},
                {"type": "table", "name": f"평소의 16배 이상이던 {span} 봉",
                 "columns": ["순위", "날짜", "요일", "시간대", "배율", "단계", "점수", "카운트 순위"],
                 "rows": [[b["rank"], b["date"], b["weekday"], b["hour"], round(b["ratio"], 1), b["level"],
                           b["score"], count_rank[id(b)]] for b in sorted(big, key=lambda b: b["rank"])]}],
       check=check("16배 이상 봉 중 상위 30위 안에 든 수", len(big),
                   sum(1 for b in big if b["rank"] <= 30)),
       source=model.score_leave_one_out)

    # 저장
    os.makedirs(out, exist_ok=True)
    window_set = set(dates)
    with open(os.path.join(out, "raw.json"), "w", encoding="utf-8") as f:
        json.dump({"fetched_at": fetched_at, "stock": stock_id,
                   "hourly": [r for r in hourly if r["date"] in window_set],
                   "daily": {k: v for k, v in daily.items() if k in window_set},
                   "closes": {k: v for k, v in closes.items() if k in window_set}}, f, ensure_ascii=False)
    with open(os.path.join(out, "bars.json"), "w", encoding="utf-8") as f:
        json.dump({"stock": stock_id, "bars": bars}, f, ensure_ascii=False)
    model_json = {
        "stock": stock_id, "name": stock["name"], "dim": D, "levels": levels, "sessions": sessions,
        "ladder": model.LADDER_NAME, "edges": edges.tolist(),
        "medians": {str(s): meds[s] for s in sessions},
        "thresholds": {"alert": round(alert, 4), "watch": round(watch, 4),
                       "alert_percentile": vc["alert_percentile"], "watch_percentile": vc["watch_percentile"]},
        "trained": {"start_after": start, "end": end, "days": len(dates), "bars": mem.n,
                    "at": now.isoformat(timespec="seconds")},
        "history": [{"at": now.isoformat(timespec="seconds"), "kind": "학습", "added": mem.n, "n_after": mem.n}],
        "memory": mem.to_json(),
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
            print(f"  {b['rank']:>3} {b['date']} {b['weekday']} {b['hour']}시대 {b['volume']:>8,}주 "
                  f"{b['ratio']:7.2f}배 {b['level']:>2}단계 점수 {b['score']:.4f} {b['verdict']}")
        print("16배 이상 봉 순위:", sorted(b["rank"] for b in big))


if __name__ == "__main__":
    main()
