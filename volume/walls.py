"""매물벽 찾기: 어느 가격에 닿으면 밀려 내려오는가.

일봉으로 본다. 가격을 2% 폭의 칸으로 나누고, 날마다 주가가 닿은 칸마다
'가격칸 ⊕ 결과(밀림·안착·돌파) ⊕ 거래량 단계'를 기억에 더한다.
기억에 '밀림 ⊕ 큰 거래량'으로 풀어 물으면 그런 일이 잦았던 가격칸이 떠오른다.

실행: python -m volume.walls [--stock 045510.KQ]
결과: docs/data/volume/<종목>/walls.json, traces/walls-*.json, docs/data/runs.json
"""

import argparse
import datetime as dt
import json
import math
import os
import traceback

import numpy as np

from hdc import D, Memory, bind, level_vectors, make_vector, noise_level, similarity
from hdc.trace import REPO_ROOT, Trace, check, vec_info, why
from volume import data, model
from volume.train import CONFIG, append_run, rel, stock_dir

OUTCOMES = ("밀림", "안착", "돌파")


# ---------- 가격칸 ----------

def cell_of(price: float, width: float) -> int:
    """가격 → 가격칸 번호. 칸 k = [(1+폭)^k, (1+폭)^(k+1)). 기준이 고정이라 실행마다, 종목마다 칸 경계가 같다."""
    return math.floor(math.log(price) / math.log(1 + width))


def cell_range(k: int, width: float) -> tuple[float, float]:
    return (1 + width) ** k, (1 + width) ** (k + 1)


def cell_label(k: int, width: float) -> str:
    lo, hi = cell_range(k, width)
    return f"{lo:,.0f}-{hi:,.0f}원"


def cell_name(k: int) -> str:
    return f"가격칸:{k}"


def outcome_name(o: str) -> str:
    return f"결과:{o}"


# ---------- 하루를 (칸, 결과) 목록으로 ----------

def touches(day: dict, width: float):
    """그날 저가-고가가 걸친 칸마다 결과를 매긴다.
    밀림: 종가가 그 칸 아래.  돌파: 종가가 그 칸 위.  안착: 종가가 그 칸 안."""
    lo_k, hi_k = cell_of(day["low"], width), cell_of(day["high"], width)
    out = []
    for k in range(lo_k, hi_k + 1):
        lo, hi = cell_range(k, width)
        if day["close"] < lo:
            o = "밀림"
        elif day["close"] >= hi:
            o = "돌파"
        else:
            o = "안착"
        out.append((k, o))
    return out


def volume_levels(days, lookback: int, edges):
    """그날 거래량 ÷ 직전 lookback거래일 거래량 중앙값 → 12단계. 직전 자료가 부족하면 None."""
    for i, d in enumerate(days):
        prev = [x["volume"] for x in days[max(0, i - lookback):i] if x["volume"] > 0]
        if len(prev) < lookback // 2:
            d["vol_ratio"], d["level"] = None, None
            continue
        med = float(np.median(prev))
        d["vol_ratio"] = d["volume"] / med if med else None
        d["level"] = data.level_of(d["vol_ratio"], edges) if med else None
    return days


# ---------- HDC ----------

class WallCodebook:
    def __init__(self, cells, levels: int, dim: int):
        self.dim = dim
        self.cells = {k: make_vector(cell_name(k), dim) for k in cells}
        self.outcomes = {o: make_vector(outcome_name(o), dim) for o in OUTCOMES}
        self.levels, _ = level_vectors(model.LADDER_NAME, levels, dim)  # 시간대 모델과 같은 사다리

    def level(self, k: int) -> np.ndarray:
        return self.levels[k - 1]


def touch_vector(cb: WallCodebook, cell: int, outcome: str, level: int) -> np.ndarray:
    """닿음 하나 = 가격칸 ⊕ 결과 ⊕ 거래량 단계."""
    return bind(bind(cb.cells[cell], cb.outcomes[outcome]), cb.level(level))


def ask_walls(cb: WallCodebook, memory_vec: np.ndarray, outcome: str, level: int):
    """기억 ⊕ 결과 ⊕ 단계 → 남은 것과 각 가격칸 벡터의 유사도.
    그 결과·그 거래량 단계로 닿은 일이 잦았던 칸일수록 높다."""
    q = bind(bind(memory_vec, cb.outcomes[outcome]), cb.level(level))
    return q, {k: similarity(q, v) for k, v in cb.cells.items()}


def count_weight(level: int, query_level: int, levels: int) -> float:
    """같은 질문의 단순 카운트판에서 한 번의 밀림이 받는 무게 = 두 단계 벡터의 유사도와 같은 꼴."""
    return 1 - abs(level - query_level) / (levels - 1)


# ---------- 실행 ----------

def fetch_daily(ticker: str):
    import yfinance as yf

    d = yf.Ticker(ticker).history(period="5y", interval="1d", auto_adjust=False)
    if d.empty:
        raise RuntimeError("빈 응답: 일봉 0행")
    return [{"date": ts.strftime("%Y-%m-%d"), "open": float(r.Open), "high": float(r.High),
             "low": float(r.Low), "close": float(r.Close), "volume": int(r.Volume)}
            for ts, r in zip(d.index, d.itertuples())]


def analyze_window(t: Trace, cb_all, days, n: int, wc: dict, levels: int):
    width, ql = wc["cell_width"], wc["query_level"]
    win = [d for d in days if d["level"] is not None][-n:]
    t.section(f"최근 {n}거래일", f"{win[0]['date']} - {win[-1]['date']}. 이 기간 일봉만으로 기억을 만들고 묻는다.")

    rows = [(d, k, o) for d in win for k, o in touches(d, width)]
    cells = sorted({k for _, k, _ in rows})
    t.step("날마다 닿은 가격칸과 결과 매기기", "일반 계산", [
        why("설계 선택", f"그날 저가부터 고가까지 걸친 가격칸마다, 종가가 그 칸 아래면 '밀림', 위면 '돌파', 안이면 '안착'으로 매긴다. "
                       f"칸 폭은 {width:.0%}다."),
        why("설계 선택", "장중에 그 가격 위까지 갔다가 아래로 마감한 날도 '밀림'이다. 그 가격 위에서 버티지 못했다는 뜻이기 때문이다."),
        why("수학적 사실", "거래량만으로는 누가 팔았는지 알 수 없다. 여기서 보는 것은 '그 가격에서 밀려 마감한 날이 잦았고, 그때 거래가 많았나'다."),
    ], outputs=[{"type": "value", "name": "닿음 수(날 × 칸)", "value": len(rows)},
                {"type": "value", "name": "닿은 가격칸 수", "value": len(cells)},
                {"type": "table", "name": "최근 5일",
                 "columns": ["날짜", "시가", "고가", "저가", "종가", "거래량 배율", "단계", "닿은 칸과 결과"],
                 "rows": [[d["date"], round(d["open"]), round(d["high"]), round(d["low"]), round(d["close"]),
                           round(d["vol_ratio"], 2), d["level"],
                           ", ".join(f"{cell_label(k, width)} {o}" for k, o in touches(d, width))] for d in win[-5:]]}],
       source=touches)

    ex_d, ex_k, ex_o = max((r for r in rows if r[2] == "밀림"), key=lambda r: r[0]["level"], default=rows[-1])
    cv, ov, lv = cb_all.cells[ex_k], cb_all.outcomes[ex_o], cb_all.level(ex_d["level"])
    tv = touch_vector(cb_all, ex_k, ex_o, ex_d["level"])
    t.step("닿음 하나를 벡터 하나로 묶기", "묶기", [
        why("설계 선택", "닿음 하나 = 가격칸 ⊕ 결과 ⊕ 거래량 단계. 가격칸과 결과는 서로 무관한 벡터, 거래량 단계는 이웃끼리 닮은 수준 벡터다."),
        why("이 데이터에서 관찰", "가격칸까지 수준 벡터로 만들면 묶은 벡터들이 한 중심으로 뭉쳐 칸이 구분되지 않았다(2026-11-11 매물대 첫 시도). "
                               "묶은 벡터끼리의 유사도가 재료 유사도의 곱이기 때문이다."),
    ], inputs=[{"type": "value", "name": "예시(거래량 단계가 가장 큰 밀림)",
                "value": f"{ex_d['date']} 고가 {ex_d['high']:,.0f}원 종가 {ex_d['close']:,.0f}원, {cell_label(ex_k, width)} {ex_o}, "
                         f"거래량 평소의 {ex_d['vol_ratio']:.1f}배({ex_d['level']}단계)"},
               vec_info(cell_name(ex_k), cv, full=True), vec_info(outcome_name(ex_o), ov, full=True),
               vec_info(f"{ex_d['level']}단계", lv, full=True)],
       outputs=[vec_info("닿음 벡터", tv, full=True)],
       check=check("닿음 벡터 ⊕ 결과 ⊕ 단계가 가격칸 벡터와 같은가", True,
                   bool(np.array_equal(bind(bind(tv, ov), lv), cv))),
       source=touch_vector)

    mem = Memory(cb_all.dim)
    for d, k, o in rows:
        mem.add(touch_vector(cb_all, k, o, d["level"]))
    mv = mem.to_vector()
    t.step("닿음을 모두 기억에 더하기", "더하기", [
        why("설계 선택", f"이 기간의 닿음 {mem.n}개를 카운터 하나에 더한다. 기간마다 기억을 따로 만든다."),
        why("수학적 사실", "다수결 기억에 담을 수 있는 양에는 한계가 있다. 더한 수가 많을수록 하나하나의 몫이 작아져 잡음에 묻히기 쉽다."),
    ], outputs=[{"type": "value", "name": "더한 닿음 수", "value": mem.n},
                {"type": "value", "name": "동점(0) 칸 수", "value": mem.tie_count()},
                vec_info("기억(다수결)", mv, full=True)],
       check=check("더한 수 = 닿음 수", len(rows), mem.n),
       source=Memory.add)

    q, sims = ask_walls(cb_all, mv, "밀림", ql)
    noise = noise_level(cb_all.dim)
    table = []
    for k in cells:
        here = [(d, o) for d, kk, o in rows if kk == k]
        pushed = [d for d, o in here if o == "밀림"]
        broke = [d for d, o in here if o == "돌파"]
        weight = sum(count_weight(d["level"], ql, levels) for d in pushed)
        vr = [d["vol_ratio"] for d in pushed]
        table.append({"cell": k, "label": cell_label(k, width), "touch": len(here), "pushed": len(pushed),
                      "broke": len(broke), "vol_ratio_median": round(float(np.median(vr)), 2) if vr else None,
                      "count_weight": round(weight, 3), "hdc": round(sims[k], 4),
                      "dates": [d["date"] for d in pushed]})
    hdc_order = sorted(table, key=lambda r: -r["hdc"])
    cnt_order = sorted(table, key=lambda r: -r["count_weight"])
    top = wc["top"]
    overlap = len({r["cell"] for r in hdc_order[:top]} & {r["cell"] for r in cnt_order[:top]})
    rho = model.spearman([r["hdc"] for r in table], [r["count_weight"] for r in table])
    for i, r in enumerate(hdc_order):
        r["hdc_rank"] = i + 1
    for i, r in enumerate(cnt_order):
        r["count_rank"] = i + 1
    t.step("기억에 '밀림 ⊕ 큰 거래량'으로 묻기", "비교", [
        why("수학적 사실", f"기억 ⊕ 밀림 ⊕ {ql}단계를 하면, 밀림으로 더했던 닿음들은 '가격칸 ⊕ (그날 단계와 {ql}단계의 차이)'로 남고 "
                       "다른 결과로 더했던 것은 잡음이 된다. 남은 것을 가격칸 벡터마다 비교하면, 그 칸에서 밀린 날이 많고 "
                       f"그날 거래량 단계가 {ql}단계에 가까울수록 유사도가 높다."),
        why("설계 선택", f"묻는 단계를 {ql}단계(가장 큰 쪽)로 둔 까닭: 수준 벡터 유사도가 단계에 정비례하므로, "
                       "밀린 날 하나하나를 '거래량 단계가 클수록 크게' 세는 것과 같아진다."),
        why("설계 선택", f"단순 카운트 정답 = 그 칸에서 밀린 날마다 (1 - |단계 - {ql}| / {levels - 1})를 더한 값. HDC 유사도가 근사하려는 바로 그 양이다."),
        why("수학적 사실", f"한 번도 밀린 적 없는 칸의 유사도는 0 근처에서 약 {noise:.2f} 흔들린다."),
    ], outputs=[{"type": "table", "name": "가격칸별 결과(HDC 순)",
                 "columns": ["HDC 순위", "가격칸", "닿은 날", "밀린 날", "돌파한 날", "밀린 날 거래량 배율(중앙값)",
                             "HDC 유사도", "카운트 정답", "카운트 순위", "밀린 날짜"],
                 "rows": [[r["hdc_rank"], r["label"], r["touch"], r["pushed"], r["broke"], r["vol_ratio_median"],
                           r["hdc"], r["count_weight"], r["count_rank"], r["dates"]] for r in hdc_order]},
                {"type": "value", "name": "잡음 수준(1/√D)", "value": noise},
                {"type": "value", "name": "HDC 순서와 카운트 순서의 순위 상관", "value": round(rho, 3)}],
       check=check(f"HDC 상위 {top}칸과 카운트 상위 {top}칸이 겹치는 수", top, overlap),
       source=ask_walls)
    return {"days": n, "from": win[0]["date"], "to": win[-1]["date"], "touches": len(rows), "rho": round(rho, 3),
            "overlap": overlap, "top": top, "cells": hdc_order}


def run(stock_id: str):
    cfg = json.load(open(CONFIG, encoding="utf-8"))
    stock = next(s for s in cfg["stocks"] if s["id"] == stock_id)
    vc, wc = cfg["volume"], cfg["walls"]
    levels = vc["levels"]
    edges = 2.0 ** np.array(vc["ratio_edge_exponents"], dtype=float)
    now = dt.datetime.now(data.KST)
    out = stock_dir(stock_id)
    trace_path = os.path.join(out, "traces", f"walls-{now.strftime('%Y%m%d-%H%M%S')}.json")
    t = Trace("volume.walls", subject={"type": "stock", "id": stock_id, "code": stock["code"], "name": stock["name"]},
              params={"D": D, "cell_width": wc["cell_width"], "windows": wc["windows"], "query_level": wc["query_level"],
                      "volume_lookback": wc["volume_lookback"]})
    t.section("공통: 일봉과 거래량 단계")
    try:
        days = fetch_daily(stock_id)
    except Exception as e:
        t.failure("yfinance 일봉 수집", f"{type(e).__name__}: {e}")
        t.save(trace_path)
        append_run({"at": now.isoformat(timespec="seconds"), "kind": "volume.walls", "subject": stock_id,
                    "fetched": {}, "failures": [f"수집 실패: {e}"], "trace": rel(trace_path)})
        raise
    days = [d for d in days if d["volume"] > 0 and d["high"] > 0]
    complete = [d for d in days if d["date"] < now.strftime("%Y-%m-%d") or now.hour >= 16]
    t.fetched("일봉", len(days))
    t.step("일봉 받기", "일반 계산", [
        why("설계 선택", "매물벽은 일봉으로 본다. 일봉은 5년치가 있고, 시간봉의 9시대 누락 문제가 없다."),
        why("설계 선택", "거래가 없던 날(거래량 0)은 뺀다. 오늘 자료는 장 마감 뒤(16시 이후)에만 쓴다."),
    ], outputs=[{"type": "value", "name": "받은 일봉", "value": len(days)},
                {"type": "value", "name": "기간", "value": f"{days[0]['date']} - {days[-1]['date']}"},
                {"type": "value", "name": "쓰는 일봉", "value": len(complete)}],
       source=fetch_daily)

    volume_levels(complete, wc["volume_lookback"], edges)
    t.step("날마다 거래량 단계 매기기", "일반 계산", [
        why("설계 선택", f"그날 거래량 ÷ 직전 {wc['volume_lookback']}거래일 거래량 중앙값을 시간대 모델과 같은 12단계로 나눈다. "
                       "'평소'가 시기마다 달라지므로 직전 기간과 비교한다."),
    ], outputs=[{"type": "table", "name": "최근 10일", "columns": ["날짜", "거래량", "평소 대비 배율", "단계"],
                 "rows": [[d["date"], d["volume"], round(d["vol_ratio"], 2) if d["vol_ratio"] else None, d["level"]]
                          for d in complete[-10:]]}],
       source=volume_levels)

    width = wc["cell_width"]
    last_n = max(wc["windows"])
    usable = [d for d in complete if d["level"] is not None][-last_n:]
    all_cells = sorted({k for d in usable for k, _ in touches(d, width)})
    cb = WallCodebook(all_cells, levels, D)
    ex = cell_of(usable[-1]["close"], width)
    t.step("가격칸 나누기와 벡터 꺼내기", "벡터 생성", [
        why("설계 선택", f"가격칸 k = [(1 + {width})^k, (1 + {width})^(k+1))원. 칸 폭은 어디서나 {width:.0%}이고, "
                       "기준이 고정이라 실행마다·종목마다 칸 경계가 같다. 예: 1,200원은 "
                       f"{cell_label(cell_of(1200, width), width)} 칸이다."),
        why("설계 선택", "가격칸마다 '가격칸:k' 이름에서 서로 무관한 벡터를, 결과(밀림·안착·돌파)마다 '결과:…' 벡터를 꺼낸다."),
    ], outputs=[{"type": "value", "name": "쓰이는 가격칸 수", "value": len(all_cells)},
                {"type": "value", "name": "가격칸 범위", "value": f"{cell_label(all_cells[0], width)} - {cell_label(all_cells[-1], width)}"},
                vec_info(cell_name(ex), cb.cells[ex], full=True)]
               + [vec_info(outcome_name(o), cb.outcomes[o], full=True) for o in OUTCOMES],
       check=check("결과 벡터 3쌍의 |유사도|가 모두 0.03 이하인가", True,
                   all(abs(similarity(cb.outcomes[a], cb.outcomes[b])) <= 0.03
                       for i, a in enumerate(OUTCOMES) for b in OUTCOMES[i + 1:])),
       source=WallCodebook)

    results = [analyze_window(t, cb, complete, n, wc, levels) for n in sorted(wc["windows"])]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "walls.json"), "w", encoding="utf-8") as f:
        json.dump({"stock": stock_id, "name": stock["name"], "at": now.isoformat(timespec="seconds"),
                   "cell_width": width, "query_level": wc["query_level"], "windows": results,
                   "trace": rel(trace_path)}, f, ensure_ascii=False)
    with open(os.path.join(out, "raw_daily.json"), "w", encoding="utf-8") as f:
        json.dump({"fetched_at": now.isoformat(timespec="seconds"), "daily": days}, f, ensure_ascii=False)
    t.save(trace_path)
    append_run({"at": now.isoformat(timespec="seconds"), "kind": "volume.walls", "subject": stock_id,
                "fetched": t.data["log"]["fetched"],
                "failures": [f"{x['what']}: {x['detail']}" for x in t.data["log"]["failures"]],
                "trace": rel(trace_path)})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock", default=None)
    a = ap.parse_args()
    cfg = json.load(open(CONFIG, encoding="utf-8"))
    for sid in ([a.stock] if a.stock else [s["id"] for s in cfg["stocks"]]):
        try:
            for w in run(sid):
                print(f"\n최근 {w['days']}거래일 ({w['from']} - {w['to']}), 닿음 {w['touches']}개, "
                      f"순위 상관 {w['rho']}, 상위 {w['top']} 겹침 {w['overlap']}")
                for r in w["cells"][:8]:
                    print(f"  HDC {r['hdc_rank']:>2} | {r['label']:>13} 닿음 {r['touch']:>2} 밀림 {r['pushed']:>2} "
                          f"돌파 {r['broke']:>2} 배율 {r['vol_ratio_median']} | HDC {r['hdc']:.4f} 카운트 {r['count_weight']} ({r['count_rank']}위)")
        except Exception:
            traceback.print_exc()


if __name__ == "__main__":
    main()
