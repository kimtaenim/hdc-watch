"""장중 채점과 장 마감 갱신.

장중(30분마다): 오늘 5분봉을 30분 칸으로 합쳐, 끝난 칸은 '확정', 진행 중인 칸은 '잠정'으로 채점한다.
장 마감 뒤: 오늘 칸을 모두 확정 채점하고, 정수 프로토타입에 오늘 봉들을 더한다(덧셈만, 재학습 없음).

실행: python -m volume.live [--stock 045510.KQ] [--at 2026-11-11T11:05:00+09:00] [--raw 저장된5분봉.json]
  --at: 그 시각에 실행한 것처럼 계산한다(그 시각 이후 5분봉은 쓰지 않는다). 확인용.
"""

import argparse
import datetime as dt
import json
import os
import traceback

import numpy as np

from hdc import Memory
from hdc.trace import Trace, check, vec_info, why
from volume import data, model
from volume.train import CONFIG, append_run, rel, stock_dir

SLOT_MIN = 30
CLOSE_AFTER = dt.time(15, 30)   # 이 시각 이후 실행은 '장 마감 뒤'로 본다
CRON_MINUTES = (5, 35)          # 워크플로 예약 시각(매시 5분, 35분)


def slot_status(slot: str, now: dt.datetime) -> str:
    """30분 칸의 상태: 끝났으면 '확정', 진행 중이면 '잠정', 아직 시작 전이면 '아직'."""
    h, m = map(int, slot.split(":"))
    start = now.replace(hour=h, minute=m, second=0, microsecond=0)
    end = start + dt.timedelta(minutes=SLOT_MIN)
    if now >= end:
        return "확정"
    return "잠정" if now >= start else "아직"


def scheduled_delay(now: dt.datetime) -> int:
    """가장 가까운 앞선 예약 시각(매시 5분, 35분)에서 몇 분 늦었나."""
    cands = []
    for back in range(0, 2):
        base = (now - dt.timedelta(hours=back)).replace(second=0, microsecond=0)
        for m in CRON_MINUTES:
            c = base.replace(minute=m)
            if c <= now:
                cands.append(c)
    return int((now - max(cands)).total_seconds() // 60)


def fetch_today(ticker: str):
    import yfinance as yf

    m = yf.Ticker(ticker).history(period="5d", interval="5m", auto_adjust=False)
    return [{"date": ts.strftime("%Y-%m-%d"), "time": ts.strftime("%H:%M"), "volume": int(r.Volume),
             "open": float(r.Open), "high": float(r.High), "low": float(r.Low), "close": float(r.Close)}
            for ts, r in zip(m.index.tz_convert("Asia/Seoul"), m.itertuples())]


def run(stock_id: str, at: dt.datetime | None = None, raw_path: str | None = None):
    cfg = json.load(open(CONFIG, encoding="utf-8"))
    stock = next(s for s in cfg["stocks"] if s["id"] == stock_id)
    out = stock_dir(stock_id)
    mpath = os.path.join(out, "model.json")
    mj = json.load(open(mpath, encoding="utf-8"))
    sessions, levels, dim = mj["sessions"], mj["levels"], mj["dim"]
    edges = np.array(mj["edges"])
    real_now = dt.datetime.now(data.KST)
    now = at or real_now
    today = now.strftime("%Y-%m-%d")
    closing = now.time() >= CLOSE_AFTER
    kind = "volume.close" if closing else "volume.live"
    stamp = real_now.strftime("%Y%m%d-%H%M%S")
    trace_path = os.path.join(out, "traces", f"{'close' if closing else 'live'}-{stamp}.json")
    t = Trace(kind, subject={"type": "stock", "id": stock_id, "code": stock["code"], "name": stock["name"]},
              params={"D": dim, "as_of": now.isoformat(timespec="minutes"), "simulated": at is not None})
    delay = None if at else scheduled_delay(real_now)

    # 같은 날 앞선 실행과의 간격: 빠진 실행 추정
    runs_path = os.path.join(os.path.dirname(os.path.dirname(out)), "runs.json")
    prev = [r for r in (json.load(open(runs_path, encoding="utf-8")) if os.path.exists(runs_path) else [])
            if r.get("subject") == stock_id and r.get("kind") in ("volume.live", "volume.close") and r["at"][:10] == today]
    gap_note = None
    if prev and not at:
        last = dt.datetime.fromisoformat(prev[-1]["at"])
        gap = (real_now - last).total_seconds() / 60
        if gap > 45:
            gap_note = f"앞선 실행({prev[-1]['at'][11:16]})과 {gap:.0f}분 간격: 사이 실행 {int(gap // 30) - 1}번이 빠졌거나 밀린 것으로 보인다"
            t.failure("예약 실행", gap_note)

    t.section("오늘 자료", f"{today} {now.strftime('%H:%M')} 기준" + (" (모의 시각)" if at else ""))
    try:
        five = json.load(open(raw_path, encoding="utf-8"))["fivemin"] if raw_path else fetch_today(stock_id)
    except Exception as e:
        t.failure("yfinance 5분봉 수집", f"{type(e).__name__}: {e}")
        t.save(trace_path)
        append_run({"at": real_now.isoformat(timespec="seconds"), "kind": kind, "subject": stock_id, "fetched": {},
                    "failures": [f"수집 실패: {e}"], "trace": rel(trace_path), "delay_min": delay})
        raise
    five = [r for r in five if r["date"] == today and r["time"] <= (now - dt.timedelta(minutes=5)).strftime("%H:%M")]
    t.fetched("오늘 5분봉", len(five))
    if not five:
        note = "오늘 5분봉이 없다(휴장일이거나 아직 장 전이거나 야후 지연)."
        t.note(note)
        t.save(trace_path)
        append_run({"at": real_now.isoformat(timespec="seconds"), "kind": kind, "subject": stock_id,
                    "fetched": t.data["log"]["fetched"], "failures": [x["what"] + ": " + x["detail"] for x in t.data["log"]["failures"]],
                    "trace": rel(trace_path), "delay_min": delay, "note": note})
        return None

    slotted = {r["slot"]: r for r in data.slots_from_5m(five, SLOT_MIN)}
    wd = data.WEEKDAYS[now.weekday()]
    bars = []
    for s in sessions:
        st = "확정" if closing else slot_status(s, now)
        r = slotted.get(s)
        if st == "확정" and r is None:
            st = "자료 없음"  # 5분봉이 하나도 없다: 거래 없음과 자료 누락을 구분할 수 없어 채점·갱신에서 뺀다
        vol = r["volume"] if r else 0
        med = mj["medians"][s]["median"]
        ratio = vol / med if med else None
        bars.append({"date": today, "weekday": wd, "slot": s, "volume": vol, "n5": r["n5"] if r else 0, "status": st,
                     "open": r["open"] if r else None, "close": r["close"] if r else None,
                     "ratio": ratio, "level": data.level_of(ratio, edges) if ratio is not None and st in ("확정", "잠정") else None})
    t.step("오늘 5분봉을 30분 칸으로", "일반 계산", [
        why("설계 선택", "끝난 30분 칸은 '확정', 지금 진행 중인 칸은 '잠정'(거래량이 아직 다 안 찼다), 시작 전 칸은 '아직'. "
                       "끝났는데 5분봉이 하나도 없는 칸은 '자료 없음'으로 두고 채점과 갱신에서 뺀다."),
        why("설계 선택", "평소 거래량(중앙값)과 단계 경계는 학습 때 값을 그대로 쓴다."),
    ], outputs=[{"type": "table", "name": "오늘 봉", "columns": ["시간", "거래량", "5분봉 수", "상태", "배율", "단계"],
                 "rows": [[b["slot"], b["volume"], b["n5"], b["status"], round(b["ratio"], 2) if b["ratio"] is not None else None,
                           b["level"]] for b in bars]}],
       source=data.slots_from_5m)

    # 채점: 오늘 봉이 이미 프로토타입에 들어 있으면 그 봉을 빼고 잰다(학습 때와 같은 조건)
    cb = model.Codebook(sessions, levels, dim)
    mem = Memory.from_json(mj["memory"])
    included = today in set(mj.get("dates", []))
    th = mj["thresholds"]
    for b in bars:
        if b["level"] is None:
            b["score"] = b["score_bin"] = b["verdict"] = None
            continue
        v = model.bar_vector(cb, b["weekday"], b["slot"], b["level"])
        b["score"] = round(model.counter_score_leave_one_out(mem, v) if included else model.counter_score(mem.counts, v), 4)
        b["score_bin"] = round(model.score_leave_one_out(mem, v) if included else model.score(mem.to_vector(), v), 4)
        b["verdict"] = "이상" if b["score"] <= th["alert"] else "주의" if b["score"] <= th["watch"] else "평소"
    scored = [b for b in bars if b["score"] is not None]
    t.section("채점")
    t.step("봉 벡터를 만들어 프로토타입과 비교", "비교", [
        why("설계 선택", "봉 벡터 = 요일 ⊕ 시간대 ⊕ 단계. 정수 점수(본)와 이진 점수(비교용)를 잰다. 판정 기준은 학습 때 값이다."),
        why("설계 선택", "'잠정' 칸은 거래량이 덜 찬 상태라 단계가 낮게 나오기 쉽다. 판정을 확정으로 읽지 않는다."),
    ] + ([why("설계 선택", "오늘 봉이 이미 프로토타입에 들어 있어(학습이나 앞선 마감 갱신), 그 봉을 빼고 잰다.")] if included else []),
       outputs=[{"type": "table", "name": "오늘 점수", "columns": ["시간", "상태", "단계", "정수 점수", "이진 점수", "판정"],
                 "rows": [[b["slot"], b["status"], b["level"], b["score"], b["score_bin"], b["verdict"]] for b in scored]},
                {"type": "value", "name": "이상 경계", "value": th["alert"]},
                {"type": "value", "name": "주의 경계", "value": th["watch"]}],
       source=model.counter_score)

    # 장 마감 뒤: 프로토타입에 더하기
    added = 0
    if closing and not included:
        before = mem.counts.copy()
        n_before = mem.n
        expect = np.zeros(dim, dtype=np.int64)
        for b in scored:
            v = model.bar_vector(cb, b["weekday"], b["slot"], b["level"])
            mem.add(v)
            expect += 2 * v.astype(np.int64) - 1
            added += 1
        t.section("장 마감 갱신")
        t.step("오늘 봉을 정수 프로토타입에 더하기", "더하기", [
            why("수학적 사실", "정수 프로토타입은 봉 벡터의 +1/-1을 칸마다 더하기만 하면 갱신된다. 다시 학습하지 않는다."),
            why("설계 선택", "평소 거래량(중앙값), 단계 경계, 판정 기준은 바꾸지 않는다. 바꾸려면 학습을 다시 돌린다."),
        ], inputs=[{"type": "value", "name": "더하기 전 봉 수", "value": n_before}],
           outputs=[{"type": "value", "name": "더한 봉", "value": added},
                    {"type": "value", "name": "더한 뒤 봉 수", "value": mem.n},
                    {"type": "table", "name": "앞 4칸의 변화", "columns": ["칸", "전", "더한 값", "후"],
                     "rows": [[i + 1, int(before[i]), int(expect[i]), int(mem.counts[i])] for i in range(4)]},
                    vec_info("갱신된 이진 프로토타입", mem.to_vector(), full=True)],
           check=check("후 - 전 = 더한 봉들의 ±1 합인 칸 수", dim, int(((mem.counts - before) == expect).sum())),
           source=Memory.add)
        mj["memory"] = mem.to_json()
        mj.setdefault("dates", []).append(today)
        mj.setdefault("history", []).append({"at": real_now.isoformat(timespec="seconds"), "kind": "장 마감 갱신",
                                             "date": today, "added": added, "n_after": mem.n})
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(mj, f, ensure_ascii=False, separators=(",", ":"))
    elif closing:
        t.note(f"{today} 봉은 이미 프로토타입에 들어 있어 더하지 않았다.")

    # 저장
    live_dir = os.path.join(out, "live")
    os.makedirs(live_dir, exist_ok=True)
    day = {"stock": stock_id, "date": today, "as_of": now.isoformat(timespec="minutes"), "closing": closing,
           "simulated": at is not None, "bars": bars, "trace": rel(trace_path)}
    with open(os.path.join(live_dir, f"{today}.json"), "w", encoding="utf-8") as f:
        json.dump(day, f, ensure_ascii=False)
    with open(os.path.join(out, "live.json"), "w", encoding="utf-8") as f:
        json.dump(day, f, ensure_ascii=False)
    t.save(trace_path)
    append_run({"at": real_now.isoformat(timespec="seconds"), "kind": kind, "subject": stock_id,
                "fetched": t.data["log"]["fetched"],
                "failures": [f"{x['what']}: {x['detail']}" for x in t.data["log"]["failures"]],
                "trace": rel(trace_path), "delay_min": delay,
                "note": ("모의 시각 " + now.isoformat(timespec="minutes")) if at else (f"프로토타입에 {added}개 더함" if added else None)})
    return day


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock", default=None)
    ap.add_argument("--at", default=None)
    ap.add_argument("--raw", default=None)
    a = ap.parse_args()
    cfg = json.load(open(CONFIG, encoding="utf-8"))
    at = dt.datetime.fromisoformat(a.at) if a.at else None
    for sid in ([a.stock] if a.stock else [s["id"] for s in cfg["stocks"]]):
        try:
            day = run(sid, at, a.raw)
            if day:
                for b in day["bars"]:
                    if b["status"] != "아직":
                        print(f"  {b['slot']} {b['status']} {b['volume']:>8,}주 단계 {b['level']} 점수 {b['score']} {b['verdict']}")
        except Exception:
            traceback.print_exc()


if __name__ == "__main__":
    main()
