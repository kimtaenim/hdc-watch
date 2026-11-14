"""거래량 원자료 받기와 하루 봉 만들기(10-14시대), 단계 나누기.

HDC 연산은 없다. 숫자를 HDC가 다룰 수 있는 이름표(요일, 시간대, 단계)로 바꾸는 앞단이다.
"""

import calendar
import datetime as dt

import numpy as np

KST = dt.timezone(dt.timedelta(hours=9))
WEEKDAYS = "월화수목금토일"


def fetch(ticker: str) -> dict:
    """yfinance로 5분봉(최근 60일, 야후가 주는 최대), 비교용 30분봉, 일봉을 받는다.
    비공식 경로라 실패할 수 있으므로 예외를 그대로 올린다.
    돌려주는 값: {"fivemin": [{"date", "time", "volume", "open", "high", "low", "close"}, ...],
                  "yahoo_30m": [{"date", "slot", "volume"}, ...], "daily": {"날짜": 거래량}}"""
    import yfinance as yf  # 수집할 때만 필요

    t = yf.Ticker(ticker)
    m = t.history(period="60d", interval="5m", auto_adjust=False)
    h = t.history(period="60d", interval="30m", auto_adjust=False)
    d = t.history(period="6mo", interval="1d", auto_adjust=False)
    if m.empty or d.empty:
        raise RuntimeError(f"빈 응답: 5분봉 {len(m)}행, 일봉 {len(d)}행")
    fivemin = [{"date": ts.strftime("%Y-%m-%d"), "time": ts.strftime("%H:%M"), "volume": int(r.Volume),
                "open": float(r.Open), "high": float(r.High), "low": float(r.Low), "close": float(r.Close)}
               for ts, r in zip(m.index.tz_convert("Asia/Seoul"), m.itertuples())]
    yahoo_30m = [{"date": ts.strftime("%Y-%m-%d"), "slot": ts.strftime("%H:%M"), "volume": int(v)}
                 for ts, v in zip(h.index.tz_convert("Asia/Seoul"), h["Volume"])]
    daily = {ts.strftime("%Y-%m-%d"): int(v) for ts, v in zip(d.index, d["Volume"])}
    return {"fivemin": fivemin, "yahoo_30m": yahoo_30m, "daily": daily}


def slot_of(time: str, minutes: int = 30) -> str:
    """"09:45" → "09:30". 시각을 minutes분 칸의 시작 시각으로."""
    h, m = map(int, time.split(":"))
    return f"{h:02d}:{(m // minutes) * minutes:02d}"


def slots_from_5m(fivemin, minutes: int = 30):
    """5분봉을 minutes분 칸으로 합친다. 거래량 = 합, 시가 = 첫 봉 시가, 고가·저가 = 최대·최소, 종가 = 마지막 봉 종가.
    야후는 거래가 없던 5분 칸은 보내지 않으므로, 한 칸의 5분봉 수가 minutes/5개보다 적을 수 있다."""
    groups = {}
    for r in sorted(fivemin, key=lambda r: (r["date"], r["time"])):
        groups.setdefault((r["date"], slot_of(r["time"], minutes)), []).append(r)
    return [{"date": d, "slot": sl, "volume": sum(x["volume"] for x in g), "n5": len(g),
             "open": g[0]["open"], "high": max(x["high"] for x in g), "low": min(x["low"] for x in g), "close": g[-1]["close"]}
            for (d, sl), g in sorted(groups.items())]


def complete_days(rows, daily, now: dt.datetime, close_hour: int = 16):
    """일별 거래량이 확정된 날짜 목록. 오늘은 close_hour시 이후에만 확정으로 본다."""
    today = now.astimezone(KST).strftime("%Y-%m-%d")
    days = sorted({r["date"] for r in rows} & set(daily))
    return [d for d in days if d < today or (d == today and now.astimezone(KST).hour >= close_hour)]


def window(days, end: str, months: int):
    """end를 포함해 거슬러 months개월 사이의 날짜. (end - months개월) 당일은 빼고 그 다음 날부터."""
    e = dt.date.fromisoformat(end)
    y, m = e.year, e.month - months
    while m <= 0:
        y, m = y - 1, m + 12
    start = dt.date(y, m, min(e.day, calendar.monthrange(y, m)[1]))
    return [d for d in days if start.isoformat() < d <= end], start.isoformat()


def build_bars(slotted, dates, sessions):
    """날짜마다 정해진 시간대(30분 칸)의 봉을 만든다.
    봉: {"date", "weekday", "slot", "volume"(없으면 None), "missing": 이유 또는 None}
    (15시대는 야후 5분봉에 없고 일별 거래량에서 역산하면 부정확해 쓰지 않는다. 2026-11-11 의장님 결정.)"""
    by_key = {(r["date"], r["slot"]): r for r in slotted}
    bars = []
    for d in dates:
        wd = WEEKDAYS[dt.date.fromisoformat(d).weekday()]
        for s in sessions:
            r = by_key.get((d, s))
            # 5분봉이 하나도 없는 칸은 '거래 없음'과 '야후 자료 누락'을 구분할 수 없다(2026-11-13 14:30처럼
            # 평소 수천 주인 칸이 통째로 비는 일이 있었다). 0주로 두지 않고 결측으로 둔다.
            bar = {"date": d, "weekday": wd, "slot": s, "volume": r["volume"] if r else None, "n5": 0,
                   "missing": None if r else "5분봉 없음(거래 없음 또는 자료 누락)"}
            if r:
                bar["ohlc"] = [r["open"], r["high"], r["low"], r["close"]]
                bar["n5"] = r["n5"]
            bars.append(bar)
    return bars


def session_medians(bars, sessions):
    """시간대마다 결측이 아닌 봉의 거래량 중앙값."""
    out = {}
    for s in sessions:
        vals = [b["volume"] for b in bars if b["slot"] == s and b["missing"] is None]
        out[s] = {"median": float(np.median(vals)) if vals else None, "n": len(vals)}
    return out


def level_of(ratio: float, edges) -> int:
    """배율 → 단계(1부터). 경계값은 위 단계에 넣는다: 정확히 1배는 6단계(1배-2배)."""
    return int(np.searchsorted(edges, ratio, side="right")) + 1


def assign_levels(bars, medians, edges):
    for b in bars:
        med = medians[b["slot"]]["median"]
        if b["missing"] is None and med:
            b["ratio"] = b["volume"] / med
            b["level"] = level_of(b["ratio"], edges)
        else:
            b["ratio"] = None
            b["level"] = None
    return bars
