"""거래량 원자료 받기와 하루 봉 만들기, 단계 나누기.

HDC 연산은 없다. 숫자를 HDC가 다룰 수 있는 이름표(요일, 시간대, 단계)로 바꾸는 앞단이다.
"""

import calendar
import datetime as dt

import numpy as np

KST = dt.timezone(dt.timedelta(hours=9))
WEEKDAYS = "월화수목금토일"


def fetch(ticker: str):
    """yfinance로 1시간봉과 일봉을 받는다. 비공식 경로라 실패할 수 있으므로 예외를 그대로 올린다.

    돌려주는 값
      hourly: [{"date": "2026-11-10", "hour": 9, "volume": 1234, "open", "high", "low", "close"}, ...]
      daily:  {"2026-11-10": 32059, ...}            일별 거래량
      closes: {"2026-11-10": 1057.0, ...}           일별 종가
    """
    import yfinance as yf  # 수집할 때만 필요

    t = yf.Ticker(ticker)
    h = t.history(period="730d", interval="1h", auto_adjust=False)
    d = t.history(period="1y", interval="1d", auto_adjust=False)
    if h.empty or d.empty:
        raise RuntimeError(f"빈 응답: 시간봉 {len(h)}행, 일봉 {len(d)}행")
    hourly = [{"date": ts.strftime("%Y-%m-%d"), "hour": ts.hour, "volume": int(r.Volume),
               "open": float(r.Open), "high": float(r.High), "low": float(r.Low), "close": float(r.Close)}
              for ts, r in zip(h.index.tz_convert("Asia/Seoul"), h.itertuples())]
    daily = {ts.strftime("%Y-%m-%d"): int(v) for ts, v in zip(d.index, d["Volume"])}
    closes = {ts.strftime("%Y-%m-%d"): float(v) for ts, v in zip(d.index, d["Close"])}
    return hourly, daily, closes


def complete_days(hourly, daily, now: dt.datetime, close_hour: int = 16):
    """일별 거래량이 확정된 날짜 목록. 오늘은 close_hour시 이후에만 확정으로 본다."""
    today = now.astimezone(KST).strftime("%Y-%m-%d")
    days = sorted({r["date"] for r in hourly} & set(daily))
    return [d for d in days if d < today or (d == today and now.astimezone(KST).hour >= close_hour)]


def after_break(day: str, days) -> bool:
    """day가 쉬는 날(주말, 휴일) 바로 다음 거래일인가: 전날이 거래일 목록에 없으면 참."""
    prev = (dt.date.fromisoformat(day) - dt.timedelta(days=1)).isoformat()
    return prev not in days


def window(days, end: str, months: int):
    """end를 포함해 거슬러 months개월 사이의 날짜. (end - months개월) 당일은 빼고 그 다음 날부터."""
    e = dt.date.fromisoformat(end)
    y, m = e.year, e.month - months
    while m <= 0:
        y, m = y - 1, m + 12
    start = dt.date(y, m, min(e.day, calendar.monthrange(y, m)[1]))
    return [d for d in days if start.isoformat() < d <= end], start.isoformat()


def build_bars(hourly, daily, dates, sessions, derived_session, zero_is_missing=(), hourly_sessions=None):
    """날짜마다 봉을 만든다. 시간봉 시간대는 그대로, 15시대는 '일별 - 시간봉 합'으로 역산.

    sessions: 봉으로 만들 시간대들. hourly_sessions: 역산에서 빼는 시간봉 시간대들(없으면 sessions에서 역산 시간대를 뺀 것).
      봉으로 만들지 않는 시간대(예: 9시대)도 역산할 때는 빼야, 그 거래량이 역산 봉에 섞이지 않는다.
    zero_is_missing: 거래량 0을 자료 누락으로 보는 시간대들. 그 봉과 그날 역산 봉을 결측으로 둔다.
      (야후 시간봉은 쉬는 날 다음 첫 거래일의 9시대를 0으로 내는데, 같은 날 30분봉·5분봉에는 거래가 있다.
       빠진 거래량은 일별에는 들어 있어 역산 봉을 부풀린다.)

    봉: {"date", "weekday", "hour", "volume"(없으면 None), "source": "시간봉"|"역산", "missing": 이유 또는 None}
    """
    by_key = {(r["date"], r["hour"]): r["volume"] for r in hourly}
    regular = hourly_sessions or [s for s in sessions if s != derived_session]
    bars, problems = [], []
    for d in dates:
        wd = WEEKDAYS[dt.date.fromisoformat(d).weekday()]
        vols = [by_key.get((d, s)) for s in regular]
        zeroed = [s for s, v in zip(regular, vols) if v == 0 and s in zero_is_missing]
        for s, v in zip(regular, vols):
            if s not in sessions:
                continue  # 역산에만 쓰고 봉으로는 만들지 않는 시간대
            miss = "시간봉 없음" if v is None else (f"{s}시대 0주: 시간봉 누락으로 봄" if s in zeroed else None)
            bars.append({"date": d, "weekday": wd, "hour": s, "volume": v, "source": "시간봉", "missing": miss})
        for s in zeroed:
            problems.append({"date": d, "reason": f"{s}시대 시간봉 0주(누락으로 봄)"})
        reason, derived = None, None
        if any(v is None for v in vols):
            reason = f"시간봉 {len(regular)}개가 다 있지 않아 역산 불가"
        elif zeroed:
            reason = f"{zeroed[0]}시대 시간봉 누락: 빠진 거래량이 역산값에 섞임"
        elif d not in daily:
            reason = "일별 거래량 없음"
        else:
            derived = daily[d] - sum(vols)
            if derived < 0:
                reason = f"역산값이 음수({derived:,}): 시간봉 합 {sum(vols):,} > 일별 {daily[d]:,}"
                derived = None
        if reason:
            problems.append({"date": d, "reason": reason})
        bars.append({"date": d, "weekday": wd, "hour": derived_session, "volume": derived, "source": "역산",
                     "daily": daily.get(d), "hourly_sum": sum(v for v in vols if v is not None),
                     "missing": reason})
    return bars, problems


def session_medians(bars, sessions):
    """시간대마다 결측이 아닌 봉의 거래량 중앙값."""
    out = {}
    for s in sessions:
        vals = [b["volume"] for b in bars if b["hour"] == s and b["missing"] is None]
        out[s] = {"median": float(np.median(vals)) if vals else None, "n": len(vals)}
    return out


def level_of(ratio: float, edges) -> int:
    """배율 → 단계(1부터). 경계값은 위 단계에 넣는다: 정확히 1배는 6단계(1배-2배)."""
    return int(np.searchsorted(edges, ratio, side="right")) + 1


def assign_levels(bars, medians, edges):
    for b in bars:
        med = medians[b["hour"]]["median"]
        if b["missing"] is None and med:
            b["ratio"] = b["volume"] / med
            b["level"] = level_of(b["ratio"], edges)
        else:
            b["ratio"] = None
            b["level"] = None
    return bars
