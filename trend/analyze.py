"""트렌드 문맥 탐지: 키워드 옆에 새로 붙기 시작한 단어 찾기.

제목에 키워드 K가 있으면, 같이 나온 명사 w마다 'K ⊕ w'를 정수 프로토타입에 더한다.
키워드마다 '최근 7일' 프로토타입과 '과거(그 전 28일)' 프로토타입을 따로 둔다.
새 조합 = 최근 7일에 여러 번 나왔는데, 과거 프로토타입에 물어보면 점수가 잡음 수준인 단어.
점수는 정수 프로토타입을 K로 풀고 단어 벡터와 코사인을 잰 값이다. 트렌드는 '얼마나 자주'가 중요해
다수결로 0/1을 만들지 않는다(2026-11-11 의장님 의견).

실행: python -m trend.analyze [--collect] [--date YYYY-MM-DD]
결과: docs/data/trend/<키워드>/day.json, proto.json, docs/data/trend/traces/trend-YYYYMMDD.json, docs/data/runs.json
"""

import argparse
import datetime as dt
import json
import math
import os
import traceback

import numpy as np

from hdc import D, Memory, bind, make_vector, noise_level, similarity
from hdc.trace import REPO_ROOT, Trace, check, vec_info, why
from trend import collect, text
from volume.train import CONFIG, append_run, rel

OUT = os.path.join(REPO_ROOT, "docs", "data", "trend")


def kw_name(kid: str) -> str:
    return f"키워드:{kid}"


def word_name(w: str) -> str:
    return f"단어:{w}"


def bipolar(v):
    return 2 * v.astype(np.int64) - 1


def ask(counts: np.ndarray, kv: np.ndarray, wv: np.ndarray) -> float:
    """정수 프로토타입을 키워드로 풀고, 단어 벡터(±1)와 코사인.
    이 저장소는 0을 -1, 1을 +1로 바꾼다. 이때 두 벡터의 XOR은 칸끼리의 곱에 -1을 곱한 것과 같다:
    ±1(a ⊕ b) = -(±1(a) × ±1(b)). 그래서 풀 때도 -1을 곱한다."""
    norm = float(np.sqrt((counts.astype(np.float64) ** 2).sum()))
    if norm == 0:
        return 0.0
    q = -counts * bipolar(kv)
    return float((q * bipolar(wv)).sum()) / (math.sqrt(len(kv)) * norm)


def cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    na, nb = float(np.sqrt((a.astype(np.float64) ** 2).sum())), float(np.sqrt((b.astype(np.float64) ** 2).sum()))
    return float((a.astype(np.float64) * b).sum()) / (na * nb) if na and nb else None


def pairs_of(titles, kw):
    """제목마다 (제목, 같이 나온 명사들). 그 키워드의 표기 구간만 명사에서 뺀다."""
    out = []
    for t in titles:
        spans = text.find_keywords(t["title"], [kw]).get(kw["id"])
        if spans:
            out.append((t, text.nouns(t["title"], spans)))
    return out


def word_counts(pairs):
    c = {}
    for _, ws in pairs:
        for w in ws:
            c[w] = c.get(w, 0) + 1
    return c


def word_titles(pt, pp, words, n_past: int = 10):
    """낱말마다 그 낱말이 키워드와 함께 나온 제목: 최근은 day.json 제목 목록의 번호, 과거는 최근 것부터 n_past개."""
    out = {}
    for w in words:
        out[w] = {"recent": [i for i, (_, ws) in enumerate(pt) if w in ws],
                  "past": [[x["date"], x["source"], x["title"], x["link"]] for x, ws in sorted(pp, key=lambda p: p[0]["date"], reverse=True)
                           if w in ws][:n_past]}
    return out


def run(date: str | None = None, do_collect: bool = False):
    cfg = json.load(open(CONFIG, encoding="utf-8"))
    tc, keywords = cfg["trend"], cfg["keywords"]
    now = dt.datetime.now(collect.KST)
    today = date or now.strftime("%Y-%m-%d")
    t = Trace("trend.analyze", subject={"type": "keywords", "id": [k["id"] for k in keywords]},
              params={"D": D, "date": today, "recent_days": tc["recent_days"], "past_days": tc["past_days"], "min_recent": tc["min_recent"], "noise_k": tc["noise_k"]})
    trace_path = os.path.join(OUT, "traces", f"trend-{today.replace('-', '')}.json")

    t.section("수집", "RSS 제목을 받아 링크로 중복을 거르고 날짜별로 쌓는다.")
    if do_collect:
        got, n_new, fails, n_old = collect.collect(tc["feeds"], now, tc.get("max_age_days"))
        if n_old:
            t.note(f"발행일이 {tc.get('max_age_days')}일 넘게 지난 글 {n_old}개는 뺐다")
        for f in fails:
            t.failure("RSS 수집", f)
        for k, v in got.items():
            t.fetched(k, v)
    else:
        got, n_new = {}, 0
    rd, pd_ = tc["recent_days"], tc["past_days"]
    RL = f"최근 {rd}일"
    base = dt.date.fromisoformat(today)
    recent_dates = [(base - dt.timedelta(days=i)).isoformat() for i in range(rd - 1, -1, -1)]
    past_dates = [(base - dt.timedelta(days=i)).isoformat() for i in range(rd + pd_ - 1, rd - 1, -1)]
    titles_recent = collect.load_titles(recent_dates)
    titles_past = collect.load_titles(past_dates)
    past_days_with = sum(1 for d in past_dates if os.path.exists(os.path.join(collect.TITLES, f"{d}.json")))
    t.step("RSS 제목 받기", "일반 계산", [
        why("설계 선택", "매체가 공식으로 내놓는 RSS의 제목, 링크, 날짜, 매체명만 저장한다. 본문은 저장하지 않는다. "
                       "RSS는 최근 글만 주므로 하루 여러 번 받아 링크로 중복을 거른다."),
    ], outputs=[{"type": "table", "name": "이번 실행에서 받은 수", "columns": ["매체", "받은 글"], "rows": [[k, v] for k, v in got.items()]},
                {"type": "value", "name": "새로 쌓은 제목", "value": n_new},
                {"type": "value", "name": f"{RL} 제목", "value": len(titles_recent)},
                {"type": "value", "name": f"과거(그 전 {pd_}일) 제목", "value": len(titles_past)},
                {"type": "value", "name": "과거 중 자료가 있는 날", "value": past_days_with},
                {"type": "table", "name": "최근 제목 예시", "columns": ["날짜", "시각", "매체", "제목"],
                 "rows": [[x["date"], x["time"], x["source"], x["title"]] for x in titles_recent[-5:]]}],
       source=collect.collect)

    noise = noise_level(D)
    thr = tc["noise_k"] * noise
    any_warm = False
    summary = []
    os.makedirs(OUT, exist_ok=True)
    for kw in keywords:
        kid = kw["id"]
        t.section(f"키워드: {kw['name']}", f"표기: {', '.join(kw['variants'])}")
        pt, pp = pairs_of(titles_recent, kw), pairs_of(titles_past, kw)
        kw_days = len({x["date"] for x, _ in pp})  # 이 키워드가 든 제목이 있었던 과거 날 수
        warm = kw_days < tc["warmup_days"]
        any_warm = any_warm or warm
        ct, cp = word_counts(pt), word_counts(pp)
        t.step("키워드가 든 제목과 같이 나온 명사", "일반 계산", [
            why("설계 선택", "키워드는 제목 원문에서 표기 사전으로 찾고, 명사는 kiwipiepy로 뽑는다. 띄어쓰기 없이 붙은 조각은 합치고, "
                           "그 키워드의 표기 부분만 뺀다. 같은 제목 안에서 같은 명사는 한 번만 센다."),
        ], outputs=[{"type": "value", "name": f"{RL} 제목", "value": len(pt)},
                    {"type": "value", "name": "과거 제목", "value": len(pp)},
                    {"type": "table", "name": f"{RL} 제목과 명사", "columns": ["매체", "제목", "명사"],
                     "rows": [[x["source"], x["title"], ws] for x, ws in pt[:20]]}],
           source=text.nouns)

        kv = make_vector(kw_name(kid))
        mt, mp = Memory(D), Memory(D)
        vecs = {}
        for pairs, mem in ((pt, mt), (pp, mp)):
            for _, ws in pairs:
                for w in ws:
                    if w not in vecs:
                        vecs[w] = make_vector(word_name(w))
                    mem.add(bind(kv, vecs[w]))
        ex_w = max(ct, key=ct.get) if ct else None
        if ex_w:
            t.step("키워드 ⊕ 단어로 묶어 더하기", "묶기", [
                why("설계 선택", f"같이 나온 명사마다 '키워드 ⊕ 단어'를 {RL} 또는 과거 정수 프로토타입에 더한다. 둘은 따로 둔다."),
                why("수학적 사실", "같은 키워드로 묶었으므로, 정수 프로토타입을 키워드로 다시 풀면 단어 벡터들의 합이 남는다. "
                               "0을 -1, 1을 +1로 볼 때 두 벡터의 XOR은 칸끼리 곱에 -1을 곱한 것이라, 풀 때 -1을 곱해 맞춘다."),
            ], inputs=[vec_info(kw_name(kid), kv, full=True), vec_info(word_name(ex_w), vecs[ex_w], full=True)],
               outputs=[vec_info(f"{kw_name(kid)} ⊕ {word_name(ex_w)}", bind(kv, vecs[ex_w]), full=True),
                        {"type": "value", "name": f"{RL} 더한 조합 수", "value": mt.n},
                        {"type": "value", "name": "과거 더한 조합 수", "value": mp.n}],
               check=check("묶은 벡터 ⊕ 키워드 = 단어 벡터인가", True, bool(np.array_equal(bind(bind(kv, vecs[ex_w]), kv), vecs[ex_w]))),
               source=bind)

        # 묻기
        rows, new = [], []
        one = 1 / math.sqrt(sum(c * c for c in cp.values())) if cp else None
        for w, c in sorted(ct.items(), key=lambda x: -x[1]):
            s_past = ask(mp.counts, kv, vecs[w]) if mp.n else 0.0
            s_today = ask(mt.counts, kv, vecs[w])
            hdc_new = c >= tc["min_recent"] and s_past < thr
            cnt_new = c >= tc["min_recent"] and cp.get(w, 0) == 0
            rows.append([w, c, round(s_today, 4), cp.get(w, 0), round(s_past, 4),
                         "새 조합" if hdc_new else "", "새 조합" if cnt_new else ""])
            if hdc_new:
                new.append({"word": w, "recent": c, "past_score": round(s_past, 4), "past_count": cp.get(w, 0)})
        agree = sum(1 for r in rows if (r[5] == "새 조합") == (r[6] == "새 조합"))
        t.step(f"과거 프로토타입에 물어보기: {RL} 단어가 과거에도 있었나", "비교", [
            why("설계 선택", f"과거 점수 = 과거 정수 프로토타입을 키워드로 풀어 단어 벡터와 잰 코사인. "
                           f"{RL}에 {tc['min_recent']}번 이상 나왔고 과거 점수가 잡음 수준({tc['noise_k']} × 1/√D = {thr:.3f}) 아래면 새 조합이다."),
            why("수학적 사실", "과거에 없던 단어의 점수는 0 근처에서 약 1/√D 흔들린다(근사). 과거에 c번 나온 단어의 점수는 대략 "
                           "c / √(과거 단어별 횟수의 제곱합)이다."),
            why("이 데이터에서 관찰", (f"과거에 한 번 나온 단어의 점수는 약 {one:.3f}로, 잡음 수준 {thr:.3f}"
                                    + ("보다 크다. 한 번 나온 것도 구별된다." if one > thr else "보다 작다. 과거가 쌓여 한 번 나온 단어는 잡음에 묻힌다(용량의 한계). 그래서 과거를 최근 며칠로 나눠 둔다."))
                                   if one else "과거 자료가 없다."),
        ] + ([why("이 데이터에서 관찰", f"과거 4주 중 이 키워드 제목이 있었던 날이 {kw_days}일로 {tc['warmup_days']}일보다 적다. 아직 과거가 얕아(예열 중) 새 조합이 많이 나오기 쉽다.")] if warm else []),
           outputs=[{"type": "table", "name": f"{RL} 단어",
                     "columns": ["단어", "최근 횟수", "최근 점수", "과거 횟수", "과거 점수", "HDC 판정", "카운트 판정"], "rows": rows},
                    {"type": "value", "name": "잡음 수준(판정선)", "value": round(thr, 4)},
                    {"type": "value", "name": "과거에 한 번 나온 단어의 점수(약)", "value": round(one, 4) if one else None}],
           check=check("HDC 판정과 카운트 판정(과거 횟수 0)이 같은 단어 수", len(rows), agree),
           source=ask)

        cos_h = cosine(mt.counts, mp.counts)
        words = sorted(set(ct) | set(cp))
        cos_c = cosine(np.array([ct.get(w, 0) for w in words]), np.array([cp.get(w, 0) for w in words])) if words else None
        t.step(f"{RL} 문맥과 과거 문맥 전체 비교", "비교", [
            why("설계 선택", f"{RL} 정수 프로토타입과 과거 정수 프로토타입의 코사인. 낮으면 키워드 옆 단어들이 통째로 바뀌었다는 뜻이다."),
            why("수학적 사실", "둘 다 같은 키워드로 묶였으므로, 이 코사인은 최근과 과거의 단어 묶음(BoW) 코사인을 근사한다."),
        ], outputs=[{"type": "value", "name": "HDC 코사인", "value": round(cos_h, 4) if cos_h is not None else None},
                    {"type": "value", "name": "단어 횟수(BoW) 코사인", "value": round(cos_c, 4) if cos_c is not None else None}],
           source=cosine)

        kdir = os.path.join(OUT, kid)
        os.makedirs(kdir, exist_ok=True)
        day = {"keyword": kid, "name": kw["name"], "date": today, "warmup": warm, "past_days_with": kw_days,
               "recent_days": rd, "recent_from": recent_dates[0], "past_from": past_dates[0], "past_to": past_dates[-1],
               "titles_recent": len(pt), "titles_past": len(pp), "threshold": round(thr, 4), "one_past": one,
               "new": new, "rows": rows, "cos_hdc": cos_h, "cos_bow": cos_c, "trace": rel(trace_path),
               "titles": [[f"{x['date'][5:]} {x['time']}", x["source"], x["title"], x["link"], ws] for x, ws in pt],
               "word_titles": word_titles(pt, pp, [r[0] for r in rows])}
        with open(os.path.join(kdir, "day.json"), "w", encoding="utf-8") as f:
            json.dump(day, f, ensure_ascii=False)
        with open(os.path.join(kdir, f"{today}.json"), "w", encoding="utf-8") as f:
            json.dump(day, f, ensure_ascii=False)
        with open(os.path.join(kdir, "proto.json"), "w", encoding="utf-8") as f:
            json.dump({"date": today, "recent": mt.to_json(), "past": mp.to_json()}, f, separators=(",", ":"))
        summary.append([kw["name"], len(pt), len(pp), len(new), round(cos_h, 3) if cos_h is not None else None, "예열 중" if warm else ""])

    t.section("요약")
    t.step(f"키워드별 {RL} 새 조합", "일반 계산", [why("설계 선택", "키워드마다 새 조합 수와 문맥 코사인을 모은다.")],
           outputs=[{"type": "table", "name": "요약", "columns": ["키워드", "최근 제목", "과거 제목", "새 조합", "문맥 코사인", "상태"], "rows": summary}],
           source=run)
    dpath = os.path.join(OUT, "daily.json")
    daily = json.load(open(dpath, encoding="utf-8")) if os.path.exists(dpath) else {}
    daily[today] = {"new_total": sum(r[3] for r in summary), "warmup": any_warm}
    with open(dpath, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(daily.items())), f, ensure_ascii=False)
    with open(os.path.join(OUT, "keywords.json"), "w", encoding="utf-8") as f:
        json.dump([{"id": k["id"], "name": k["name"]} for k in keywords], f, ensure_ascii=False)
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"date": today, "at": now.isoformat(timespec="minutes"), "warmup": any_warm,
                   "new_total": sum(r[3] for r in summary), "rows": summary, "trace": rel(trace_path)}, f, ensure_ascii=False)
    t.save(trace_path)
    append_run({"at": now.isoformat(timespec="seconds"), "kind": "trend", "subject": "키워드",
                "fetched": t.data["log"]["fetched"],
                "failures": [f"{x['what']}: {x['detail']}" for x in t.data["log"]["failures"]],
                "trace": rel(trace_path), "note": f"새 제목 {n_new}개" if do_collect else None})
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--date", default=None)
    a = ap.parse_args()
    try:
        for r in run(a.date, a.collect):
            print(r)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
