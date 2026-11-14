"""RSS 제목 수집. 제목, 링크, 날짜, 매체명만 저장한다(본문 저장 안 함). 링크와 (매체, 제목)으로 중복을 거른다.

저장: docs/data/trend/titles/YYYY-MM-DD.json (발행일 기준, 한국 시각)
"""

import datetime as dt
import email.utils
import glob
import json
import os
import urllib.request
import xml.etree.ElementTree as ET

KST = dt.timezone(dt.timedelta(hours=9))
UA = "hdc-watch-study/1.0 (personal HDC learning project)"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TITLES = os.path.join(ROOT, "docs", "data", "trend", "titles")


def parse_date(text: str | None, fallback: dt.datetime, as_kst: bool = False) -> dt.datetime:
    """as_kst: 시간대 표시를 무시하고 적힌 시각을 한국 시각으로 읽는다(시간대를 잘못 적는 피드용)."""
    if text:
        try:
            d = email.utils.parsedate_to_datetime(text)
            return d.replace(tzinfo=KST) if as_kst else d.astimezone(KST)
        except Exception:
            try:
                d = dt.datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
                # 시간대가 없는 날짜("2026-11-13 23:48:00")는 한국 시각으로 본다(국내 매체, 실행 서버는 UTC)
                return d.replace(tzinfo=KST) if d.tzinfo is None else d.astimezone(KST)
            except Exception:
                pass
    return fallback


def fix_future(when: dt.datetime, now: dt.datetime):
    """발행 시각이 수집 시각보다 10분 넘게 뒤면, 한국 시각을 UTC로 잘못 적은 것으로 보고 9시간을 뺀다.
    (2026-11-11 확인: 매일경제와 블로터 RSS가 한국 시각에 +0000을 붙인다.)"""
    if when > now + dt.timedelta(minutes=10):
        return when - dt.timedelta(hours=9), True
    return when, False


def fetch_feed(url: str, source: str, now: dt.datetime, as_kst: bool = False):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=20) as r:
        root = ET.fromstring(r.read())
    out = []
    for it in root.findall(".//item"):
        title = (it.findtext("title") or "").strip()
        link = (it.findtext("link") or "").strip()
        if not title or not link:
            continue
        when = parse_date(it.findtext("pubDate") or it.findtext("{http://purl.org/dc/elements/1.1/}date"), now, as_kst)
        when, fixed = fix_future(when, now)
        out.append({"title": title, "link": link, "date": when.strftime("%Y-%m-%d"),
                    "time": when.strftime("%H:%M"), "source": source, **({"time_fixed": True} if fixed else {})})
    return out


def known_keys() -> set:
    """이미 쌓은 링크와 (매체, 제목). 같은 기사가 링크만 달리해 다시 나오는 경우가 있어 둘 다로 거른다."""
    keys = set()
    for f in glob.glob(os.path.join(TITLES, "*.json")):
        for x in json.load(open(f, encoding="utf-8")):
            keys.add(x["link"])
            keys.add((x["source"], x["title"]))
    return keys


def collect(feeds, now: dt.datetime | None = None, max_age_days: int = 3):
    """모든 피드를 받아 새 링크만 날짜별 파일에 더한다. 발행일이 수집 시각보다 max_age_days일 넘게 지난 글은 뺀다.
    (매체별 받은 수, 새 제목 수, 실패 목록, 오래돼서 뺀 수)를 돌려준다."""
    now = now or dt.datetime.now(KST)
    seen = known_keys()
    got, new, fails = {}, {}, []
    skipped_old = 0
    for f in feeds:
        try:
            items = fetch_feed(f["url"], f["source"], now, f.get("pubdate_kst", False))
        except Exception as e:
            fails.append(f"{f['source']} {f['url']}: {type(e).__name__}: {e}"[:200])
            continue
        got[f["source"]] = got.get(f["source"], 0) + len(items)
        oldest = (now - dt.timedelta(days=max_age_days)).strftime("%Y-%m-%d")
        for x in items:
            if x["date"] < oldest:
                skipped_old += 1  # 몇 달 치를 한꺼번에 주는 피드가 있다. 지난 것을 채우지 않고 쌓아 가기로 했으므로 뺀다
                continue
            if x["link"] in seen or (x["source"], x["title"]) in seen:
                continue
            seen.add(x["link"])
            seen.add((x["source"], x["title"]))
            x["seen_at"] = now.isoformat(timespec="minutes")
            new.setdefault(x["date"], []).append(x)
    os.makedirs(TITLES, exist_ok=True)
    for day, rows in new.items():
        path = os.path.join(TITLES, f"{day}.json")
        old = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else []
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(old + rows, fh, ensure_ascii=False, indent=0)
    return got, sum(len(v) for v in new.values()), fails, skipped_old


def load_titles(days):
    out = []
    for d in days:
        path = os.path.join(TITLES, f"{d}.json")
        if os.path.exists(path):
            out += json.load(open(path, encoding="utf-8"))
    return out
