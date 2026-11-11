"""RSS 제목 수집. 제목, 링크, 날짜, 매체명만 저장한다(본문 저장 안 함). 링크로 중복을 거른다.

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


def parse_date(text: str | None, fallback: dt.datetime) -> dt.datetime:
    if text:
        try:
            return email.utils.parsedate_to_datetime(text).astimezone(KST)
        except Exception:
            try:
                return dt.datetime.fromisoformat(text.strip().replace("Z", "+00:00")).astimezone(KST)
            except Exception:
                pass
    return fallback


def fetch_feed(url: str, source: str, now: dt.datetime):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=20) as r:
        root = ET.fromstring(r.read())
    out = []
    for it in root.findall(".//item"):
        title = (it.findtext("title") or "").strip()
        link = (it.findtext("link") or "").strip()
        if not title or not link:
            continue
        when = parse_date(it.findtext("pubDate") or it.findtext("{http://purl.org/dc/elements/1.1/}date"), now)
        out.append({"title": title, "link": link, "date": when.strftime("%Y-%m-%d"),
                    "time": when.strftime("%H:%M"), "source": source})
    return out


def known_links() -> set:
    links = set()
    for f in glob.glob(os.path.join(TITLES, "*.json")):
        links |= {x["link"] for x in json.load(open(f, encoding="utf-8"))}
    return links


def collect(feeds, now: dt.datetime | None = None):
    """모든 피드를 받아 새 링크만 날짜별 파일에 더한다. (매체별 받은 수, 새 제목 수, 실패 목록)을 돌려준다."""
    now = now or dt.datetime.now(KST)
    seen = known_links()
    got, new, fails = {}, {}, []
    for f in feeds:
        try:
            items = fetch_feed(f["url"], f["source"], now)
        except Exception as e:
            fails.append(f"{f['source']} {f['url']}: {type(e).__name__}: {e}"[:200])
            continue
        got[f["source"]] = got.get(f["source"], 0) + len(items)
        for x in items:
            if x["link"] in seen:
                continue
            seen.add(x["link"])
            x["seen_at"] = now.isoformat(timespec="minutes")
            new.setdefault(x["date"], []).append(x)
    os.makedirs(TITLES, exist_ok=True)
    for day, rows in new.items():
        path = os.path.join(TITLES, f"{day}.json")
        old = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else []
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(old + rows, fh, ensure_ascii=False, indent=0)
    return got, sum(len(v) for v in new.values()), fails


def load_titles(days):
    out = []
    for d in days:
        path = os.path.join(TITLES, f"{d}.json")
        if os.path.exists(path):
            out += json.load(open(path, encoding="utf-8"))
    return out
