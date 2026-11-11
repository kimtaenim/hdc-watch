"""RSS 후보 주소가 실제로 열리는지, robots.txt가 허용하는지 확인한다. GitHub Actions에서 돌린다.

실행: python -m trend.probe  → docs/data/trend/feed_check.json
"""

import datetime as dt
import json
import os
import urllib.request
import urllib.robotparser
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

UA = "hdc-watch-study/1.0 (personal HDC learning project)"
CANDIDATES = {
    "한국경제": ["https://www.hankyung.com/feed/all-news", "https://www.hankyung.com/feed/economy", "https://www.hankyung.com/feed/it"],
    "매일경제": ["https://www.mk.co.kr/rss/30000001/", "https://www.mk.co.kr/rss/50300009/", "https://www.mk.co.kr/rss/30100041/"],
    "이데일리": ["http://rss.edaily.co.kr/edaily_news.xml", "https://rss.edaily.co.kr/edaily_news.xml", "http://rss.edaily.co.kr/economy_news.xml"],
    "지디넷코리아": ["https://feeds.feedburner.com/zdkorea", "https://zdnet.co.kr/rss/newsall.xml"],
    "블로터": ["https://www.bloter.net/rss/allArticle.xml", "https://www.bloter.net/feed"],
    "전자신문": ["http://rss.etnews.com/Section901.xml", "https://rss.etnews.com/Section901.xml", "http://rss.etnews.com/Section902.xml"],
    "뉴시스": ["https://newsis.com/RSS/sokbo.xml", "https://www.newsis.com/RSS/economy.xml", "https://newsis.com/RSS/economy.xml"],
}
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "data", "trend", "feed_check.json")


def robots_ok(url: str):
    p = urlparse(url)
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(f"{p.scheme}://{p.netloc}/robots.txt")
    try:
        req = urllib.request.Request(rp.url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=15) as r:
            rp.parse(r.read().decode("utf-8", "replace").splitlines())
        return rp.can_fetch(UA, url), "robots.txt 읽음"
    except Exception as e:
        return None, f"robots.txt 못 읽음: {type(e).__name__}"


def probe(url: str) -> dict:
    res = {"url": url}
    res["robots"], res["robots_note"] = robots_ok(url)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=20) as r:
            res["status"] = r.status
            body = r.read()
        root = ET.fromstring(body)
        items = root.findall(".//item") or root.findall(".//{http://www.w3.org/2005/Atom}entry")
        titles = []
        for it in items[:3]:
            tt = it.find("title")
            if tt is None:
                tt = it.find("{http://www.w3.org/2005/Atom}title")
            titles.append((tt.text or "").strip() if tt is not None else "")
        res.update({"ok": len(items) > 0, "items": len(items), "sample": titles})
    except Exception as e:
        res.update({"ok": False, "error": f"{type(e).__name__}: {e}"[:200]})
    return res


def main():
    out = {"at": dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(timespec="seconds"), "feeds": {}}
    for name, urls in CANDIDATES.items():
        out["feeds"][name] = [probe(u) for u in urls]
        for r in out["feeds"][name]:
            print(name, r["url"], r.get("status"), r.get("items"), r.get("robots"), r.get("error", ""))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
