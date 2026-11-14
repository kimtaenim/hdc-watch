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
    # 2026-11-14 추가 후보: 산업·IT·로봇·에너지 분야
    "한국경제 산업": ["https://www.hankyung.com/feed/industry"],
    "매일경제 기업": ["https://www.mk.co.kr/rss/50100032/", "https://www.mk.co.kr/rss/50200011/"],
    "전자신문 분야": ["https://rss.etnews.com/Section902.xml", "https://rss.etnews.com/Section903.xml", "https://rss.etnews.com/Section904.xml"],
    "로봇신문": ["https://www.irobotnews.com/rss/allArticle.xml"],
    "디일렉": ["https://www.thelec.kr/rss/allArticle.xml"],
    "에너지신문": ["https://www.energy-news.co.kr/rss/allArticle.xml"],
    "에너지경제": ["https://www.ekn.kr/rss/allArticle.xml"],
    "테크엠": ["https://www.techm.kr/rss/allArticle.xml"],
    "아이뉴스24": ["https://www.inews24.com/rss/news_it.xml", "https://www.inews24.com/rss/allArticle.xml"],
    "디지털타임스": ["https://www.dt.co.kr/rss/all.xml", "https://www.dt.co.kr/rss/it.xml"],
    "머니투데이": ["https://rss.mt.co.kr/mt_news.xml"],
    "조선비즈": ["https://biz.chosun.com/arc/outboundfeeds/rss/?outputType=xml"],
    "서울경제": ["https://www.sedaily.com/rss/newsall", "https://www.sedaily.com/RSS/S11.xml"],
    "헬로티": ["https://www.hellot.net/rss/allArticle.xml"],
    "인더스트리뉴스": ["https://www.industrynews.co.kr/rss/allArticle.xml"],
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
        dates = [(it.findtext("pubDate") or "") for it in items]
        res.update({"ok": len(items) > 0, "items": len(items), "sample": titles, "first_date": dates[0] if dates else None,
                    "last_date": dates[-1] if dates else None})
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
