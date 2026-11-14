"""트렌드: 키워드 찾기, 명사 뽑기, 새 조합 판정. 네트워크 없이 가짜 제목으로."""

import datetime as dt
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trend import analyze, collect, text

KW = [{"id": "ai-infra", "name": "AI 인프라", "variants": ["AI 인프라", "AI인프라", "인공지능 인프라"]},
      {"id": "robot", "name": "로봇", "variants": ["로봇"]}]


class TestText(unittest.TestCase):
    def test_find_keywords(self):
        f = text.find_keywords("AI인프라 투자에 로봇개까지", KW)
        self.assertEqual(set(f), {"ai-infra", "robot"})

    def test_nouns_merge_and_skip(self):
        title = "AI 인프라 확충에 데이터센터 전력 수요 급증"
        spans = text.find_keywords(title, KW)["ai-infra"]
        ws = text.nouns(title, spans)
        self.assertIn("데이터센터", ws)          # 붙은 조각 합치기
        self.assertNotIn("인프라", ws)           # 그 키워드 표기는 뺀다
        self.assertNotIn("AI", ws)


class TestEnglish(unittest.TestCase):
    def test_english_keyword_whole_word(self):
        kw = [{"id": "robot", "name": "로봇", "variants": ["로봇", "Robot", "Robotics"]}]
        self.assertIn("robot", text.find_keywords("New robots for warehouses", kw))
        self.assertNotIn("robot", text.find_keywords("Robotaxi rollout expands", kw))
        self.assertIn("robot", text.find_keywords("로봇개 상용화", kw))  # 한국어 표기는 붙은 말도 센다

    def test_english_words(self):
        title = "Fluence wins 400MWh BESS contract in Australia"
        kw = [{"id": "bess", "name": "BESS", "variants": ["BESS"]}]
        ws = text.nouns(title, text.find_keywords(title, kw)["bess"])
        self.assertEqual(ws, ["fluence", "mwh", "contract", "australia"])  # wins, in은 흔한 낱말이라 뺀다


class TestNewCombination(unittest.TestCase):
    def test_new_word_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            collect.TITLES = os.path.join(tmp, "titles")
            analyze.OUT = os.path.join(tmp, "out")
            os.makedirs(collect.TITLES)
            today = dt.date(2026, 11, 11)
            for i in range(1, 21):  # 과거 20일: 데이터센터, 전력만
                d = (today - dt.timedelta(days=i)).isoformat()
                rows = [{"title": f"AI 인프라 데이터센터 전력 확충 {j}", "link": f"p{i}-{j}", "date": d, "time": "09:00", "source": "가짜"}
                        for j in range(5)]
                json.dump(rows, open(os.path.join(collect.TITLES, f"{d}.json"), "w", encoding="utf-8"), ensure_ascii=False)
            rows = [{"title": "AI 인프라 데이터센터 냉각 경쟁", "link": f"t{j}", "date": today.isoformat(), "time": "10:00", "source": "가짜"}
                    for j in range(3)]
            json.dump(rows, open(os.path.join(collect.TITLES, f"{today}.json"), "w", encoding="utf-8"), ensure_ascii=False)
            cfg = json.load(open(analyze.CONFIG, encoding="utf-8"))
            tc = cfg["trend"]
            kw = KW[0]
            pt = analyze.pairs_of(json.load(open(os.path.join(collect.TITLES, f"{today}.json"))), kw)
            pp = analyze.pairs_of(collect.load_titles([(today - dt.timedelta(days=i)).isoformat() for i in range(1, 21)]), kw)
            from hdc import D, Memory, bind, make_vector
            kv = make_vector(analyze.kw_name(kw["id"]))
            mp = Memory(D)
            for _, ws in pp:
                for w in ws:
                    mp.add(bind(kv, make_vector(analyze.word_name(w))))
            thr = tc["noise_k"] / D ** 0.5
            s_new = analyze.ask(mp.counts, kv, make_vector(analyze.word_name("냉각")))
            s_old = analyze.ask(mp.counts, kv, make_vector(analyze.word_name("데이터센터")))
            self.assertLess(s_new, thr)       # 과거에 없던 '냉각'은 잡음 수준
            self.assertGreater(s_old, 0.3)    # 늘 나오던 '데이터센터'는 뚜렷
            self.assertEqual(sorted(analyze.word_counts(pt)), ["경쟁", "냉각", "데이터센터"])


if __name__ == "__main__":
    unittest.main()
