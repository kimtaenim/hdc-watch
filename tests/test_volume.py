"""거래량 앞단(봉 만들기, 단계 나누기)과 모델 확인. 네트워크 없이 돈다."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hdc import similarity
from volume import data, model

SESSIONS = [9, 10, 11, 12, 13, 14, 15]
EDGES = 2.0 ** np.arange(-4, 7)


class TestBars(unittest.TestCase):
    def test_derived_session(self):
        hourly = [{"date": "2026-11-10", "hour": h, "volume": 100 * h} for h in range(9, 15)]
        hourly += [{"date": "2026-11-11", "hour": h, "volume": 1000} for h in range(9, 15)]
        daily = {"2026-11-10": 10000, "2026-11-11": 5000}
        bars, problems = data.build_bars(hourly, daily, ["2026-11-10", "2026-11-11"], SESSIONS, 15)
        b15 = [b for b in bars if b["hour"] == 15]
        self.assertEqual(b15[0]["volume"], 10000 - sum(100 * h for h in range(9, 15)))
        self.assertEqual(b15[0]["volume"] + b15[0]["hourly_sum"], 10000)
        self.assertIsNone(b15[1]["volume"])  # 5000 - 6000 < 0 → 결측
        self.assertEqual(len(problems), 1)
        self.assertEqual(bars[0]["weekday"], "화")

    def test_zero_first_session_is_missing(self):
        hourly = [{"date": "2026-11-09", "hour": h, "volume": 0 if h == 9 else 1000} for h in range(9, 15)]
        bars, problems = data.build_bars(hourly, {"2026-11-09": 9000}, ["2026-11-09"], SESSIONS, 15, [9])
        self.assertIsNotNone(bars[0]["missing"])   # 9시대
        self.assertIsNone(bars[1]["missing"])      # 10시대
        self.assertIsNone(bars[-1]["volume"])      # 15시대 역산도 결측
        self.assertTrue(data.after_break("2026-11-09", ["2026-11-06", "2026-11-09"]))
        self.assertFalse(data.after_break("2026-11-10", ["2026-11-09", "2026-11-10"]))

    def test_session_used_only_for_derivation(self):
        hourly = [{"date": "2026-11-10", "hour": h, "volume": 1000} for h in range(9, 15)]
        bars, _ = data.build_bars(hourly, {"2026-11-10": 9000}, ["2026-11-10"], [10, 11, 12, 13, 14, 15], 15,
                                  [9], [9, 10, 11, 12, 13, 14])
        self.assertEqual([b["hour"] for b in bars], [10, 11, 12, 13, 14, 15])
        self.assertEqual(bars[-1]["volume"], 3000)  # 9시대 1000주도 뺀다

    def test_levels(self):
        self.assertEqual(data.level_of(0.0, EDGES), 1)
        self.assertEqual(data.level_of(1.0, EDGES), 6)
        self.assertEqual(data.level_of(1.999, EDGES), 6)
        self.assertEqual(data.level_of(2.0, EDGES), 7)
        self.assertEqual(data.level_of(16.0, EDGES), 10)
        self.assertEqual(data.level_of(64.0, EDGES), 12)
        self.assertEqual(data.level_of(1e9, EDGES), 12)

    def test_window(self):
        days = ["2026-05-08", "2026-05-11", "2026-06-01", "2026-11-09", "2026-11-10"]
        got, start = data.window(days, "2026-11-09", 6)
        self.assertEqual(start, "2026-05-09")
        self.assertEqual(got, ["2026-05-11", "2026-06-01", "2026-11-09"])


class TestModel(unittest.TestCase):
    def test_rare_level_scores_lower(self):
        """같은 요일·시간대에 6단계만 쌓인 기억에서, 6단계 새 봉은 높게, 12단계 새 봉은 낮게 나와야 한다."""
        cb = model.Codebook(SESSIONS, 12, 10000)
        bars = [{"weekday": wd, "hour": h, "level": 6} for wd in "월화수목금" for h in SESSIONS for _ in range(5)]
        mem = model.train(cb, bars)
        mv = mem.to_vector()
        usual = model.score(mv, model.bar_vector(cb, "수", 14, 6))
        near = model.score(mv, model.bar_vector(cb, "수", 14, 7))
        rare = model.score(mv, model.bar_vector(cb, "수", 14, 12))
        self.assertGreater(usual, near)
        self.assertGreater(near, rare)

    def test_leave_one_out_equals_training_without_it(self):
        cb = model.Codebook(SESSIONS, 12, 2000)
        bars = [{"weekday": wd, "hour": 9 + i % 7, "level": 1 + i % 12} for i, wd in enumerate("월화수목금" * 7)]
        mem = model.train(cb, bars)
        v = model.bar_vector(cb, bars[3]["weekday"], bars[3]["hour"], bars[3]["level"])
        without = model.train(cb, bars[:3] + bars[4:])
        self.assertEqual(model.score_leave_one_out(mem, v), similarity(v, without.to_vector()))

    def test_spearman(self):
        self.assertAlmostEqual(model.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(model.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)


if __name__ == "__main__":
    unittest.main()
