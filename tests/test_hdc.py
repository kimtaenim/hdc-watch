"""공용 HDC 부품 확인. 실행: python -m unittest discover -s tests -v"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hdc import (D, Memory, bind, hamming, level_boundaries, level_vectors, make_vector,
                 nearest, noise_level, similarity, unbind)
from hdc import reference as ref
from hdc.trace import Trace, bits_from_b64, check, vec_info, why

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestVectors(unittest.TestCase):
    def test_shape_and_balance(self):
        v = make_vector("요일:수")
        self.assertEqual(v.shape, (D,))
        self.assertTrue(set(np.unique(v)) <= {0, 1})
        # 1의 개수는 대략 절반 (이항분포, 표준편차 50)
        self.assertLess(abs(int(v.sum()) - D // 2), 250)

    def test_same_name_same_vector(self):
        self.assertTrue(np.array_equal(make_vector("시간대:14"), make_vector("시간대:14")))

    def test_unrelated_similarity_near_zero(self):
        # 서로 다른 이름 300쌍
        sims = np.array([similarity(make_vector(f"가{i}"), make_vector(f"나{i}")) for i in range(300)])
        inside = np.mean(np.abs(sims) <= 0.03)
        print(f"\n  무관한 300쌍: 평균 {sims.mean():+.4f}, 표준편차 {sims.std():.4f}, "
              f"최대 |유사도| {np.abs(sims).max():.4f}, ±0.03 안 비율 {inside:.3f}")
        self.assertLess(abs(sims.mean()), 0.003)
        self.assertAlmostEqual(sims.std(), noise_level(D), delta=0.002)
        self.assertGreaterEqual(inside, 0.99)

    def test_noise_shrinks_with_dimension(self):
        for dim in (1000, 10000):
            sims = np.array([similarity(make_vector(f"a{i}", dim), make_vector(f"b{i}", dim))
                             for i in range(300)])
            self.assertAlmostEqual(sims.std(), noise_level(dim), delta=noise_level(dim) * 0.2)


class TestBind(unittest.TestCase):
    def test_unbind_recovers(self):
        a, b = make_vector("요일:수"), make_vector("시간대:14")
        ab = bind(a, b)
        self.assertTrue(np.array_equal(unbind(ab, b), a))
        self.assertTrue(np.array_equal(unbind(ab, a), b))

    def test_bound_is_unlike_inputs(self):
        a, b = make_vector("요일:수"), make_vector("시간대:14")
        ab = bind(a, b)
        self.assertLess(abs(similarity(ab, a)), 0.03)
        self.assertLess(abs(similarity(ab, b)), 0.03)


class TestLevels(unittest.TestCase):
    def test_even_decay(self):
        vecs, flips = level_vectors("거래량단계", 12)
        bounds = level_boundaries(12)
        # 단계마다 뒤집은 칸은 겹치지 않는다
        allcells = [p for f in flips for p in f]
        self.assertEqual(len(allcells), len(set(allcells)))
        self.assertEqual(len(allcells), D // 2)
        print("\n  단계마다 뒤집은 칸 수:", [len(f) for f in flips])
        # 두 단계 사이 거리 = 그 사이에 뒤집은 칸 수 (정확히)
        for i in range(12):
            for j in range(12):
                self.assertEqual(hamming(vecs[i], vecs[j]), abs(bounds[i] - bounds[j]))
        sims = [similarity(vecs[0], v) for v in vecs]
        print("  1단계와의 유사도:", [round(s, 3) for s in sims])
        self.assertEqual(sims[-1], 0.0)
        steps = np.diff(sims)
        self.assertTrue(np.all(steps < 0))
        self.assertLess(steps.max() - steps.min(), 2.5 / D)  # 454칸 또는 455칸

    def test_other_dimension(self):
        vecs, _ = level_vectors("거래량단계", 12, 1000)
        self.assertEqual(similarity(vecs[0], vecs[-1]), 0.0)


class TestReferenceMatchesFast(unittest.TestCase):
    """빠른 경로(numpy)와 읽기용 기준 구현이 같은 답을 내는가."""

    def test_vectors(self):
        for name, dim in [("요일:월", D), ("피지컬 AI", D), ("x", 1000), ("x", 333)]:
            self.assertEqual(make_vector(name, dim).tolist(), ref.make_vector(name, dim))

    def test_levels(self):
        fast, _ = level_vectors("거래량단계", 12, 2000)
        slow = ref.level_vectors("거래량단계", 12, 2000)
        self.assertEqual([v.tolist() for v in fast], slow)

    def test_bind_compare_bundle(self):
        dim = 2000
        a, b, c = (make_vector(n, dim) for n in ("A", "B", "C"))
        self.assertEqual(bind(a, b).tolist(), ref.bind(a.tolist(), b.tolist()))
        self.assertEqual(hamming(a, b), ref.hamming(a.tolist(), b.tolist()))
        self.assertEqual(similarity(a, b), ref.similarity(a.tolist(), b.tolist()))
        m = Memory(dim)
        for v in (a, b, c, bind(a, c)):  # 짝수 개 → 동점 칸이 생긴다
            m.add(v)
        counts, vec = ref.bundle([v.tolist() for v in (a, b, c, bind(a, c))], dim)
        self.assertGreater(m.tie_count(), 0)
        self.assertEqual(m.counts.tolist(), counts)
        self.assertEqual(m.to_vector().tolist(), vec)


class TestMemory(unittest.TestCase):
    def test_unbind_query_finds_level(self):
        """기억에 (요일 × 시간대 × 단계)를 여러 개 더한 뒤 (요일 × 시간대)로 풀어 물으면 단계가 나오는가."""
        levels, _ = level_vectors("거래량단계", 12)
        day = {d: make_vector(f"요일:{d}") for d in "월화수목금"}
        hour = {h: make_vector(f"시간대:{h}") for h in range(9, 16)}
        truth = {}
        m = Memory(D)
        for i, d in enumerate("월화수목금"):
            for j, h in enumerate(range(9, 16)):
                lv = (i * 7 + j * 5) % 12
                truth[(d, h)] = lv
                m.add(bind(bind(day[d], hour[h]), levels[lv]))
        mv = m.to_vector()
        book = {k + 1: v for k, v in enumerate(levels)}
        hits = 0
        for (d, h), lv in truth.items():
            best = nearest(unbind(mv, bind(day[d], hour[h])), book)[0][0]
            hits += best == lv + 1
        print(f"\n  35조합을 더한 기억에서 풀어 물은 단계 맞힌 수: {hits}/35")
        self.assertEqual(hits, 35)

    def test_save_load(self):
        m = Memory(D)
        m.add(make_vector("A"))
        m.add(make_vector("B"), weight=3)
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "m.json")
            m.save(p)
            m2 = Memory.load(p)
        self.assertEqual(m2.n, 4)
        self.assertTrue(np.array_equal(m.counts, m2.counts))


class TestTrace(unittest.TestCase):
    def test_roundtrip(self):
        t = Trace("test", params={"D": D})
        v = make_vector("A")
        t.step("벡터 꺼내기", "벡터 생성", [why("수학적 사실", "SHA-256으로 만든다")],
               outputs=[vec_info("A", v, full=True)], check=check("1의 개수", int(v.sum()), int(v.sum())),
               source=make_vector)
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "t.json")
            t.save(p)
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        s = data["steps"][0]
        self.assertEqual(s["source"]["file"], "hdc/vectors.py")
        self.assertTrue(np.array_equal(bits_from_b64(s["outputs"][0]["bits_b64"], D), v))
        self.assertTrue(s["check"]["agree"])


@unittest.skipIf(shutil.which("node") is None, "node 없음")
class TestPythonMatchesJS(unittest.TestCase):
    """파이썬과 브라우저판(docs/js/hdc.js)이 같은 이름에서 같은 벡터를 내는가."""

    def test_same_bits(self):
        req = {
            "vectors": [{"name": n, "dim": d} for n, d in
                        [("요일:수", D), ("시간대:14", D), ("피지컬 AI", D), ("Physical AI", D),
                         ("x", 1000), ("이모지🙂", D), ("", D), ("x", 333)]],
            "levels": [{"name": "거래량단계", "levels": 12, "dim": D},
                       {"name": "거래량단계", "levels": 12, "dim": 1000}],
            "bundles": [{"pairs": [["피지컬 AI", "로봇"], ["피지컬 AI", "휴머노이드"],
                                   ["AI 인프라", "데이터센터"], ["AI 인프라", "전력"]], "dim": D}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "req.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump(req, f, ensure_ascii=False)
            res = subprocess.run(["node", os.path.join(ROOT, "tests", "js_vectors.mjs"), p],
                                 capture_output=True, text=True, check=True)
        out = json.loads(res.stdout)

        def hexbits(v):
            return np.packbits(v).tobytes().hex()

        for spec, js in zip(req["vectors"], out["vectors"]):
            self.assertEqual(hexbits(make_vector(spec["name"], spec["dim"])), js, spec)
        for spec, js in zip(req["levels"], out["levels"]):
            vecs, _ = level_vectors(spec["name"], spec["levels"], spec["dim"])
            self.assertEqual([hexbits(v) for v in vecs], js)
        for spec, js in zip(req["bundles"], out["bundles"]):
            m = Memory(spec["dim"])
            for a, b in spec["pairs"]:
                m.add(bind(make_vector(a), make_vector(b)))
            self.assertEqual(m.counts.tolist(), js["counts"])
            self.assertEqual(hexbits(m.to_vector()), js["vector"])
        print(f"\n  파이썬과 JS 일치: 벡터 {len(req['vectors'])}개, 수준 사다리 {len(req['levels'])}개, 기억 1개")


if __name__ == "__main__":
    unittest.main()
