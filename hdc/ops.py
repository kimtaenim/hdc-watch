"""HDC의 네 가지 연산: 묶기, 비교, 더하기(기억), 풀어서 묻기.

벡터는 0과 1로 된 numpy 배열(uint8)이다.
"""

import json
import math

import numpy as np

from .vectors import make_vector

TIE_NAME = "__동점깨기__"  # 더하기 결과가 0:0으로 비긴 칸을 정할 고정 벡터의 이름


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """묶기 = 칸마다 XOR. 같으면 0, 다르면 1.
    XOR을 한 번 더 하면 원래대로 돌아오므로 풀기(unbind)도 같은 함수다."""
    return np.bitwise_xor(a, b)


unbind = bind


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    """해밍 거리 = XOR한 뒤 1의 개수 = 서로 다른 칸의 수."""
    return int(np.count_nonzero(np.bitwise_xor(a, b)))


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """유사도 = 1 - 2 × 거리 / D.
    완전히 같으면 1, 무관하면 0 근처, 정반대면 -1."""
    return 1.0 - 2.0 * hamming(a, b) / len(a)


def noise_level(dim: int) -> float:
    """무관한 두 벡터의 유사도가 흔들리는 폭(표준편차) = 1/√D.
    (칸마다 같을 확률 1/2인 동전 D개 → 다른 칸 수는 이항분포, 표준편차 √D/2)"""
    return 1.0 / math.sqrt(dim)


class Memory:
    """더하기(bundle)로 만든 기억.

    칸마다 정수 카운터를 둔다. 들어온 벡터의 칸이 1이면 +1, 0이면 -1.
    비교할 때는 다수결로 0/1 벡터를 만든다(양수→1, 음수→0, 0이면 동점깨기 벡터의 칸을 따른다).
    """

    def __init__(self, dim: int, counts=None, n: int = 0):
        self.dim = dim
        self.counts = np.zeros(dim, dtype=np.int64) if counts is None else np.asarray(counts, dtype=np.int64)
        self.n = n  # 더한 벡터 수

    def add(self, v: np.ndarray, weight: int = 1) -> None:
        self.counts += weight * (2 * v.astype(np.int64) - 1)  # 1→+1, 0→-1
        self.n += weight

    def to_vector(self) -> np.ndarray:
        tie = make_vector(TIE_NAME, self.dim)
        out = (self.counts > 0).astype(np.uint8)
        zero = self.counts == 0
        out[zero] = tie[zero]
        return out

    def tie_count(self) -> int:
        return int(np.count_nonzero(self.counts == 0))

    def to_json(self) -> dict:
        return {"dim": self.dim, "n": self.n, "counts": self.counts.tolist()}

    @classmethod
    def from_json(cls, data: dict) -> "Memory":
        return cls(data["dim"], data["counts"], data["n"])

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, separators=(",", ":"))

    @classmethod
    def load(cls, path) -> "Memory":
        with open(path, encoding="utf-8") as f:
            return cls.from_json(json.load(f))


def nearest(query: np.ndarray, codebook: dict) -> list:
    """후보 벡터 목록(이름→벡터) 중 query와 가까운 순으로 (이름, 유사도)를 돌려준다."""
    scored = [(name, similarity(query, v)) for name, v in codebook.items()]
    return sorted(scored, key=lambda x: -x[1])
