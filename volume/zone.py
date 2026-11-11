"""매물대 모델: 봉 하나 = 가격대 ⊕ 거래량 단계. 어느 가격대에서 거래가 평소보다 쏟아지는지 본다.

시간대 모델(volume/model.py)과 같은 봉, 같은 거래량 단계를 쓰고, 묶는 짝만 '요일 ⊕ 시간대' 대신 '가격대'다.
기억도 따로 둔다.
"""

import math

import numpy as np

from hdc import Memory, bind, make_vector, nearest, similarity

ZONE_PREFIX = "가격대"  # 가격대 k의 벡터 이름은 "가격대:k"


def zone_edges(low: float, high: float, zones: int) -> list[float]:
    """최저가-최고가를 로그 눈금으로 같은 간격 zones칸으로 나눈 경계(zones+1개).
    로그 눈금이면 1,000원→1,050원과 700원→735원이 같은 폭(5%)이 된다."""
    a, b = math.log(low), math.log(high)
    edges = [math.exp(a + (b - a) * k / zones) for k in range(zones + 1)]
    edges[0], edges[-1] = low, high  # 양 끝은 반올림 오차 없이 그대로
    return edges


def zone_of(price: float, edges) -> tuple[int, bool]:
    """가격 → 가격대(1부터). 범위 밖이면 맨 끝 칸에 넣고 두 번째 값으로 알린다."""
    zones = len(edges) - 1
    k = int(np.searchsorted(edges, price, side="right"))
    clamped = price < edges[0] or price > edges[-1]
    return min(max(k, 1), zones), clamped


def zone_name(k: int) -> str:
    return f"{ZONE_PREFIX}:{k}"


class ZoneCodebook:
    """가격대 벡터는 서로 무관한 무작위 벡터다(수준 벡터 아님).

    처음에는 가격대도 수준 벡터(이웃끼리 닮은 사다리)로 만들었다. 그런데 묶은 벡터끼리의 유사도는
    재료 유사도의 곱이라(sim(a⊕b, c⊕d) ≈ sim(a,c) × sim(b,d)), 사다리 둘을 묶으면 가격대가 다른 조합도 서로 닮는다.
    기억에 더하면 한 중심으로 뭉쳐, 가격대마다 풀어 물어도 12칸 모두 같은 답(6단계)이 나왔다.
    """

    def __init__(self, zones: int, levels, dim: int):
        self.dim = dim
        self.zones = [make_vector(zone_name(k), dim) for k in range(1, zones + 1)]
        self.levels = levels  # 시간대 모델과 같은 거래량 단계 벡터

    def zone(self, k: int) -> np.ndarray:
        return self.zones[k - 1]

    def level(self, k: int) -> np.ndarray:
        return self.levels[k - 1]


def bar_vector(cb: ZoneCodebook, zone: int, level: int) -> np.ndarray:
    """봉 하나 = 가격대 ⊕ 거래량 단계."""
    return bind(cb.zone(zone), cb.level(level))


def train(cb: ZoneCodebook, bars) -> Memory:
    m = Memory(cb.dim)
    for b in bars:
        m.add(bar_vector(cb, b["zone"], b["level"]))
    return m


def score_leave_one_out(memory: Memory, v: np.ndarray) -> float:
    rest = Memory(memory.dim, memory.counts - (2 * v.astype(np.int64) - 1), memory.n - 1)
    return similarity(v, rest.to_vector())


def ask_zone(cb: ZoneCodebook, memory_vec: np.ndarray, zone: int):
    """기억을 가격대 벡터로 풀어 '이 가격대에서 흔한 거래량 단계'를 묻는다.
    풀린 벡터와 각 단계 벡터의 유사도를 돌려준다(단계 1부터 순서대로)."""
    q = bind(memory_vec, cb.zone(zone))
    book = {k: cb.level(k) for k in range(1, len(cb.levels) + 1)}
    sims = dict(nearest(q, book))
    return q, [sims[k] for k in range(1, len(cb.levels) + 1)]
