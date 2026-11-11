"""거래량 HDC 모델: 봉 하나 = 요일 ⊕ 시간대 ⊕ 단계, 모델 = 모든 봉 벡터를 더한 기억.

요일, 시간대, 단계 벡터는 이름에서 만들어지므로 종목과 무관하게 같다.
종목마다 다른 것은 기억(카운터)과 중앙값, 판정 기준뿐이다.
"""

import numpy as np

from hdc import Memory, bind, level_vectors, make_vector, similarity

LADDER_NAME = "거래량단계"


def day_name(wd: str) -> str:
    return f"요일:{wd}"


def session_name(hour: int) -> str:
    return f"시간대:{hour}"


class Codebook:
    """이 모델이 쓰는 모든 기본 벡터. 이름만 알면 어디서든 다시 만들 수 있다."""

    def __init__(self, sessions, levels: int, dim: int):
        self.dim = dim
        self.days = {wd: make_vector(day_name(wd), dim) for wd in "월화수목금"}
        self.sessions = {s: make_vector(session_name(s), dim) for s in sessions}
        self.levels, self.flips = level_vectors(LADDER_NAME, levels, dim)

    def level(self, k: int) -> np.ndarray:  # k는 1부터
        return self.levels[k - 1]


def bar_vector(cb: Codebook, weekday: str, hour: int, level: int) -> np.ndarray:
    """봉 하나를 벡터 하나로: 요일 ⊕ 시간대 ⊕ 단계."""
    return bind(bind(cb.days[weekday], cb.sessions[hour]), cb.level(level))


def train(cb: Codebook, bars) -> Memory:
    """결측이 아닌 봉 벡터를 모두 더한다. 이것이 모델의 전부다."""
    m = Memory(cb.dim)
    for b in bars:
        if b["level"] is not None:
            m.add(bar_vector(cb, b["weekday"], b["hour"], b["level"]))
    return m


def score(memory_vec: np.ndarray, v: np.ndarray) -> float:
    """점수 = 봉 벡터와 기억의 유사도. 낮을수록 기억에 없던 조합."""
    return similarity(v, memory_vec)


def score_leave_one_out(memory: Memory, v: np.ndarray) -> float:
    """자기 자신을 기억에서 뺀 뒤 채점한다.
    새 봉은 채점할 때 아직 기억에 없으므로, 학습 봉도 같은 조건에서 채점해야 판정 기준이 공정하다."""
    rest = Memory(memory.dim, memory.counts - (2 * v.astype(np.int64) - 1), memory.n - 1)
    return similarity(v, rest.to_vector())


def count_closeness(bars, target, levels: int):
    """같은 질문을 단순 카운트로: 같은 요일·시간대의 다른 학습 봉들과 단계가 얼마나 가까운가.
    가까움 = 평균(1 - |단계 차이| / (단계 수 - 1)). 수준 벡터 유사도와 같은 꼴이다.
    """
    peers = [b["level"] for b in bars
             if b is not target and b["level"] is not None
             and b["weekday"] == target["weekday"] and b["hour"] == target["hour"]]
    if not peers:
        return None, 0, 0
    lv = target["level"]
    closeness = float(np.mean([1 - abs(p - lv) / (levels - 1) for p in peers]))
    same = sum(p == lv for p in peers)
    return closeness, same, len(peers)


def ranks(x) -> np.ndarray:
    """순위(1부터). 같은 값은 평균 순위를 준다."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        same = x == v
        r[same] = r[same].mean()
    return r


def spearman(x, y) -> float:
    """두 줄 세우기가 얼마나 같은가(순위 상관). 1이면 순서가 완전히 같다."""
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])
