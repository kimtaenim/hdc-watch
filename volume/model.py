"""거래량 HDC 모델 (Binary Spatter Code).

봉 하나 = 요일 ⊕ 시간대 ⊕ 단계 (세 벡터를 XOR로 binding).
기억 = 모든 봉 벡터의 bundling. 칸마다 정수 누산기(accumulator)로 두고, 필요할 때 다수결로 0/1 원형(prototype)을 만든다.
채점 두 가지:
  누산기 점수(본 점수): 봉 벡터(±1)와 누산기의 코사인. 다수결 전 정보를 그대로 쓴다.
  이진 점수(비교용):   봉 벡터와 0/1 원형의 해밍 유사도. BSC의 원형 그대로, 다수결에서 정보를 잃는다.
요일, 시간대, 단계 벡터는 이름에서 만들어지므로 종목과 무관하게 같다.
종목마다 다른 것은 누산기, 중앙값, 단계 경계, 판정 기준이다.
"""

import numpy as np

from hdc import Memory, bind, level_vectors, make_vector, similarity

LADDER_NAME = "거래량단계"


def day_name(wd: str) -> str:
    return f"요일:{wd}"


def session_name(slot: str) -> str:
    return f"시간대:{slot}"


class Codebook:
    """이 모델이 쓰는 모든 기본 벡터. 이름만 알면 어디서든 다시 만들 수 있다."""

    def __init__(self, sessions, levels: int, dim: int, salt: str = ""):
        """salt: 이름 앞에 붙일 글자. 이름을 바꿔 다시 만들면 점수가 얼마나 달라지는지 볼 때만 쓴다."""
        self.dim = dim
        self.days = {wd: make_vector(salt + day_name(wd), dim) for wd in "월화수목금"}
        self.sessions = {s: make_vector(salt + session_name(s), dim) for s in sessions}
        self.levels, self.flips = level_vectors(salt + LADDER_NAME, levels, dim)

    def level(self, k: int) -> np.ndarray:  # k는 1부터
        return self.levels[k - 1]


def bar_vector(cb: Codebook, weekday: str, slot: str, level: int) -> np.ndarray:
    """봉 하나를 벡터 하나로: 요일 ⊕ 시간대 ⊕ 단계."""
    return bind(bind(cb.days[weekday], cb.sessions[slot]), cb.level(level))


def train(cb: Codebook, bars) -> Memory:
    """결측이 아닌 봉 벡터를 모두 더한다. 이것이 모델의 전부다."""
    m = Memory(cb.dim)
    for b in bars:
        if b["level"] is not None:
            m.add(bar_vector(cb, b["weekday"], b["slot"], b["level"]))
    return m


def score(memory_vec: np.ndarray, v: np.ndarray) -> float:
    """점수 = 봉 벡터와 기억의 유사도. 낮을수록 기억에 없던 조합."""
    return similarity(v, memory_vec)


def without(memory: Memory, v: np.ndarray) -> Memory:
    """누산기에서 봉 하나를 뺀 기억. 새 봉은 채점할 때 아직 기억에 없으므로, 학습 봉도 같은 조건에서 채점한다."""
    return Memory(memory.dim, memory.counts - (2 * v.astype(np.int64) - 1), memory.n - 1)


def counter_score(counts: np.ndarray, v: np.ndarray) -> float:
    """누산기 점수 = 봉 벡터(0→-1, 1→+1)와 누산기의 코사인. 범위 -1-1."""
    bip = 2 * v.astype(np.int64) - 1
    norm = float(np.sqrt((counts.astype(np.float64) ** 2).sum()))
    return float((bip * counts).sum()) / (np.sqrt(len(v)) * norm) if norm else 0.0


def counter_score_leave_one_out(memory: Memory, v: np.ndarray) -> float:
    return counter_score(without(memory, v).counts, v)


def score_leave_one_out(memory: Memory, v: np.ndarray) -> float:
    """이진 점수(자기 제외): 다수결로 만든 0/1 원형과의 해밍 유사도."""
    return similarity(v, without(memory, v).to_vector())


def count_closeness(bars, target, levels: int):
    """같은 질문을 단순 카운트로: 같은 요일·시간대의 다른 학습 봉들과 단계가 얼마나 가까운가.
    가까움 = 1 - |단계 차이| / (단계 수 - 1). 수준 벡터 유사도와 같은 꼴이다.
    돌려주는 값: (가까움 평균, 같은 단계 수, 같은 칸 봉 수, 가까움 합)
    누산기 점수의 기댓값은 '가까움 합'에 비례한다(다른 요일·시간대 봉은 기댓값 0). 다수결을 거친 이진 점수는 비례하지 않는다.
    """
    peers = [b["level"] for b in bars
             if b is not target and b["level"] is not None
             and b["weekday"] == target["weekday"] and b["slot"] == target["slot"]]
    if not peers:
        return None, 0, 0, 0.0
    lv = target["level"]
    close = [1 - abs(p - lv) / (levels - 1) for p in peers]
    same = sum(p == lv for p in peers)
    return float(np.mean(close)), same, len(peers), float(np.sum(close))


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
