"""벡터 만들기: 이름에서 1만 비트 벡터를, 그리고 수준 벡터 사다리를 만든다.

규칙(파이썬과 브라우저가 똑같이 따라야 하는 약속)
① 이름 + "#" + 일련번호(0, 1, 2, ...)를 UTF-8로 바꿔 SHA-256 해시를 낸다.
② 해시 한 번이 32바이트 = 256비트. 바이트마다 큰 자리 비트부터 꺼내 이어 붙인다.
③ D비트가 찰 때까지 반복하고, 남는 비트는 버린다. (D=10,000이면 해시 40번, 10,240비트 중 앞 10,000비트)

난수 생성기를 쓰지 않으므로 같은 이름은 언제 어디서나 같은 벡터가 된다.
그래서 벡터를 파일로 저장할 필요가 없다.
"""

import hashlib

import numpy as np

D = 10_000  # 기본 차원


def sha256(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


def make_vector(name: str, dim: int = D) -> np.ndarray:
    """이름 하나 → 0과 1로 된 dim칸 벡터."""
    chunks = []
    i = 0
    while len(chunks) * 256 < dim:
        chunks.append(sha256(f"{name}#{i}"))
        i += 1
    raw = np.frombuffer(b"".join(chunks), dtype=np.uint8)
    bits = np.unpackbits(raw)  # 바이트마다 큰 자리 비트부터
    return bits[:dim].copy()


def flip_order(name: str, dim: int = D) -> list[int]:
    """수준 벡터에서 어느 칸을 어떤 순서로 뒤집을지 정한다.

    칸 p마다 "뒤집기|이름|p"의 SHA-256 앞 8바이트를 제비 번호로 주고,
    번호가 작은 칸부터 줄 세운다(번호가 같으면 칸 번호가 작은 쪽이 먼저).
    """
    keys = [int.from_bytes(sha256(f"뒤집기|{name}|{p}")[:8], "big") for p in range(dim)]
    return sorted(range(dim), key=lambda p: (keys[p], p))


def level_boundaries(levels: int, dim: int = D) -> list[int]:
    """k단계까지 누적으로 뒤집은 칸 수. 마지막 단계에서 정확히 dim/2가 된다."""
    half = dim // 2
    return [(k * half) // (levels - 1) for k in range(levels)]


def level_vectors(name: str, levels: int = 12, dim: int = D):
    """수준 벡터 사다리.

    1단계 = make_vector(name).
    한 단계 오를 때마다 flip_order의 다음 묶음(겹치지 않는 칸들)을 뒤집는다.
    돌려주는 값: (벡터 목록, 단계마다 뒤집은 칸 목록)
      flips[k]는 (k+1)단계 → (k+2)단계로 갈 때 뒤집은 칸들 (k는 0부터)
    """
    base = make_vector(name, dim)
    order = flip_order(name, dim)
    bounds = level_boundaries(levels, dim)
    vectors = [base]
    flips = []
    for k in range(1, levels):
        cells = order[bounds[k - 1]:bounds[k]]
        v = vectors[-1].copy()
        v[cells] ^= 1  # 이 칸들만 0↔1
        vectors.append(v)
        flips.append(cells)
    return vectors, flips
