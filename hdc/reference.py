"""읽기용 기준 구현. numpy 없이 파이썬 리스트와 for문만 쓴다.

hdc/vectors.py, hdc/ops.py(빠른 경로)와 결과가 같은지 tests에서 확인한다.
느리지만 한 줄씩 따라 읽을 수 있게 쓴다.
"""

import hashlib


def make_vector(name, dim):
    bits = []
    i = 0
    while len(bits) < dim:
        digest = hashlib.sha256(f"{name}#{i}".encode("utf-8")).digest()
        for byte in digest:
            for k in range(7, -1, -1):  # 큰 자리 비트부터
                bits.append((byte >> k) & 1)
        i += 1
    return bits[:dim]


def flip_order(name, dim):
    tickets = []
    for p in range(dim):
        digest = hashlib.sha256(f"뒤집기|{name}|{p}".encode("utf-8")).digest()
        tickets.append((int.from_bytes(digest[:8], "big"), p))
    tickets.sort()
    return [p for _, p in tickets]


def level_vectors(name, levels, dim):
    order = flip_order(name, dim)
    half = dim // 2
    current = make_vector(name, dim)
    vectors = [list(current)]
    done = 0
    for k in range(1, levels):
        upto = (k * half) // (levels - 1)
        for p in order[done:upto]:
            current[p] = 1 - current[p]
        done = upto
        vectors.append(list(current))
    return vectors


def bind(a, b):
    return [x ^ y for x, y in zip(a, b)]


def hamming(a, b):
    count = 0
    for x, y in zip(a, b):
        if x != y:
            count += 1
    return count


def similarity(a, b):
    return 1 - 2 * hamming(a, b) / len(a)


def bundle(vectors, dim, tie_name="__동점깨기__"):
    counts = [0] * dim
    for v in vectors:
        for i in range(dim):
            counts[i] += 1 if v[i] == 1 else -1
    tie = make_vector(tie_name, dim)
    out = []
    for i in range(dim):
        if counts[i] > 0:
            out.append(1)
        elif counts[i] < 0:
            out.append(0)
        else:
            out.append(tie[i])
    return counts, out
