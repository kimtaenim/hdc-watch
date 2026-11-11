"""공용 부품 자체 점검을 추적 기록으로 남긴다. 추적 기록 형식의 실제 예시이기도 하다.

실행: python -m hdc.selfcheck  → docs/data/traces/hdc/selfcheck.json
"""

import os

import numpy as np

from . import D, Memory, bind, hamming, level_vectors, make_vector, nearest, noise_level, similarity, unbind
from .trace import REPO_ROOT, Trace, check, source_of, vec_info, why

OUT = os.path.join(REPO_ROOT, "docs", "data", "traces", "hdc", "selfcheck.json")


def run(path: str = OUT) -> dict:
    t = Trace("hdc.selfcheck", params={"D": D, "levels": 12})

    # ① 벡터 생성
    a, b = make_vector("요일:수"), make_vector("시간대:14")
    t.step("이름에서 벡터 꺼내기", "벡터 생성", [
        why("설계 선택", "난수 생성기 대신 SHA-256으로 '이름#0', '이름#1', ...을 해시해 1만 비트를 이어 붙인다. "
                       "같은 이름은 파이썬에서도 브라우저에서도 같은 벡터가 되므로 벡터를 저장하지 않는다."),
        why("설계 선택", "SHA-256 출력을 무작위 비트처럼 볼 수 있다고 가정한다. 증명된 정리는 아니고 널리 받아들이는 가정이다. "
                       "그 가정 아래 1의 개수는 약 5,000개가 된다."),
    ], inputs=["요일:수", "시간대:14"],
       outputs=[vec_info("요일:수", a, full=True), vec_info("시간대:14", b, full=True)],
       source=make_vector)

    # ② 무관한 두 벡터 비교
    h = hamming(a, b)
    t.step("무관한 두 벡터 비교", "비교", [
        why("수학적 사실", f"무관한 두 벡터는 칸마다 같을 확률이 1/2이라, 다른 칸 수는 평균 D/2, "
                       f"유사도는 0 근처에서 표준편차 약 1/√D = {noise_level(D):.3f}로 흔들린다(근사)."),
    ], inputs=[vec_info("요일:수", a), vec_info("시간대:14", b)],
       outputs=[{"type": "value", "name": "해밍 거리", "value": h},
                {"type": "value", "name": "유사도", "value": similarity(a, b)},
                {"type": "value", "name": "잡음 수준(1/√D)", "value": noise_level(D)}],
       check=check("다른 칸 수(한 칸씩 직접 센 값)", int(np.sum(a != b)), h),
       source=hamming)

    # ③ 묶기와 풀기
    ab = bind(a, b)
    back = unbind(ab, b)
    t.step("묶고 다시 풀기", "묶기", [
        why("수학적 사실", "XOR은 같은 값을 두 번 하면 원래대로 돌아온다: (A⊕B)⊕B = A. 그래서 묶기와 풀기가 같은 연산이다."),
        why("수학적 사실", "A, B가 서로 무관한 무작위 벡터면, 묶은 결과 A⊕B는 A와도 B와도 무관한(유사도 0 근처) 새 벡터가 된다."),
    ], inputs=[vec_info("요일:수", a, full=True), vec_info("시간대:14", b, full=True)],
       outputs=[vec_info("요일:수 ⊕ 시간대:14", ab, full=True),
                {"type": "value", "name": "결과와 A의 유사도", "value": similarity(ab, a)},
                {"type": "value", "name": "결과와 B의 유사도", "value": similarity(ab, b)},
                {"type": "value", "name": "(A⊕B)⊕B와 A의 유사도", "value": similarity(back, a)}],
       check=check("(A⊕B)⊕B가 A와 칸마다 같은가", True, bool(np.array_equal(back, a))),
       source=bind)

    # ④ 수준 벡터
    levels, flips = level_vectors("거래량단계", 12)
    t.step("수준 벡터 사다리 만들기", "뒤집기", [
        why("설계 선택", "1단계는 이름에서 꺼낸 벡터, 단계가 오를 때마다 아직 안 뒤집은 칸 약 455개를 뒤집는다. "
                       "11번 오르면 정확히 5,000칸(절반)이 바뀐다."),
        why("수학적 사실", "뒤집은 칸이 겹치지 않으므로 두 단계의 해밍 거리는 그 사이에 뒤집은 칸 수와 정확히 같다. "
                       "거리가 단계 차이에 (반올림 오차 안에서) 비례하므로, 유사도는 1에서 0까지 일정하게 줄어든다."),
    ], inputs=["거래량단계"],
       outputs=[vec_info(f"{k + 1}단계", v, full=(k < 2)) for k, v in enumerate(levels)]
               + [{"type": "flips", "name": "1→2단계에서 뒤집은 칸", "cells": flips[0]},
                  {"type": "table", "name": "1단계와의 유사도",
                   "rows": [[k + 1, similarity(levels[0], v)] for k, v in enumerate(levels)]}],
       check=check("1단계와 12단계의 다른 칸 수", D // 2, hamming(levels[0], levels[-1])),
       source=level_vectors)

    # ⑤ 더하기와 풀어 묻기
    m = Memory(D)
    stored = {("월", 9): 3, ("수", 14): 9, ("금", 11): 6}
    for (d, hr), lv in stored.items():
        m.add(bind(bind(make_vector(f"요일:{d}"), make_vector(f"시간대:{hr}")), levels[lv - 1]))
    mv = m.to_vector()
    q = unbind(mv, bind(make_vector("요일:수"), make_vector("시간대:14")))
    ranking = nearest(q, {f"{k + 1}단계": v for k, v in enumerate(levels)})
    t.step("프로토타입에 더하고 풀어서 묻기", "더하기", [
        why("설계 선택", "세 조합(요일×시간대×단계)을 칸마다 +1/-1로 더해 정수 프로토타입 하나에 담는다. "
                       "비교할 때는 다수결로 0/1 벡터로 만들고, 비긴 칸은 고정된 동점깨기 벡터를 따른다."),
        why("수학적 사실", "이진 프로토타입을 (수×14시대)로 풀면 수×14시대에 묶였던 단계 벡터는 비슷하게 남고, "
                       "나머지 두 조합은 무관한 잡음이 된다. 그래서 높은 확률로 가장 가까운 단계가 저장했던 단계다(확실한 것은 아니다)."),
    ], inputs=[{"type": "table", "name": "더한 조합", "rows": [[d, hr, lv] for (d, hr), lv in stored.items()]}],
       outputs=[{"type": "value", "name": "정수 프로토타입의 동점 칸 수", "value": m.tie_count()},
                vec_info("이진 프로토타입", mv, full=True),
                vec_info("이진 프로토타입 ⊕ (수×14시대)", q, full=True),
                {"type": "table", "name": "풀어 물은 결과와 단계별 유사도", "rows": ranking}],
       check=check("수요일 14시대에 저장한 단계", "9단계", ranking[0][0]),
       source=source_of(Memory))

    t.save(path)
    return t.data


if __name__ == "__main__":
    data = run()
    for s in data["steps"]:
        c = s["check"]
        tail = f" 정답={c['count_answer']} HDC={c['hdc_answer']} 일치={c['agree']}" if c else ""
        print(f"{s['index']}. {s['name']} [{s['op']}]{tail}")
