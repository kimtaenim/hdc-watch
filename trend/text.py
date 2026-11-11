"""제목에서 키워드 찾기와 명사 뽑기. HDC 연산은 없다.

키워드는 형태소 분석 전에 제목 원문에서 찾는다(분석기가 '로봇개'를 '로봇'+'개'로 쪼개기 때문).
명사는 kiwipiepy로 뽑되, 띄어쓰기 없이 붙어 있던 명사 조각은 다시 하나로 합친다('데이터'+'센터' → '데이터센터').
"""

import re

_kiwi = None


def kiwi():
    global _kiwi
    if _kiwi is None:
        from kiwipiepy import Kiwi
        _kiwi = Kiwi()
    return _kiwi


def variant_pattern(variant: str) -> re.Pattern:
    """표기 하나 → 정규식. 영문은 대소문자 무시, 공백은 있거나 없거나."""
    parts = [re.escape(p) for p in variant.split()]
    return re.compile(r"\s*".join(parts), re.IGNORECASE)


def find_keywords(title: str, keywords) -> dict:
    """제목에 나온 키워드 → 그 표기가 차지한 글자 구간들."""
    found = {}
    for kw in keywords:
        spans = []
        for v in kw["variants"]:
            spans += [m.span() for m in variant_pattern(v).finditer(title)]
        if spans:
            found[kw["id"]] = spans
    return found


def nouns(title: str, skip_spans=()) -> list[str]:
    """제목의 명사(일반·고유명사, 외국어). 붙어 있던 조각은 합치고, skip_spans(그 키워드의 표기 구간)와 겹치는 것은 뺀다.
    다른 키워드의 표기는 빼지 않는다: '피지컬 AI' 옆의 '휴머노이드'는 중요한 옆 단어다.
    한 글자 한글 명사는 뺀다. 같은 제목 안에서 중복은 한 번만."""
    toks = [t for t in kiwi().tokenize(title) if t.tag in ("NNG", "NNP", "SL")]
    merged = []
    for t in toks:
        if merged and merged[-1][2] == t.start:  # 바로 붙어 있으면 합친다
            form, s, _ = merged[-1]
            merged[-1] = (form + t.form, s, t.start + t.len)
        else:
            merged.append((t.form, t.start, t.start + t.len))
    out = []
    for form, s, e in merged:
        if any(s < b and a < e for a, b in skip_spans):
            continue
        if len(form) < 2:
            continue
        if form not in out:
            out.append(form)
    return out
