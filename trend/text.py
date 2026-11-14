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


import os

_STOP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stopwords")


def _load_stopwords(name: str) -> set:
    """trend/stopwords/<name>.txt: 한 줄에 여럿, 공백으로 구분, '#' 뒤는 설명."""
    words = set()
    with open(os.path.join(_STOP_DIR, name), encoding="utf-8") as f:
        for line in f:
            words.update(line.split("#", 1)[0].split())
    return words


# 영어: 형태소 분석기가 품사를 가리지 못해(모두 '외국어'), 명사를 고르는 대신 이 목록을 뺀다.
# 그래서 영어 쪽 '같이 나온 낱말'에는 목록에 없는 명사 아닌 낱말이 섞일 수 있다.
EN_STOP = {w.lower() for w in _load_stopwords("en.txt")}
# 한국어: 명사이지만 내용이 없는 기사 형식어, 때를 나타내는 말, 연재 꼬리표
KO_STOP = _load_stopwords("ko.txt")


def is_ascii(text: str) -> bool:
    return all(ord(c) < 128 for c in text)


def variant_pattern(variant: str) -> re.Pattern:
    """표기 하나 → 정규식. 영문은 대소문자 무시, 공백은 있거나 없거나.
    영어 표기는 다른 낱말 속에 들어간 것은 세지 않는다(robotaxi는 robot이 아님). 끝의 s, es(복수)는 허용."""
    parts = [re.escape(p) for p in variant.split()]
    core = r"\s*".join(parts)
    if is_ascii(variant):
        return re.compile(r"(?<![A-Za-z])" + core + r"(?:s|es)?(?![A-Za-z])", re.IGNORECASE)
    return re.compile(core, re.IGNORECASE)


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
        if is_ascii(form):
            form = form.lower()  # 영어는 소문자로 맞춘다(AI와 ai를 한 낱말로)
            if form in EN_STOP or len(form) < 2 or not any(c.isalpha() for c in form):
                continue
        elif len(form) < 2 or form in KO_STOP:
            continue
        if form not in out:
            out.append(form)
    return out
