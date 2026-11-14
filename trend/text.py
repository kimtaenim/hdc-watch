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


# 영어 제목에서 뺄 흔한 낱말. 형태소 분석기는 영어 품사를 가리지 않아(모두 '외국어'), 명사를 고르는 대신 이 목록을 뺀다.
# 그래서 영어 쪽 '같이 나온 낱말'에는 명사 아닌 낱말도 섞일 수 있다.
EN_STOP = set("""
a an the and or but nor for of in on at to from by with as is are was were be been being it its this that these those
into over under about after before up down out off than then now just also via vs amid among per new more most less
how why what when where who whom which will can could would should may might must not no yes all any some each every
our your their his her we you they he she i my me us them here there s says said say get gets got has have had having
do does did done make makes made take takes took see sees seen show shows set sets use uses using used plan plans
launch launches launched announce announces announced unveil unveils unveiled win wins won add adds added open opens
first top big latest inside week weeks year years day days today news report reports update updates one two three
""".split())


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
        elif len(form) < 2:
            continue
        if form not in out:
            out.append(form)
    return out
