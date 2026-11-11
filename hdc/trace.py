"""추적 기록(trace): 실행 한 번의 모든 단계를 순서대로 남기는 JSON.

화면은 이 파일만 읽어 톱니바퀴를 그린다. 설명 문구도 여기서 나온다.

형식 (format = "hdc-watch-trace/1")
{
  "format": "hdc-watch-trace/1",
  "kind": "hdc.selfcheck" 같은 실행 종류,
  "subject": {"type": "stock"|"keyword"|null, "id": ..., "name": ...},
  "started_at", "finished_at": ISO 시각(KST),
  "params": {"D": 10000, ...},
  "steps": [
    {
      "index": 1,
      "section": "시간대 모델" 같은 묶음 이름(sections 참고),
      "name": "단계 이름",
      "op": "벡터 생성" | "뒤집기" | "묶기" | "더하기" | "비교" | "일반 계산",
      "why": [{"kind": "수학적 사실"|"이 데이터에서 관찰"|"비유"|"설계 선택", "text": "..."}],
      "inputs":  [값 또는 벡터 요약, ...],
      "outputs": [값 또는 벡터 요약, ...],
      "check": {"question", "count_answer", "hdc_answer", "agree"} 또는 null,
      "source": {"file", "line", "function"} 또는 null
    }, ...
  ],
  "log": {"fetched": {...}, "failures": [...], "notes": [...]}
}

벡터 요약(vec_info)
{"type": "vector", "name", "dim", "ones", "head": 앞 64비트 문자열, "bits_b64": (선택) 전체 비트}
"""

import base64
import datetime as dt
import inspect
import json
import os

import numpy as np

FORMAT = "hdc-watch-trace/1"
OPS = ("벡터 생성", "뒤집기", "묶기", "더하기", "비교", "일반 계산")
WHY_KINDS = ("수학적 사실", "이 데이터에서 관찰", "비유", "설계 선택")
KST = dt.timezone(dt.timedelta(hours=9))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def now_kst() -> str:
    return dt.datetime.now(KST).isoformat(timespec="seconds")


def why(kind: str, text: str) -> dict:
    assert kind in WHY_KINDS, kind
    return {"kind": kind, "text": text}


def vec_info(name: str, v: np.ndarray, full: bool = False, head: int = 64) -> dict:
    """벡터를 화면에 보일 수 있게 요약한다. full=True면 100×100 격자를 그릴 전체 비트를 담는다."""
    info = {
        "type": "vector",
        "name": name,
        "dim": int(len(v)),
        "ones": int(np.count_nonzero(v)),
        "head": "".join(str(int(b)) for b in v[:head]),
    }
    if full:
        info["bits_b64"] = base64.b64encode(np.packbits(v).tobytes()).decode("ascii")
    return info


def bits_from_b64(text: str, dim: int) -> np.ndarray:
    return np.unpackbits(np.frombuffer(base64.b64decode(text), dtype=np.uint8))[:dim]


def source_of(func) -> dict:
    """함수가 정의된 파일과 줄 번호. 화면에서 소스 코드 줄로 가는 링크가 된다."""
    func = inspect.unwrap(func)
    path = inspect.getsourcefile(func)
    line = inspect.getsourcelines(func)[1]
    return {
        "file": os.path.relpath(path, REPO_ROOT).replace(os.sep, "/"),
        "line": line,
        "function": func.__qualname__,
    }


class Trace:
    def __init__(self, kind: str, subject: dict | None = None, params: dict | None = None):
        self.data = {
            "format": FORMAT,
            "kind": kind,
            "subject": subject,
            "started_at": now_kst(),
            "finished_at": None,
            "params": params or {},
            "sections": [],
            "steps": [],
            "log": {"fetched": {}, "failures": [], "notes": []},
        }

    def section(self, title: str, text: str = "") -> None:
        """여기서부터의 단계가 속한 묶음(예: '시간대 모델', '매물대 모델')."""
        self.data["sections"].append({"title": title, "text": text, "from_index": len(self.data["steps"]) + 1})
        self._section = title

    def step(self, name, op, why_list, inputs=None, outputs=None, check=None, source=None) -> dict:
        assert op in OPS, op
        if callable(source):
            source = source_of(source)
        s = {
            "index": len(self.data["steps"]) + 1,
            "section": getattr(self, "_section", None),
            "name": name,
            "op": op,
            "why": why_list,
            "inputs": inputs or [],
            "outputs": outputs or [],
            "check": check,
            "source": source,
        }
        self.data["steps"].append(s)
        return s

    def failure(self, what: str, detail: str) -> None:
        self.data["log"]["failures"].append({"at": now_kst(), "what": what, "detail": detail})

    def note(self, text: str) -> None:
        self.data["log"]["notes"].append(text)

    def fetched(self, key: str, count) -> None:
        self.data["log"]["fetched"][key] = count

    def save(self, path: str) -> None:
        self.data["finished_at"] = now_kst()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=1, default=_jsonable)


def check(question: str, count_answer, hdc_answer) -> dict:
    """같은 질문을 단순 카운트로 푼 정답과 HDC 답을 나란히 둔다."""
    return {"question": question, "count_answer": count_answer, "hdc_answer": hdc_answer,
            "agree": count_answer == hdc_answer}


def _jsonable(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))
