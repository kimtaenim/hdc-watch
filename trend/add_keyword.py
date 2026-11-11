"""키워드 추가(Actions 수동 버튼에서 부른다). 설정 파일 config/watch.json의 keywords 목록에 더한다.

실행: python -m trend.add_keyword --id cooling --name "액침냉각" --variants "액침냉각, 액침 냉각"
"""

import argparse
import json
import re

from volume.train import CONFIG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--variants", default="")
    a = ap.parse_args()
    kid = a.id.strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,40}", kid):
        raise SystemExit("id는 영문 소문자, 숫자, 하이픈만 (예: cooling, physical-ai)")
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    if a.name.strip() not in variants:
        variants.insert(0, a.name.strip())
    cfg = json.load(open(CONFIG, encoding="utf-8"))
    if any(k["id"] == kid for k in cfg["keywords"]):
        raise SystemExit(f"이미 있는 id: {kid}")
    cfg["keywords"].append({"id": kid, "name": a.name.strip(), "variants": variants})
    with open(CONFIG, "w", encoding="utf-8") as f:
        f.write(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
    print("추가:", kid, variants)


if __name__ == "__main__":
    main()
