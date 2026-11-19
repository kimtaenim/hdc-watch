"""이어달리기 자동 실행: 거래량과 트렌드를 정해진 시각에 돌린다.

GitHub 예약 실행은 몇 시간씩 밀리거나 아예 빠진다(2026-11-17-19 기록). 그래서 예약에 기대지 않는다:
  ① 실행 하나가 최대 5시간 45분 동안 살아 있으면서, 정해진 시각마다 일을 하고 커밋한다.
     거래량: 평일 09:05-15:35 매시 5분·35분, 16:05(마감 갱신 한 번 더).
     트렌드: 매일 07:20, 12:20, 17:20, 23:20.
  ② 시간이 다 차면 같은 워크플로를 다시 켜고(workflow_dispatch) 끝난다. 다음 실행이 이어 받는다.
  ③ 예약 실행은 이어달리기가 끊겼을 때 다시 잇는 예비다. 워크플로의 concurrency 때문에 동시에 둘이 돌지 않는다.
  ④ 켜지자마자 밀린 일을 먼저 한다: 장중이면 거래량, 마지막 트렌드 수집이 6시간 넘게 지났으면 트렌드.
다시 켜기에는 GitHub가 실행마다 자동으로 주는 GITHUB_TOKEN을 쓴다(저장된 비밀값 아님).
한 실행 안에서 일을 하나씩 차례로 하므로 거래량과 트렌드가 같은 파일을 두고 부딪히지 않는다.

실행: python -m keeper   (GitHub Actions 안에서)
"""

import datetime as dt
import json
import os
import subprocess
import time
import urllib.request

KST = dt.timezone(dt.timedelta(hours=9))
LIFE = dt.timedelta(hours=5, minutes=45)  # 한 실행의 수명. GitHub 한도(6시간)보다 짧게.
WORKFLOW = "volume-live.yml"
TREND_AT = [(7, 20), (12, 20), (17, 20), (23, 20)]
TREND_STALE = dt.timedelta(hours=6)

JOBS = {
    "거래량": ("python -m volume.live", "거래량 채점"),
    "트렌드": ("python -m trend.analyze --collect", "트렌드 수집"),
}


def at(day: dt.date, h: int, m: int) -> dt.datetime:
    return dt.datetime(day.year, day.month, day.day, h, m, tzinfo=KST)


def due_times(day: dt.date) -> list[tuple[dt.datetime, str]]:
    """그날 할 일과 시각."""
    out = [(at(day, h, m), "트렌드") for h, m in TREND_AT]
    if day.weekday() < 5:
        out += [(at(day, h, m), "거래량") for h in range(9, 16) for m in (5, 35) if (h, m) <= (15, 35)]
        out.append((at(day, 16, 5), "거래량"))
    return sorted(out)


def next_due(now: dt.datetime) -> tuple[dt.datetime, str]:
    for k in range(0, 8):
        for t, job in due_times((now + dt.timedelta(days=k)).date()):
            if t > now:
                return t, job
    raise RuntimeError("다음 실행 시각을 찾지 못함")


COMMIT = r'''
for i in 1 2 3 4; do
  git fetch -q origin main && git reset -q --hard origin/main
  {cmd} || true
  git add docs/data
  git diff --cached --quiet && exit 0
  git commit -q -m "{msg} $(TZ=Asia/Seoul date +%Y-%m-%dT%H:%M)"
  git push -q origin HEAD:main && exit 0
  sleep $((i * 5))
done
exit 1
'''


def work(job: str):
    cmd, msg = JOBS[job]
    print(dt.datetime.now(KST).strftime("%m-%d %H:%M"), job, flush=True)
    subprocess.run(["bash", "-c", COMMIT.format(cmd=cmd, msg=msg)])


def last_trend() -> dt.datetime | None:
    try:
        runs = json.load(open("docs/data/runs.json", encoding="utf-8"))
        return max(dt.datetime.fromisoformat(r["at"]) for r in runs if r.get("kind") == "trend")
    except Exception:
        return None


def relay():
    """같은 워크플로를 다시 켠다."""
    repo, token = os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_TOKEN"]
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/actions/workflows/{WORKFLOW}/dispatches",
        data=json.dumps({"ref": "main"}).encode(), method="POST",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})
    for i in range(5):
        try:
            urllib.request.urlopen(req, timeout=30)
            print("다음 실행을 켰다", flush=True)
            return
        except Exception as e:
            print("다시 켜기 실패:", e, flush=True)
            time.sleep(20 * (i + 1))


def main():
    start = dt.datetime.now(KST)
    life = dt.timedelta(minutes=int(os.environ["LIFE_MIN"])) if os.environ.get("LIFE_MIN") else LIFE
    end = start + life
    print("수명", life, "끝", end.strftime("%m-%d %H:%M"), flush=True)
    # 켜지자마자 밀린 일
    lt = last_trend()
    if lt is None or start - lt > TREND_STALE:
        work("트렌드")
    if start.weekday() < 5 and dt.time(9, 5) <= start.time() <= dt.time(17, 0):
        work("거래량")
    cursor = dt.datetime.now(KST)  # 일하다 지나간 시각도 건너뛰지 않도록, 지금이 아니라 앞선 예정 시각에서 다음을 찾는다
    while True:
        now = dt.datetime.now(KST)
        t, job = next_due(cursor)
        cursor = t
        if t >= end:
            wait = (end - now).total_seconds()
            if wait > 0:
                time.sleep(wait)
            relay()
            return
        time.sleep(max(0, (t - now).total_seconds()))
        work(job)


if __name__ == "__main__":
    main()
