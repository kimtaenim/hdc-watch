"""거래량 이어달리기 실행.

GitHub 예약 실행은 몇 시간씩 밀리거나 아예 빠진다(2026-11-17-19 기록). 그래서 예약에 기대지 않는다:
  ① 실행 하나가 최대 5시간 45분 동안 살아 있으면서, 평일 09:05-15:35 매시 5분·35분과 16:05에 volume.live를 돌리고 커밋한다.
  ② 시간이 다 차면 같은 워크플로를 다시 켜고(workflow_dispatch) 끝난다. 다음 실행이 이어 받는다.
  ③ 예약 실행은 이어달리기가 끊겼을 때 다시 잇는 예비다. 워크플로의 concurrency 때문에 동시에 둘이 돌지 않는다.
다시 켜기에는 GitHub가 실행마다 자동으로 주는 GITHUB_TOKEN을 쓴다(저장된 비밀값 아님).

실행: python -m volume.keeper   (GitHub Actions 안에서)
"""

import datetime as dt
import json
import os
import subprocess
import time
import urllib.request

KST = dt.timezone(dt.timedelta(hours=9))
LIFE = dt.timedelta(hours=5, minutes=45)  # 한 실행의 수명. GitHub 한도(6시간)보다 짧게.


def due_times(day: dt.date) -> list[dt.datetime]:
    """그날 volume.live를 돌릴 시각들. 평일만."""
    if day.weekday() >= 5:
        return []
    out = []
    for h in range(9, 16):
        for m in (5, 35):
            t = dt.datetime(day.year, day.month, day.day, h, m, tzinfo=KST)
            if t <= dt.datetime(day.year, day.month, day.day, 15, 35, tzinfo=KST):
                out.append(t)
    out.append(dt.datetime(day.year, day.month, day.day, 16, 5, tzinfo=KST))  # 마감 갱신 한 번 더(빠졌을 때 대비)
    return out


def next_due(now: dt.datetime) -> dt.datetime:
    for k in range(0, 8):
        for t in due_times((now + dt.timedelta(days=k)).date()):
            if t > now:
                return t
    raise RuntimeError("다음 실행 시각을 찾지 못함")


COMMIT = r'''
for i in 1 2 3 4; do
  git fetch -q origin main && git reset -q --hard origin/main
  python -m volume.live || true
  git add docs/data
  git diff --cached --quiet && exit 0
  git commit -q -m "거래량 채점 $(TZ=Asia/Seoul date +%Y-%m-%dT%H:%M)"
  git push -q origin HEAD:main && exit 0
  sleep $((i * 5))
done
exit 1
'''


def work():
    print(dt.datetime.now(KST).strftime("%m-%d %H:%M"), "채점", flush=True)
    subprocess.run(["bash", "-c", COMMIT])


def relay():
    """같은 워크플로를 다시 켠다."""
    repo, token = os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_TOKEN"]
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/actions/workflows/volume-live.yml/dispatches",
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
    end = start + LIFE
    # 켜지자마자: 장중이나 마감 직후면 한 번 돌린다(빠진 날 따라잡기 포함)
    if start.weekday() < 5 and dt.time(9, 5) <= start.time() <= dt.time(17, 0):
        work()
    while True:
        now = dt.datetime.now(KST)
        t = next_due(now)
        if t >= end:
            wait = (end - now).total_seconds()
            if wait > 0:
                time.sleep(wait)
            relay()
            return
        time.sleep(max(0, (t - now).total_seconds()))
        work()


if __name__ == "__main__":
    main()
