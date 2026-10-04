"""Verify the PUBLISHED Space end to end: real Chrome, the live static host, the production API (no ?api=).

    python docs/phases/1.0b/e2e/verify_live.py

Only cached requests are sent (domain samples + the seeded own-labels default), so the check never wakes the
production model.
"""
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

LIVE = "https://dataagent-quorum-demo.static.hf.space/index.html"
SHOTS = Path(__file__).resolve().parents[1] / "e2e-screenshots"
results, errors = [], []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


def verdict(pg):
    pg.wait_for_selector(".ruling .name", timeout=30_000)
    pg.wait_for_timeout(2200)
    return pg.inner_text(".footnote")


with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome")
    for name, vp in [("live-desktop", (1440, 1000)), ("live-mobile", (390, 844))]:
        ctx = b.new_context(viewport={"width": vp[0], "height": vp[1]})
        pg = ctx.new_page()
        pg.on("console", lambda m: m.type == "error" and errors.append(m.text))
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(f"{LIVE}?v={int(time.time())}", wait_until="networkidle")  # bypass stale caches
        new_design = pg.query_selector(".court") is not None
        check(f"{name}: the redesigned page is live", new_design)
        if not new_design:
            continue
        pg.wait_for_selector(".chip[aria-pressed=true]", timeout=15_000)
        pg.click("#go")
        foot = verdict(pg)
        check(f"{name}: default Banking sentence answered from the cache", "answered from the cache" in foot,
              foot.splitlines()[0])
        pg.screenshot(path=str(SHOTS / f"{name}-verdict.png"), full_page=True)
        if name == "live-desktop":
            for key in ("clinc150", "snips", "bitext"):
                pg.click(f"[data-key={key}]"); pg.click("#go")
                foot = verdict(pg)
                check(f"{name}: {key} answered from the cache", "answered from the cache" in foot,
                      foot.splitlines()[0])
            pg.click("[data-key=own]"); pg.click("#go")
            foot = verdict(pg)
            check(f"{name}: own-labels default answered from the cache", "answered from the cache" in foot,
                  foot.splitlines()[0])
        ctx.close()
    b.close()
check("no console errors", not errors, "; ".join(errors[:3]))
print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
