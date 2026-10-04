"""Phase 1.0b browser E2E — the real page against a real (local) Quorum API.

Setup (see the conversation file): a local API on :8003 (serve.app, CORS *, cache seeded from results/predictions,
models starting cold) and space/ served on :8004. Then:

    python docs/phases/1.0b/e2e/demo_e2e.py

Every acceptance criterion prints PASS/FAIL; screenshots go to docs/phases/1.0b/e2e-screenshots/.
"""
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

PAGE = "http://127.0.0.1:8004/index.html?api=http://127.0.0.1:8003"
SHOTS = Path(__file__).resolve().parents[1] / "e2e-screenshots"
SHOTS.mkdir(exist_ok=True)
DOMAINS = {"banking77": 77, "clinc150": 151, "hwu64": 64, "massive": 60, "mtop": 113, "snips": 7, "bitext": 27}
results, errors = [], []
expect_failed_requests = {"on": False}  # AC-5 injects 503s on purpose; Chrome logs those as console errors


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


def verdict(pg, timeout=30_000):
    pg.wait_for_selector(".ruling .name", timeout=timeout)
    pg.wait_for_timeout(2200)  # let the reveal choreography finish before asserting/screenshotting
    return pg.inner_text(".footnote")


def new_page(b, w=1440, h=1000, scheme="light", motion="no-preference"):
    ctx = b.new_context(viewport={"width": w, "height": h}, color_scheme=scheme, reduced_motion=motion,
                        device_scale_factor=1)
    pg = ctx.new_page()
    pg.on("console", lambda m: m.type == "error" and not (expect_failed_requests["on"] and "Failed to load resource" in m.text)
          and errors.append(m.text))
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(PAGE, wait_until="networkidle")
    pg.wait_for_selector(".chip[aria-pressed=true]")
    return ctx, pg


with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome")
    ctx, pg = new_page(b)

    # AC-1 default: Banking + a real test sentence -> real cached verdict
    pressed = pg.get_attribute(".chip[aria-pressed=true]", "data-key")
    pg.click("#go")
    foot = verdict(pg)
    truth = pg.get_attribute(".truth", "class")
    check("AC-1 default Banking sentence answered from the cache",
          pressed == "banking77" and "answered from the cache" in foot and ("ok" in truth or "no" in truth),
          f"[{pressed}] {foot.splitlines()[0]} · truth={truth}")
    pg.screenshot(path=str(SHOTS / "ac1-banking-verdict.png"), full_page=True)

    # AC-7 accessibility basics
    live = pg.get_attribute("#verdict", "aria-live")
    pg.focus("[data-key=snips]"); pg.keyboard.press("Enter")
    check("AC-7 aria-live verdict + chips work from the keyboard",
          live == "polite" and pg.get_attribute("[data-key=snips]", "aria-pressed") == "true")

    # AC-2 every domain: chip, 🎲 another sentence, verdict from the cache with K labels
    for key, k in DOMAINS.items():
        pg.click(f"[data-key={key}]")
        first = pg.input_value("#msg")
        pg.click("#another")
        second = pg.input_value("#msg")
        pg.click("#go")
        foot = verdict(pg)
        more = pg.inner_text("#more") if pg.query_selector("#more") else ""
        n_rows = pg.locator(".bars .bar-row").count()
        ok = "answered from the cache" in foot and first != second and n_rows == k
        check(f"AC-2 {key}: new real sentence, cache hit, {k} labels ranked", ok,
              f"rows={n_rows} more='{more}' {foot.splitlines()[0]}")
        pg.screenshot(path=str(SHOTS / f"ac2-{key}.png"))

    # AC-4 cold start: own labels, a new sentence, models unloaded on the local API
    pg.click("[data-key=own]")
    pg.fill("#msg", "My package arrived damaged and I would like a replacement sent out")
    t0 = time.time()
    pg.click("#go")
    try:
        pg.wait_for_selector(".waking", timeout=10_000)
        pg.wait_for_timeout(4000)
        clock = pg.inner_text("#clock")
        pg.screenshot(path=str(SHOTS / "ac4-waking.png"), full_page=True)
        foot = verdict(pg, timeout=200_000)
        check("AC-4 cold start shows the waking message + clock, then a live verdict (no client timeout)",
              "computed live" in foot, f"clock showed '{clock}', verdict after {time.time() - t0:.0f} s")
    except Exception as e:  # noqa: BLE001
        check("AC-4 cold start", False, repr(e)[:200])
    pg.screenshot(path=str(SHOTS / "ac4-cold-verdict.png"), full_page=True)

    # AC-3 edits: >50 labels blocked with an explanation; <=50 computed live
    pg.click("[data-key=clinc150]")
    pg.fill("#msg", "how do i say good morning in japanese, thanks")
    blocked = pg.is_disabled("#go") and "50" in pg.inner_text("#labels-hint")
    pg.screenshot(path=str(SHOTS / "ac3-edited-blocked.png"), full_page=True)
    pg.click("[data-key=snips]")
    pg.fill("#msg", "play some relaxing jazz in the kitchen please")
    enabled = not pg.is_disabled("#go")
    pg.click("#go")
    foot = verdict(pg, timeout=120_000)
    check("AC-3 edited: 151 labels blocked with a reason; SNIPS (7) computed live without a dataset answer",
          blocked and enabled and "computed live" in foot and "na" in pg.get_attribute(".truth", "class"))

    # AC-5 recovery: first /predict fails with 503 -> one automatic retry; persistent failure -> retry button
    calls = {"n": 0}

    def flaky(route):
        calls["n"] += 1
        if calls["n"] == 1:
            route.fulfill(status=503, content_type="application/json", body='{"detail":"busy"}')
        else:
            route.continue_()
    expect_failed_requests["on"] = True
    pg.route("**/predict", flaky)
    pg.click("[data-key=banking77]")
    pg.click("#go")
    foot = verdict(pg, timeout=30_000)
    check("AC-5a a failed first attempt is retried automatically", calls["n"] == 2, f"calls={calls['n']}")
    pg.unroute("**/predict")
    pg.route("**/predict", lambda r: r.fulfill(status=503, content_type="application/json", body='{"detail":"busy"}'))
    pg.click("#go")
    pg.wait_for_selector("#retry", timeout=30_000)
    pg.screenshot(path=str(SHOTS / "ac5-error-retry.png"), full_page=True)
    check("AC-5b persistent failure shows a calm error + Try again", pg.is_visible("#retry"))
    pg.unroute("**/predict")
    pg.click("#retry")
    verdict(pg)
    expect_failed_requests["on"] = False
    check("AC-5c Try again recovers", True)
    ctx.close()

    # AC-6 visuals: dark desktop, mobile; reduced motion shows the final state immediately
    for name, kw in [("ac6-desktop-dark", dict(scheme="dark")), ("ac6-mobile", dict(w=390, h=844)),
                     ("ac6-mobile-dark", dict(w=390, h=844, scheme="dark"))]:
        c2, p2 = new_page(b, **kw)
        p2.click("#go"); verdict(p2)
        overflow = p2.evaluate("document.documentElement.scrollWidth > document.documentElement.clientWidth")
        p2.screenshot(path=str(SHOTS / f"{name}.png"), full_page=True)
        check(f"AC-6 {name}: renders without horizontal overflow", not overflow)
        c2.close()
    c3, p3 = new_page(b, motion="reduce")
    p3.click("#go"); p3.wait_for_selector(".ruling .name"); p3.wait_for_timeout(150)
    conf = p3.inner_text("#conf")
    check("AC-6 reduced motion: final confidence shown immediately", conf not in ("", "0.0%"), conf)
    c3.close()
    b.close()

check("AC-7 no console errors", not errors, "; ".join(errors[:3]))
failed = [n for n, ok in results if not ok]
print(f"\n{len(results) - len(failed)}/{len(results)} passed" + (f" — FAILED: {failed}" if failed else ""))
sys.exit(1 if failed else 0)
