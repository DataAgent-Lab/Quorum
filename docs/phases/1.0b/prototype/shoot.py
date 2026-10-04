"""Screenshot the 1.0b prototype in real Chromium (desktop light, desktop dark, mobile) and report console errors."""
from pathlib import Path
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
URL = (HERE / "prototype.html").as_uri()
OUT = HERE / "shots"
errors = []

with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome")
    for name, vp, scheme in [("desktop-light", (1440, 1000), "light"), ("desktop-dark", (1440, 1000), "dark"),
                             ("mobile-light", (390, 844), "light")]:
        ctx = b.new_context(viewport={"width": vp[0], "height": vp[1]}, color_scheme=scheme, device_scale_factor=2)
        pg = ctx.new_page()
        pg.on("console", lambda m: m.type == "error" and errors.append(f"{name}: {m.text}"))
        pg.on("pageerror", lambda e: errors.append(f"{name}: {e}"))
        pg.goto(URL, wait_until="networkidle")
        pg.screenshot(path=str(OUT / f"{name}-1-start.png"), full_page=True)
        pg.click("#go")
        pg.wait_for_timeout(3200)  # deliberation + reveal choreography
        pg.screenshot(path=str(OUT / f"{name}-2-verdict.png"), full_page=True)
        if name == "desktop-light":
            pg.click("text=Show all")
            pg.wait_for_timeout(600)
            pg.click("[data-key=clinc150]")
            pg.fill("#msg", "how do i say hello in french, please")
            pg.wait_for_timeout(300)
            pg.screenshot(path=str(OUT / f"{name}-3-edited-hint.png"), full_page=True)
            assert pg.is_disabled("#go"), "edited 151-label sentence must disable Decide"
            pg.click("[data-key=snips]")
            pg.click("#go")
            pg.wait_for_timeout(3200)
            pg.screenshot(path=str(OUT / f"{name}-4-snips.png"), full_page=True)
            pg.click("[data-key=own]")
            pg.click("#go")
            pg.wait_for_timeout(3200)
            pg.screenshot(path=str(OUT / f"{name}-5-own.png"), full_page=True)
        ctx.close()
    b.close()
print("console/page errors:", errors or "none")
print("\n".join(sorted(f.name for f in OUT.glob("*.png"))))
