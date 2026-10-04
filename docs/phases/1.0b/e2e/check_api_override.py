"""The ?api= override must only work when the page itself is served locally (testing / self-hosting); on any other
host a crafted link must not redirect what visitors type to another server.

    (cd space && python -m http.server 8004 --bind 0.0.0.0) &
    python docs/phases/1.0b/e2e/check_api_override.py

Result 2026-10-04: PASS local page honours ?api= -> 127.0.0.1:8099; PASS non-local page ignores ?api= ->
quorum-api.idataagent.com.
"""
import socket
import sys

from playwright.sync_api import sync_playwright

ip = socket.gethostbyname(socket.gethostname())
cases = [("local page honours ?api=", "http://127.0.0.1:8004/index.html?api=http://127.0.0.1:8099", "127.0.0.1:8099"),
         ("non-local page ignores ?api=", f"http://{ip}:8004/index.html?api=http://127.0.0.1:8099",
          "quorum-api.idataagent.com")]
ok_all = True
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome")
    for name, url, want in cases:
        pg, seen = b.new_page(), []
        pg.on("request", lambda r: ("/predict" in r.url or "/health" in r.url) and seen.append(r.url))
        pg.goto(url, wait_until="networkidle")
        pg.wait_for_selector(".chip[aria-pressed=true]")
        pg.click("#go")
        pg.wait_for_timeout(1500)
        hosts = sorted({u.split("/")[2] for u in seen})
        ok = hosts == [want]
        ok_all &= ok
        print("PASS" if ok else "FAIL", name, "->", hosts)
        pg.close()
    b.close()
sys.exit(0 if ok_all else 1)
