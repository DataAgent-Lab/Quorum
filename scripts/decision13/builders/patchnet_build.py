#!/usr/bin/env python3
"""Decision-model study (provenance/decision13_protocol.md), block E — PatchNet (Linux stable-patch identification) as noul decisions.

Input: the Zenodo record 3965234 `test.pkl` (cc-by-4.0; 10,896 mainline commits, label `stable`), loaded with an
unpickler that refuses every class, and a local bare clone of torvalds/linux (one bulk download instead of ~10k
requests to git.kernel.org). Every test commit is used (no sampling).

Per commit: subject + body + unified diff from `git show`. Label-leak lines are removed from the message (measured
on 400 rows: `Cc: …stable@…` 68/200 stable vs 0/200 non-stable; `Fixes:` 44/200 vs 3/200). The gold label is the
dataset's (merged into a stable tree OR merged at rc2 or later), so the question asks about suitability, not about
what actually happened.

    python -I patchnet_build.py --pkl test.pkl --repo linux.git --out items.jsonl.gz --manifest manifest.json
"""
import argparse, gzip, hashlib, json, pickle, re, subprocess
from concurrent.futures import ThreadPoolExecutor

# Whole lines that state or address the stable process (the label) are removed. Measured after a first, narrower
# pass: residual `stable@` 11 vs 0, `Fixes:` 6 vs 0, the word "stable" 126 vs 5, "backport" 50 vs 0 (stable vs
# non-stable). Reported-by:, Link: and CVE ids stay: they correlate with the label but are ordinary bug evidence
# a human reviewer also sees (reported in the protocol).
LEAK = [re.compile(r"^.*stable@.*$", re.I | re.M),
        re.compile(r"^\s*Fixes:.*$", re.I | re.M),
        re.compile(r"^.*\bstable\b.*$", re.I | re.M),
        re.compile(r"^.*\bbackport.*$", re.I | re.M)]
QUESTION = {
    "type": "noul",
    "instructions": "Is this Linux kernel patch a bug fix that is suitable for the stable kernel releases?",
    "criteria": {"true": "Yes: it fixes a real bug (crash, data corruption, security issue, hang, wrong behaviour) "
                         "in a way small and safe enough for stable releases.",
                 "false": "No: it is a feature, cleanup, refactor, documentation or other change that does not "
                          "belong in stable releases."},
}


class Refuse(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refused {module}.{name}")


def commit_text(repo, sha):
    msg = subprocess.run(["git", "-C", repo, "log", "-1", "--format=%s%n%n%b", sha],
                         capture_output=True, text=True, errors="replace", check=True).stdout.strip()
    diff = subprocess.run(["git", "-C", repo, "show", "--format=", "--no-color", "--patch", sha],
                          capture_output=True, text=True, errors="replace", check=True).stdout.strip()
    return msg, diff


def strip_leaks(msg):
    for rx in LEAK:
        msg = rx.sub("", msg)
    return re.sub(r"\n{3,}", "\n\n", msg).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True); ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--manifest", required=True)
    a = ap.parse_args()
    rows = Refuse(open(a.pkl, "rb")).load()
    rows = rows if isinstance(rows, list) else list(rows.values())

    def build(i_r):
        i, r = i_r
        msg, diff = commit_text(a.repo, r["id"])
        clean = strip_leaks(msg)
        body = {"model": "decision", "state": {"commit_message": clean, "diff": diff}, "questions": {"stable": QUESTION}}
        raw = json.dumps(body).encode()
        label = {"true": True, "false": False}[r["stable"]]          # the pickle stores the strings 'true' / 'false'
        return {"idx": i, "commit": r["id"], "gold": label, "body": body,
                "body_sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                "removed_lines": sum(len(rx.findall(msg)) for rx in LEAK)}

    with ThreadPoolExecutor(8) as ex:
        items = list(ex.map(build, enumerate(rows)))
    with gzip.open(a.out, "wt") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    still = sum(bool(LEAK[0].search(it["body"]["state"]["commit_message"])) for it in items)
    man = {"source": "zenodo:3965234 test.pkl", "pkl_sha256": hashlib.sha256(open(a.pkl, "rb").read()).hexdigest(),
           "items": len(items), "stable": sum(it["gold"] for it in items),
           "items_with_removed_lines": sum(it["removed_lines"] > 0 for it in items),
           "cc_stable_left_after_strip": still,
           "bytes_p50": sorted(it["bytes"] for it in items)[len(items) // 2],
           "bytes_max": max(it["bytes"] for it in items), "question": QUESTION,
           "items_sha256": hashlib.sha256("".join(it["body_sha256"] for it in items).encode()).hexdigest()}
    json.dump(man, open(a.manifest, "w"), indent=1)
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
