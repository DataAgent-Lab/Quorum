"""Example: per-class *descriptions* as label text (the "+quality" path).

Quorum's default matches an utterance against bare label *names*. When names are short or opaque, a one-line
natural-language *description* per class carries far more signal — and the ensemble already accepts it via the
`label_texts` argument, so this needs no extra code or model.

Measured on SNIPS (full test, 1400): names 0.8529 -> descriptions 0.9443 (+0.0914, McNemar p<1e-6), which is
above the description-based "dataless" reference (0.9257). See docs/phases/1.1/.

Run:  python examples/snips_descriptions.py
"""
from quorum import ZeroShotEnsemble

# 7 SNIPS intents. Keys = the label names; values = a short description of what each intent is about.
SNIPS_DESCRIPTIONS = {
    "add to playlist": "adding a song or artist to a music playlist",
    "book restaurant": "booking or reserving a table at a restaurant",
    "get weather": "the weather or the forecast for a place or time",
    "play music": "playing a song, album, or artist",
    "rate book": "giving a book a rating or review score",
    "search creative work": "finding a creative work such as a movie, book, song, or TV show",
    "search screening event": "finding movie showtimes or a cinema screening",
}

if __name__ == "__main__":
    labels = list(SNIPS_DESCRIPTIONS)
    descriptions = [SNIPS_DESCRIPTIONS[name] for name in labels]

    clf = ZeroShotEnsemble()

    text = "play the latest album by taylor swift"
    # `labels` are what gets returned; `label_texts` are what the model actually matches against.
    proba = clf.predict_proba(text, labels, label_texts=descriptions)
    ranked = sorted(zip(labels, proba), key=lambda kv: kv[1], reverse=True)
    print(f"text: {text!r}")
    for name, p in ranked[:3]:
        print(f"  {p:6.3f}  {name}")
