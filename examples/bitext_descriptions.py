"""Example: 27 generic per-class descriptions for Bitext customer-support — the "+quality" label-text path.

Plain one-line rephrasings of each intent name (anyone can write these; nothing from the dataset).
Pass as `label_texts`. See docs/phases/1.1/.

Run:  python scripts/eval_zeroshot.py --dataset bitext --descriptions
"""
from quorum import ZeroShotEnsemble

BITEXT_DESCRIPTIONS = {
    "cancel order": "cancelling an order",
    "change order": "changing an existing order",
    "change shipping address": "changing the shipping address",
    "check cancellation fee": "asking about the cancellation fee",
    "check invoice": "viewing or checking an invoice",
    "check payment methods": "asking which payment methods are accepted",
    "check refund policy": "asking about the refund policy",
    "complaint": "making a complaint",
    "contact customer service": "asking to contact customer service",
    "contact human agent": "asking to speak to a human agent",
    "create account": "creating a new account",
    "delete account": "deleting an account",
    "delivery options": "asking about delivery options",
    "delivery period": "asking how long delivery will take",
    "edit account": "editing account details",
    "get invoice": "requesting an invoice",
    "get refund": "requesting a refund",
    "newsletter subscription": "subscribing to or managing the newsletter",
    "payment issue": "reporting a payment problem",
    "place order": "placing an order",
    "recover password": "recovering a forgotten password",
    "registration problems": "having trouble registering an account",
    "review": "leaving a review",
    "set up shipping address": "setting up a shipping address",
    "switch account": "switching to a different account",
    "track order": "tracking an order",
    "track refund": "tracking a refund",
}

if __name__ == "__main__":
    labels = list(BITEXT_DESCRIPTIONS)
    clf = ZeroShotEnsemble()
    text = "I want my money back for this"
    proba = clf.predict_proba(text, labels, label_texts=[BITEXT_DESCRIPTIONS[l] for l in labels])
    for name, p in sorted(zip(labels, proba), key=lambda kv: -kv[1])[:3]:
        print(f"  {p:6.3f}  {name}")
