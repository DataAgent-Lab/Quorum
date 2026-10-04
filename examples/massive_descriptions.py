"""Example: 60 generic per-class descriptions for MASSIVE (en) — the "+quality" label-text path.

Plain one-line rephrasings of each intent name (anyone can write these; nothing from the dataset).
Pass as `label_texts`. See docs/phases/1.1/.

Run:  python scripts/eval_zeroshot.py --dataset massive --descriptions
"""
from quorum import ZeroShotEnsemble

MASSIVE_DESCRIPTIONS = {
    "alarm query": "asking about an alarm that is set",
    "alarm remove": "removing or cancelling an alarm",
    "alarm set": "setting an alarm",
    "audio volume down": "turning the volume down",
    "audio volume mute": "muting the audio",
    "audio volume other": "some other change to the audio volume",
    "audio volume up": "turning the volume up",
    "calendar query": "asking what is on the calendar",
    "calendar remove": "removing a calendar event",
    "calendar set": "adding a calendar event",
    "cooking query": "asking a cooking question",
    "cooking recipe": "getting a cooking recipe",
    "datetime convert": "converting a time or date between zones or formats",
    "datetime query": "asking for the current date or time",
    "email addcontact": "adding a contact from an email",
    "email query": "asking about emails in the inbox",
    "email querycontact": "looking up an email contact",
    "email sendemail": "sending an email",
    "general greet": "a greeting such as hello",
    "general joke": "asking for a joke",
    "general quirky": "an odd or off-topic remark",
    "iot cleaning": "starting a cleaning device such as a robot vacuum",
    "iot coffee": "making coffee with a smart coffee machine",
    "iot hue lightchange": "changing the colour of the smart lights",
    "iot hue lightdim": "dimming the smart lights",
    "iot hue lightoff": "turning the smart lights off",
    "iot hue lighton": "turning the smart lights on",
    "iot hue lightup": "making the smart lights brighter",
    "iot wemo off": "turning a smart plug or switch off",
    "iot wemo on": "turning a smart plug or switch on",
    "lists createoradd": "creating a list or adding an item to it",
    "lists query": "asking what is on a list",
    "lists remove": "removing an item from a list",
    "music dislikeness": "saying you dislike the current music",
    "music likeness": "saying you like the current music",
    "music query": "asking what music is playing",
    "music settings": "changing music settings",
    "news query": "asking for the news",
    "play audiobook": "playing an audiobook",
    "play game": "playing a game",
    "play music": "playing music",
    "play podcasts": "playing a podcast",
    "play radio": "playing the radio",
    "qa currency": "asking about currency or an exchange rate",
    "qa definition": "asking for the definition of a word",
    "qa factoid": "asking a general-knowledge question",
    "qa maths": "asking for a maths calculation",
    "qa stock": "asking about a stock price",
    "recommendation events": "asking for event recommendations",
    "recommendation locations": "asking for place or location recommendations",
    "recommendation movies": "asking for movie recommendations",
    "social post": "posting to social media",
    "social query": "asking about social media",
    "takeaway order": "ordering takeaway food",
    "takeaway query": "asking about a takeaway order",
    "transport query": "asking about transport or directions",
    "transport taxi": "booking a taxi",
    "transport ticket": "buying a transport ticket",
    "transport traffic": "asking about traffic conditions",
    "weather query": "asking about the weather",
}

if __name__ == "__main__":
    labels = list(MASSIVE_DESCRIPTIONS)
    clf = ZeroShotEnsemble()
    text = "what's the forecast for tomorrow"
    proba = clf.predict_proba(text, labels, label_texts=[MASSIVE_DESCRIPTIONS[l] for l in labels])
    for name, p in sorted(zip(labels, proba), key=lambda kv: -kv[1])[:3]:
        print(f"  {p:6.3f}  {name}")
