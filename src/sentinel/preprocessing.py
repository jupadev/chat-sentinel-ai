"""Tweet preprocessing for the RoBERTuito models.

A trimmed port of pysentimiento.preprocessing.preprocess_tweet (lang="es", default arguments).
The models were fine-tuned on text processed this way, so inputs must match it. Ported instead of
imported because pysentimiento pulls torch, which the runtime image does not ship.
"""

import re

import emoji

_USER = re.compile(r"@[a-zA-Z0-9_]{0,15}")
_URL = re.compile(
    r"((?<=[^a-zA-Z0-9])(?:https?\:\/\/|[a-zA-Z0-9]{1,}\.{1}|\b)(?:\w{1,}\.{1}){1,5}"
    r"(?:com|co|org|edu|gov|uk|net|ca|de|jp|fr|au|us|ru|ch|it|nl|se|no|es|mil|iq|io|ac|ly|sm){1}"
    r"(?:\/[a-zA-Z0-9]{1,})*)"
)
_HASHTAG = re.compile(r"\B#(\w*[a-zA-Z]+\w*)")
_CAMEL = re.compile(r"([A-Z]+)")
_EMOJI = re.compile(r"\|([^\|]+)\|")
_REPEATED = re.compile(r"(.)\1\1+")
_LAUGHTER = re.compile("[ja][ja]+aj[ja]+")

_REPLACEMENTS: dict[str, str | None] = {
    "~": None, "£": None, "¥": None, "¨": '"', "©": None, "«": '"', "°": None, "´": None, "¶": None,
    "·": None, "º": None, "»": '"', "×": None, "ا": None, "–": None, "—": None, "‘": None, "’": None,
    "“": '"', "”": '"', "•": None, "…": None, "‼": "!", "€": "$", "™": None, "●": None, "☠": None,
    "☹": None, "☺": None, "☻": "😃", "♀": None, "♂": None, "♡": "❤️", "♥": "❤️", "⚰": None, "⛱": None,
    "⛹": None, "✈": None, "✓": None, "❤": None, "ー": None, "🕯": None, "🛰": None,
}  # fmt: skip


def preprocess_tweet_es(text: str) -> str:
    text = _HASHTAG.sub(lambda m: "hashtag " + _CAMEL.sub(r" \1", m.group(1)).strip().lower(), text)
    text = "".join(r for c in text if (r := _REPLACEMENTS.get(c, c)))
    text = _USER.sub("@usuario", text)
    text = _URL.sub("url", text)
    text = _REPEATED.sub(r"\1\1\1", text)
    text = emoji.demojize(text, language="es", delimiters=("|", "|"))
    text = _EMOJI.sub(lambda m: " emoji " + " ".join(m.group(1).split("_")) + " emoji ", text)
    text = _LAUGHTER.sub("jaja", text)
    return text.strip()
