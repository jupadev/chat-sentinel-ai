import re
import unicodedata
from datetime import timedelta
from urllib.parse import urlsplit

from sentinel.config import RuleConfig
from sentinel.models import ChatMessage, History, RuleHit

_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
_LONG_RUNS = re.compile(r"(\w)\1{2,}")


def normalize(text: str, leet: bool = False) -> str:
    """Lowercase, strip accents and collapse runs of 3+ equal letters."""
    text = text.lower()
    if leet:
        text = text.translate(_LEET)
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return _LONG_RUNS.sub(r"\1", text)


_PHONE = re.compile(r"(?<!\w)\+?\d(?:[\s\-.()]{0,2}\d){7,14}(?!\w)")
_NOT_PHONE = [
    re.compile(r"^\d{1,3}(?:\.\d{3})+$"),  # 15.000.000
    re.compile(r"^\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}$"),  # 28.10.2026
    re.compile(r"^(?:19|20)\d{2}\s*-\s*(?:19|20)\d{2}$"),  # 2026-2027
]

_URL = re.compile(r"(?:https?://|www\.)[^\s<>\"']+", re.IGNORECASE)
_BARE_DOMAIN = re.compile(
    r"(?<![@\w.])(?:[a-z0-9-]+\.)+(?:com|net|org|io|ly|me|app|link|co|gg|xyz|info|biz|ar)(?:\.ar)?\b(?:/[^\s]*)?",
    re.IGNORECASE,
)


def _wordlist_regex(words: list[str]) -> re.Pattern[str]:
    alternatives = sorted({r"\s+".join(map(re.escape, normalize(w).split())) for w in words}, key=len, reverse=True)
    return re.compile(r"(?<!\w)(?:" + "|".join(alternatives) + r")(?!\w)")


def _host(url: str) -> str:
    if not re.match(r"^[a-z]+://", url, re.IGNORECASE):
        url = "http://" + url
    host = (urlsplit(url).hostname or "").lower()
    return host.removeprefix("www.")


class RuleEngine:
    def __init__(self, rules: list[RuleConfig]):
        self._rules = rules
        self._compiled: dict[str, list[re.Pattern[str]]] = {}
        for rule in rules:
            if rule.type == "regex":
                flags = 0 if rule.normalized else re.IGNORECASE
                self._compiled[rule.name] = [re.compile(p, flags) for p in rule.patterns]
            elif rule.type == "wordlist":
                self._compiled[rule.name] = [_wordlist_regex(rule.words)]

    def evaluate(self, message: ChatMessage, history: History | None = None) -> list[RuleHit]:
        text = message.text
        normalized = normalize(text)
        hits: list[RuleHit] = []
        for rule in self._rules:
            match = self._match(rule, message, text, normalized, history)
            if match is not None:
                hits.append(RuleHit(rule.name, rule.action, rule.category, match[:120]))
        return hits

    def _match(
        self, rule: RuleConfig, message: ChatMessage, text: str, normalized: str, history: History | None
    ) -> str | None:
        if rule.type == "regex":
            target = normalized if rule.normalized else text
            return _first_match(self._compiled[rule.name], target)
        if rule.type == "wordlist":
            return _first_match(self._compiled[rule.name], normalize(text, leet=True)) or _first_match(
                self._compiled[rule.name], normalized
            )
        if rule.type == "phone":
            return _find_phone(text, rule.min_digits)
        if rule.type == "link":
            return _find_external_link(text, rule.allowed_domains)
        if rule.type == "repeated":
            return _find_repeat(rule, message, normalized, history)
        return None


def _first_match(patterns: list[re.Pattern[str]], text: str) -> str | None:
    for pattern in patterns:
        if m := pattern.search(text):
            return m.group(0)
    return None


def _find_phone(text: str, min_digits: int) -> str | None:
    for m in _PHONE.finditer(text):
        candidate = m.group(0).strip()
        if sum(c.isdigit() for c in candidate) < min_digits:
            continue
        if any(p.match(candidate) for p in _NOT_PHONE):
            continue
        if "$" in text[max(0, m.start() - 2) : m.start()]:
            continue
        return candidate
    return None


def _find_external_link(text: str, allowed_domains: list[str]) -> str | None:
    for m in [*_URL.finditer(text), *_BARE_DOMAIN.finditer(text)]:
        host = _host(m.group(0))
        if not host:
            continue
        if any(host == d or host.endswith("." + d) for d in allowed_domains):
            continue
        return m.group(0)
    return None


def _find_repeat(rule: RuleConfig, message: ChatMessage, normalized: str, history: History | None) -> str | None:
    if history is None or len(normalized.strip()) < rule.min_length:
        return None
    since = message.created_at - timedelta(minutes=rule.window_minutes)
    previous = history.recent_texts_by_author(message, since)
    repeats = 1 + sum(normalize(t).strip() == normalized.strip() for t in previous)
    return normalized if repeats >= rule.min_repeats else None
