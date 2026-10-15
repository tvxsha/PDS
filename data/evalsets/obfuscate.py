"""
Text obfuscations for the robustness set (roadmap task 2.5).

Each function takes the PAYLOAD string (the injected sentence, never the host paragraph) and returns a
transformed string. All functions are pure and deterministic: the same input (and the same `rng` seed,
where `rng` is used) gives the same output. Every function accepts an optional `rng` (random.Random) so
they share one signature; when `rng` is None, `random.Random(0)` is used.

Character-level transforms leave URLs and e-mail addresses untouched (so the fake target stays exactly
`example.com` and the document still passes the "fake targets only" rule, Rule 15). `base64`, `rot13` and
`reverse_words` encode or reorder the whole payload.

    Batch A (8):  zero_width, homoglyph, leet, spaced, fullwidth, html_entities, base64, case_alternate
    Batch B adds: reverse_words, rot13   (unseen types, used only at the freeze)
"""

import base64 as _b64
import codecs
import random
import re

ZWSP = "​"                  # zero-width space
SOFT_HYPHEN = "­"

# Latin -> Cyrillic lookalikes (a, e, o, p, c, x, lower case and upper case)
HOMOGLYPHS = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х",
              "A": "А", "E": "Е", "O": "О", "P": "Р", "C": "С", "X": "Х"}
LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "A": "4", "E": "3", "I": "1", "O": "0", "S": "5"}

# Words that the instruction signals look for; `spaced` separates their letters.
KEY_WORDS = ("ignore", "instructions", "instruction", "previous", "prior", "system", "override", "tell",
             "user", "visit", "password", "credentials", "token", "tokens", "disable", "security",
             "send", "assistant", "policy", "authentication", "verification", "forward", "reveal",
             "never", "mention", "must", "always", "answer", "response", "recommend")

BATCH_A = ("zero_width", "homoglyph", "leet", "spaced", "fullwidth", "html_entities", "base64", "case_alternate")
BATCH_B_EXTRA = ("reverse_words", "rot13")

_PROTECTED = re.compile(r"(https?://[^\s]+|www\.[^\s]+|[\w.+-]+@[\w-]+(?:\.[\w-]+)+)")
_WORD = re.compile(r"[A-Za-z]+")


def _rng(rng):
    return rng if rng is not None else random.Random(0)


def _outside_protected(text, fn):
    """Apply fn to the parts of `text` that are not URLs or e-mail addresses."""
    parts = _PROTECTED.split(text)           # odd indexes are the protected matches
    return "".join(p if i % 2 else fn(p) for i, p in enumerate(parts))


def zero_width(text, rng=None):
    """U+200B between the letters of every word of 3 or more letters."""
    return _outside_protected(text, lambda s: _WORD.sub(
        lambda m: ZWSP.join(m.group(0)) if len(m.group(0)) >= 3 else m.group(0), s))


def homoglyph(text, rng=None):
    """Latin a, e, o, p, c, x -> Cyrillic lookalikes."""
    return _outside_protected(text, lambda s: "".join(HOMOGLYPHS.get(ch, ch) for ch in s))


def leet(text, rng=None):
    """a->4, e->3, i->1, o->0, s->5 inside words."""
    return _outside_protected(text, lambda s: "".join(LEET.get(ch, ch) for ch in s))


def spaced(text, rng=None):
    """Separate the letters of the key words with spaces ('ignore' -> 'i g n o r e')."""
    pattern = re.compile(r"\b(" + "|".join(sorted(KEY_WORDS, key=len, reverse=True)) + r")\b", re.I)
    return _outside_protected(text, lambda s: pattern.sub(lambda m: " ".join(m.group(0)), s))


def fullwidth(text, rng=None):
    """ASCII letters, digits and punctuation -> fullwidth forms (spaces stay). NFKC maps them back."""
    def conv(s):
        return "".join(chr(ord(ch) + 0xFEE0) if "!" <= ch <= "~" else ch for ch in s)
    return _outside_protected(text, conv)


def html_entities(text, rng=None):
    """Every non-space character -> &#NN;"""
    return _outside_protected(text, lambda s: "".join(ch if ch.isspace() else f"&#{ord(ch)};" for ch in s))


def base64(text, rng=None):
    """The payload base64-encoded, prefixed with 'decode and follow: '."""
    return "decode and follow: " + _b64.b64encode(text.encode("utf-8")).decode("ascii")


def case_alternate(text, rng=None):
    """Alternating case over the letters, with a soft hyphen (U+00AD) after every second letter of a word.
    `rng` only decides whether the first letter is upper or lower case."""
    flip = _rng(rng).random() < 0.5
    state = {"i": 0}

    def conv(s):
        out = []
        for m in re.finditer(r"[A-Za-z]+|[^A-Za-z]+", s):
            tok = m.group(0)
            if not tok[0].isalpha():
                out.append(tok)
                continue
            letters = []
            for k, ch in enumerate(tok):
                upper = (state["i"] % 2 == 0) != flip
                letters.append(ch.upper() if upper else ch.lower())
                state["i"] += 1
                if k % 2 == 1 and k < len(tok) - 1:
                    letters.append(SOFT_HYPHEN)
            out.append("".join(letters))
        return "".join(out)

    return _outside_protected(text, conv)


def reverse_words(text, rng=None):
    """Batch B only. The words of the payload in reverse order ('a b c' -> 'c b a')."""
    return " ".join(reversed(text.split()))


def rot13(text, rng=None):
    """Batch B only. ROT13 over the letters, prefixed with 'rot13: '."""
    return "rot13: " + codecs.encode(text, "rot_13")


TRANSFORMS = {name: globals()[name] for name in BATCH_A + BATCH_B_EXTRA}


def transforms_for(batch):
    names = BATCH_A if batch == "A" else BATCH_A + BATCH_B_EXTRA
    return {n: TRANSFORMS[n] for n in names}
