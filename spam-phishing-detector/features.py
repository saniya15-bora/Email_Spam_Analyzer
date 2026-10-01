"""Text cleaning + hand-crafted phishing indicators shared by training and prediction."""
import re

import numpy as np

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
IP_URL_RE = re.compile(r"https?://\d{1,3}(?:\.\d{1,3}){3}")
EMAIL_RE = re.compile(r"\S+@\S+")
SHORTENERS = ("bit.ly", "tinyurl", "goo.gl", "t.co", "ow.ly", "is.gd", "cutt.ly")
URGENCY = (
    "urgent", "immediately", "verify", "suspend", "account", "password", "login",
    "confirm", "limited time", "act now", "click here", "security alert",
    "winner", "prize", "free", "bank", "invoice", "expire",
)


def clean_text(text: str) -> str:
    """Lowercase and replace URLs / emails / numbers with tokens (keeps the signal, drops noise)."""
    text = str(text).lower()
    text = URL_RE.sub(" urltoken ", text)
    text = EMAIL_RE.sub(" emailtoken ", text)
    text = re.sub(r"\d+", " numtoken ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_indicators(text: str) -> dict:
    """Human-readable indicators; also used to explain predictions in the app."""
    raw = str(text)
    low = raw.lower()
    urls = URL_RE.findall(raw)
    letters = [c for c in raw if c.isalpha()]
    return {
        "num_urls": len(urls),
        "has_ip_url": int(bool(IP_URL_RE.search(raw))),
        "has_shortener": int(any(s in low for s in SHORTENERS)),
        "urgency_words": sum(w in low for w in URGENCY),
        "exclamations": raw.count("!"),
        "caps_ratio": round(sum(c.isupper() for c in letters) / max(len(letters), 1), 3),
        "has_html": int("<html" in low or "<a href" in low),
        "length": min(len(raw), 5000) / 5000,
    }


def handcrafted_features(texts) -> np.ndarray:
    rows = []
    for t in texts:
        f = extract_indicators(t)
        rows.append([
            min(f["num_urls"], 10) / 10, f["has_ip_url"], f["has_shortener"],
            min(f["urgency_words"], 10) / 10, min(f["exclamations"], 10) / 10,
            f["caps_ratio"], f["has_html"], f["length"],
        ])
    return np.array(rows, dtype=float)
