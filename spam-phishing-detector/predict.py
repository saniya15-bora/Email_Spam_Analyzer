"""Classify a single email.

    python predict.py "Your account is suspended, verify at http://192.168.1.5/login"
    python predict.py --file email.txt
"""
import argparse

import joblib
from scipy.sparse import hstack, csr_matrix

from features import extract_indicators, handcrafted_features


def load_model(path="model.joblib"):
    return joblib.load(path)


def classify(text: str, bundle: dict, threshold: float = 0.5) -> dict:
    """Returns label, confidence, per-class probabilities and rule-based indicators.

    threshold: a message is only flagged if the top malicious class probability >= threshold.
    Raising it (e.g. 0.7) trades a few missed spam for fewer false positives.
    """
    tfidf, clf, safe = bundle["tfidf"], bundle["clf"], bundle["safe_labels"]
    X = hstack([tfidf.transform([text]), csr_matrix(handcrafted_features([text]))]).tocsr()
    probs = dict(zip(clf.classes_, clf.predict_proba(X)[0]))
    bad = {k: v for k, v in probs.items() if k not in safe}
    label = max(probs, key=probs.get)
    if label not in safe and bad[label] < threshold:
        label = next((k for k in probs if k in safe), label)
    return {
        "label": label,
        "confidence": float(probs[label]),
        "probabilities": {k: float(v) for k, v in probs.items()},
        "indicators": extract_indicators(text),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?")
    ap.add_argument("--file")
    ap.add_argument("--model", default="model.joblib")
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    body = open(a.file, encoding="utf-8", errors="ignore").read() if a.file else a.text
    r = classify(body, load_model(a.model), a.threshold)
    print(f"Verdict: {r['label'].upper()}  (confidence {r['confidence']:.1%})")
    print("Probabilities:", {k: f'{v:.1%}' for k, v in r["probabilities"].items()})
    print("Indicators:", r["indicators"])
