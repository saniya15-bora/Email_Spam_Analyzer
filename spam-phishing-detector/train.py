"""Train the Email Spam & Phishing Detection model.

Usage:
    python train.py --data data/emails.csv
    python train.py --data data/emails.csv --text-col text --label-col label

CSV needs a text column and a label column. Labels can be e.g. ham/spam/phishing
(multi-class) or ham/spam (binary). Any label that is not "ham"/"legitimate"/"safe"/0
is treated as malicious in the summary metrics.
"""
import argparse
import re

import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from features import clean_text, handcrafted_features

SAFE_LABELS = {"ham", "legitimate", "safe", "not spam", "0", "legit"}


def load(path, text_col, label_col):
    df = pd.read_csv(path, encoding="latin-1")
    df = df[[text_col, label_col]].dropna()
    df.columns = ["text", "label"]
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df.drop_duplicates(subset="text")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--out", default="model.joblib")
    args = ap.parse_args()

    df = load(args.data, args.text_col, args.label_col)
    print(f"Loaded {len(df)} emails\n{df['label'].value_counts()}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
    )

    tfidf = TfidfVectorizer(
        preprocessor=clean_text, ngram_range=(1, 2), min_df=2,
        max_features=50000, sublinear_tf=True, stop_words="english",
    )
    Xtr = hstack([tfidf.fit_transform(X_train), csr_matrix(handcrafted_features(X_train))]).tocsr()
    Xte = hstack([tfidf.transform(X_test), csr_matrix(handcrafted_features(X_test))]).tocsr()

    # class_weight balances rare classes (e.g. phishing); C controls regularisation
    clf = LogisticRegression(max_iter=2000, C=5, class_weight="balanced")
    clf.fit(Xtr, y_train)

    pred = clf.predict(Xte)
    print(classification_report(y_test, pred, digits=3))
    labels = sorted(df["label"].unique())
    print("Confusion matrix (rows=true, cols=pred):", labels)
    print(confusion_matrix(y_test, pred, labels=labels))

    # False-positive rate: legitimate emails wrongly flagged
    safe = y_test.isin(SAFE_LABELS)
    fp = (~pd.Series(pred, index=y_test.index)[safe].isin(SAFE_LABELS)).sum()
    print(f"\nFalse positives (legit flagged as spam/phishing): {fp}/{safe.sum()}")

    joblib.dump({"tfidf": tfidf, "clf": clf, "safe_labels": SAFE_LABELS}, args.out)
    print(f"Saved model -> {args.out}")


if __name__ == "__main__":
    main()
