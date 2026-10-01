# Email Spam and Phishing Detection System

Machine-learning classifier that separates **legitimate**, **spam** and **phishing** emails using NLP (TF-IDF) plus hand-crafted phishing indicators.

**Tools:** Python, NLP, TF-IDF, Scikit-learn, Streamlit, Plotly

## Structure
```
features.py   text cleaning + phishing indicators (URLs, IP links, shorteners, urgency words)
train.py      trains + evaluates the model, saves model.joblib
predict.py    classify one email from the command line
app.py        Streamlit web UI
data/         put your dataset here (sample_emails.csv is a tiny synthetic demo)
```

## Setup
```bash
pip install -r requirements.txt
python train.py --data data/sample_emails.csv      # demo only
```

## Use a real dataset (important for real metrics)
The bundled sample is synthetic and tiny, so its 100% accuracy means nothing. Download a real one and put it in `data/`:
- **SMS Spam Collection** (UCI / Kaggle): ham vs spam
- **Enron-Spam** or **SpamAssassin** public corpus: ham vs spam emails
- **Phishing Email Dataset** (Kaggle, "Phishing_Email.csv"): safe vs phishing

Then:
```bash
python train.py --data data/your_file.csv --text-col "Email Text" --label-col "Email Type"
```
To get true 3-class (ham/spam/phishing), combine a spam dataset and a phishing dataset into one CSV with `text,label` columns.

## Run
```bash
python predict.py "Verify your account at http://192.168.1.5/login"
streamlit run app.py
```

## How false positives are reduced
1. URLs, emails and numbers are normalised to tokens, so the model learns patterns and not specific strings.
2. TF-IDF with bigrams and sublinear scaling down-weights common words.
3. Extra features (IP-based links, shorteners, urgency words) catch phishing that wording alone misses.
4. `class_weight="balanced"` handles imbalanced classes.
5. An adjustable confidence threshold (`--threshold` or the app slider) only flags emails above a chosen probability.

Evaluate on the held-out test set with the printed precision/recall, confusion matrix and false-positive count.

## Ideas to extend
Compare against Naive Bayes / Linear SVM, add email header analysis (SPF/DKIM, Reply-To mismatch), and parse `.eml` files.
