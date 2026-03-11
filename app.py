from flask import Flask, render_template, request
import re
import numpy as np
import joblib
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load TF-IDF vectorizer
vectorizer = joblib.load("tfidf_summarizer_model.pkl")

# ------------------ TEXT UTILITIES ------------------

def split_paragraphs(text):
    return [p.strip() for p in text.split("\n") if p.strip()]

def split_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text.strip())

def clean_sentence(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r'[^a-zA-Z ]', '', sentence)
    return sentence

# ------------------ ADAPTIVE SUMMARY LENGTH ------------------

def decide_summary_length(sentences, paragraphs, mode):
    n_sent = len(sentences)
    n_para = len(paragraphs)

    if mode == "short":
        return max(n_para, int(n_sent * 0.2))
    elif mode == "medium":
        return max(n_para, int(n_sent * 0.35))
    else:  # detailed
        return max(n_para * 2, int(n_sent * 0.5))

# ------------------ DUPLICATE REMOVAL ------------------

def remove_duplicates(sentences):
    seen = set()
    unique = []
    for s in sentences:
        key = clean_sentence(s)
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique

# ------------------ MAIN SUMMARIZER ------------------

def summarize_text(text, mode):
    paragraphs = split_paragraphs(text)
    sentences = split_sentences(text)
    sentences = remove_duplicates(sentences)

    if len(sentences) < 3:
        return text

    cleaned = [clean_sentence(s) for s in sentences]
    tfidf = vectorizer.transform(cleaned)
    scores = np.asarray(tfidf.sum(axis=1)).ravel()

    summary_len = decide_summary_length(sentences, paragraphs, mode)

    top_indices = scores.argsort()[-summary_len:][::-1]
    top_indices = sorted(top_indices)

    summary = [sentences[i] for i in top_indices]
    return " ".join(summary)

# ------------------ ROUTES ------------------

@app.route('/', methods=['GET', 'POST'])
def index():
    summary = ""
    if request.method == 'POST':

        mode = request.form.get('mode', 'medium')

        # Case 1: Text input
        text = request.form.get('text', '').strip()

        # Case 2: File upload
        file = request.files.get('file')
        if file and file.filename.endswith('.txt'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()

        if text:
            summary = summarize_text(text, mode)

    return render_template('index.html', summary=summary)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
