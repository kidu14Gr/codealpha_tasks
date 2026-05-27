# CodeAlpha FAQ Chatbot

A portfolio-grade FAQ chatbot for a fictional SaaS product called **CloudDesk**. The bot uses NLP preprocessing and dual-stage similarity matching to return relevant answers from a curated FAQ dataset.

## Domain

**Topic:** SaaS customer support (account, billing, plans, security, integrations, collaboration, and technical issues)

## Features

- 40 curated FAQ Q&A pairs in `data/faq.json`
- NLP preprocessing with NLTK:
  - lowercasing
  - punctuation removal
  - tokenization
  - stopword removal
  - lemmatization
- Spelling correction using `pyspellchecker`
- Dual-layer matching:
  1. TF-IDF + cosine similarity (primary)
  2. sentence-transformers semantic fallback (`all-MiniLM-L6-v2`)
- Fallback response when confidence is below threshold
- Returns answer + category + confidence + matched question
- Modern chat UI with:
  - user/bot chat bubbles
  - online bot header
  - suggested question chips
  - typing indicator
  - timestamps
  - clear chat button
  - copy answer button
  - thumbs up/down feedback buttons
  - dark/light mode (persisted in localStorage)
  - responsive mobile-friendly design
- Feedback logging to `feedback.json`

## Project Structure

```text
CodeAlpha_Chatbot for FAQs
├── app.py
├── requirements.txt
├── .env.example
├── data/
│   └── faq.json
├── nlp/
│   ├── preprocessor.py
│   └── matcher.py
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── feedback.json   (generated at runtime)
└── README.md
```

## NLP Pipeline (How it Works)

1. User sends a message to `/chat`.
2. Text is corrected for obvious spelling issues.
3. Text is normalized via NLTK preprocessing.
4. Bot computes TF-IDF cosine similarity against all FAQ questions.
5. If top TF-IDF confidence is weak, it attempts semantic similarity using sentence embeddings.
6. If still below threshold, bot returns a graceful fallback message.
7. API responds with:
   - `answer`
   - `category`
   - `confidence`
   - `matched_question`

## Setup Instructions

### 1. Go to project folder

```bash
cd "CodeAlpha_Chatbot for FAQs"
```

### 2. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run backend server

```bash
python app.py
```

Server runs at: `http://127.0.0.1:8000`

### 5. Open chatbot UI

Open this URL in your browser:

`http://127.0.0.1:8000`

## Example Questions to Try

- How do I reset my password?
- Do you support single sign-on?
- What payment methods do you accept?
- How do I export project data?
- Can I invite external clients to selected projects?
- Why is my account locked?

## Screenshot

Add screenshot image(s) in this folder and reference here, e.g.:

- `frontend-screenshot.png`

## Notes

- The semantic fallback downloads a sentence-transformer model on first run.
- Feedback from thumbs up/down is appended to `feedback.json`.
