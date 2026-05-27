"""Flask backend for CodeAlpha FAQ Chatbot."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from nlp.matcher import FAQMatcher

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
FEEDBACK_PATH = BASE_DIR / "feedback.json"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)

matcher = FAQMatcher()

SUGGESTED_QUESTIONS = [
    "How do I reset my password?",
    "What payment methods do you accept?",
    "How do I invite team members to my workspace?",
    "Do you support single sign-on?",
    "How can I report a bug?",
    "How do I export project data?",
]


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "faq-chatbot"})


@app.route("/suggestions", methods=["GET"])
def suggestions():
    return jsonify({"suggestions": SUGGESTED_QUESTIONS})


@app.route("/chat", methods=["POST"])
def chat():
    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    user_message = str(payload.get("message", "")).strip()

    if not user_message:
        return jsonify({"error": "Message is required."}), 400

    result = matcher.find_best_match(user_message)
    return jsonify(
        {
            "answer": result["answer"],
            "category": result["category"],
            "confidence": result["confidence"],
            "matched_question": result["matched_question"],
            "method": result["method"],
            "corrected_message": result["corrected_message"],
        }
    )


@app.route("/feedback", methods=["POST"])
def feedback():
    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    feedback_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "message": payload.get("message", ""),
        "answer": payload.get("answer", ""),
        "rating": payload.get("rating", ""),
    }

    existing = []
    if FEEDBACK_PATH.exists():
        try:
            with FEEDBACK_PATH.open("r", encoding="utf-8") as file:
                existing = json.load(file)
        except json.JSONDecodeError:
            existing = []

    existing.append(feedback_entry)
    with FEEDBACK_PATH.open("w", encoding="utf-8") as file:
        json.dump(existing, file, indent=2)

    return jsonify({"status": "saved"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
