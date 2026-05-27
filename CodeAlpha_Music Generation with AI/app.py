"""Flask web UI for AI music generation."""

from __future__ import annotations

import traceback
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

import config
from generate import generate_batch, midi_to_wav

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/")
def index():
    return render_template(
        "index.html",
        genres=list(config.GENRES),
        default_notes=config.DEFAULT_NUM_NOTES,
        default_temp=config.DEFAULT_TEMPERATURE,
        min_temp=config.MIN_TEMPERATURE,
        max_temp=config.MAX_TEMPERATURE,
        max_notes=config.MAX_NUM_NOTES,
    )


@app.route("/api/generate", methods=["POST"])
def api_generate():
    payload = request.get_json(silent=True) or {}
    try:
        num_notes = int(payload.get("num_notes", config.DEFAULT_NUM_NOTES))
        temperature = float(payload.get("temperature", config.DEFAULT_TEMPERATURE))
        num_outputs = int(payload.get("num_outputs", config.DEFAULT_NUM_OUTPUTS))
        genre = payload.get("genre") or None

        num_notes = max(50, min(num_notes, config.MAX_NUM_NOTES))
        temperature = max(config.MIN_TEMPERATURE, min(temperature, config.MAX_TEMPERATURE))
        num_outputs = max(1, min(num_outputs, 5))

        paths = generate_batch(
            num_outputs=num_outputs,
            num_notes=num_notes,
            temperature=temperature,
            genre=genre,
            output_dir=config.OUTPUT_DIR,
        )

        files = []
        for midi_path in paths:
            wav_path = midi_path.with_suffix(".wav")
            has_audio = midi_to_wav(midi_path, wav_path)
            files.append(
                {
                    "midi": f"/download/{midi_path.name}",
                    "wav": f"/download/{wav_path.name}" if has_audio else None,
                    "name": midi_path.name,
                }
            )

        return jsonify({"status": "ok", "files": files})
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/files", methods=["GET"])
def list_files():
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mids = sorted(config.OUTPUT_DIR.glob("output_*.mid"))
    files = []
    for midi_path in mids:
        wav_path = midi_path.with_suffix(".wav")
        files.append(
            {
                "midi": f"/download/{midi_path.name}",
                "wav": f"/download/{wav_path.name}" if wav_path.exists() else None,
                "name": midi_path.name,
            }
        )
    return jsonify({"files": files})


@app.route("/download/<path:filename>")
def download_file(filename: str):
    return send_from_directory(config.OUTPUT_DIR, filename, as_attachment=False)


@app.route("/assets/<path:filename>")
def assets_file(filename: str):
    return send_from_directory(config.ASSETS_DIR, filename)


if __name__ == "__main__":
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not config.BEST_MODEL_PATH.exists():
        print(
            "Warning: best_model.pth not found. Train first with: "
            "python download_data.py && python preprocess.py && python train.py"
        )
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.FLASK_DEBUG)
