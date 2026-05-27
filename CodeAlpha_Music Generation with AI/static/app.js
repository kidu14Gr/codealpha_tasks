const tempInput = document.getElementById("temperature");
const notesInput = document.getElementById("numNotes");
const outputsInput = document.getElementById("numOutputs");
const genreSelect = document.getElementById("genre");
const generateBtn = document.getElementById("generateBtn");
const statusEl = document.getElementById("status");
const fileList = document.getElementById("fileList");

function bindRange(input, labelEl) {
  const update = () => {
    labelEl.textContent = input.value;
  };
  input.addEventListener("input", update);
  update();
}

bindRange(tempInput, document.getElementById("tempValue"));
bindRange(notesInput, document.getElementById("notesValue"));
bindRange(outputsInput, document.getElementById("outputsValue"));

function renderFiles(files) {
  if (!files.length) {
    fileList.innerHTML = '<p class="muted">No files generated yet.</p>';
    return;
  }

  fileList.innerHTML = "";
  files.forEach((file, idx) => {
    const card = document.createElement("div");
    card.className = "track-card";
    card.innerHTML = `<h3>${file.name || `Track ${idx + 1}`}</h3>`;

    if (file.wav) {
      const audio = document.createElement("audio");
      audio.controls = true;
      audio.src = file.wav;
      card.appendChild(audio);
    } else {
      const note = document.createElement("p");
      note.className = "muted";
      note.textContent = "Audio preview unavailable (install FluidSynth for WAV playback).";
      card.appendChild(note);
    }

    const actions = document.createElement("div");
    actions.className = "track-actions";
    actions.innerHTML = `<a href="${file.midi}" download>Download MIDI</a>`;
    if (file.wav) {
      actions.innerHTML += `<a href="${file.wav}" download>Download WAV</a>`;
    }
    card.appendChild(actions);
    fileList.appendChild(card);
  });
}

async function loadExisting() {
  try {
    const res = await fetch("/api/files");
    const data = await res.json();
    renderFiles(data.files || []);
  } catch {
    /* ignore */
  }
}

generateBtn.addEventListener("click", async () => {
  generateBtn.disabled = true;
  statusEl.textContent = "Generating music... this may take a few seconds.";

  try {
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        temperature: Number(tempInput.value),
        num_notes: Number(notesInput.value),
        num_outputs: Number(outputsInput.value),
        genre: genreSelect.value || null,
      }),
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.message || "Generation failed");
    }

    renderFiles(data.files || []);
    statusEl.textContent = `Generated ${data.files.length} track(s) successfully.`;
  } catch (err) {
    statusEl.textContent = err.message;
  } finally {
    generateBtn.disabled = false;
  }
});

loadExisting();
