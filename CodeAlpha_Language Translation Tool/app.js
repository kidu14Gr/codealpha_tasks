const API_BASE_URL = "https://api.cognitive.microsofttranslator.com";
const API_REGION = window.TRANSLATOR_REGION || "";
const API_KEY = window.TRANSLATOR_API_KEY || "";

const MAX_CHARS = 5000;
const DEBOUNCE_MS = 600;
const RECENT_PAIR_LIMIT = 5;
const HISTORY_LIMIT = 10;

const languageOptions = [
  { code: "auto", name: "Auto Detect" },
  { code: "af", name: "Afrikaans" },
  { code: "sq", name: "Albanian" },
  { code: "am", name: "Amharic" },
  { code: "ar", name: "Arabic" },
  { code: "hy", name: "Armenian" },
  { code: "as", name: "Assamese" },
  { code: "az", name: "Azerbaijani" },
  { code: "bn", name: "Bangla" },
  { code: "ba", name: "Bashkir" },
  { code: "eu", name: "Basque" },
  { code: "bs", name: "Bosnian" },
  { code: "bg", name: "Bulgarian" },
  { code: "ca", name: "Catalan" },
  { code: "zh-Hans", name: "Chinese (Simplified)" },
  { code: "zh-Hant", name: "Chinese (Traditional)" },
  { code: "hr", name: "Croatian" },
  { code: "cs", name: "Czech" },
  { code: "da", name: "Danish" },
  { code: "nl", name: "Dutch" },
  { code: "en", name: "English" },
  { code: "et", name: "Estonian" },
  { code: "fj", name: "Fijian" },
  { code: "fil", name: "Filipino" },
  { code: "fi", name: "Finnish" },
  { code: "fr", name: "French" },
  { code: "fr-ca", name: "French (Canada)" },
  { code: "gl", name: "Galician" },
  { code: "ka", name: "Georgian" },
  { code: "de", name: "German" },
  { code: "el", name: "Greek" },
  { code: "gu", name: "Gujarati" },
  { code: "ht", name: "Haitian Creole" },
  { code: "ha", name: "Hausa" },
  { code: "he", name: "Hebrew" },
  { code: "hi", name: "Hindi" },
  { code: "hu", name: "Hungarian" },
  { code: "is", name: "Icelandic" },
  { code: "id", name: "Indonesian" },
  { code: "ga", name: "Irish" },
  { code: "it", name: "Italian" },
  { code: "ja", name: "Japanese" },
  { code: "kn", name: "Kannada" },
  { code: "kk", name: "Kazakh" },
  { code: "ko", name: "Korean" },
  { code: "ku", name: "Kurdish (Central)" },
  { code: "ky", name: "Kyrgyz" },
  { code: "lo", name: "Lao" },
  { code: "lv", name: "Latvian" },
  { code: "lt", name: "Lithuanian" },
  { code: "mk", name: "Macedonian" },
  { code: "mg", name: "Malagasy" },
  { code: "ms", name: "Malay" },
  { code: "ml", name: "Malayalam" },
  { code: "mt", name: "Maltese" },
  { code: "mi", name: "Maori" },
  { code: "mr", name: "Marathi" },
  { code: "mn-Cyrl", name: "Mongolian (Cyrillic)" },
  { code: "my", name: "Myanmar" },
  { code: "ne", name: "Nepali" },
  { code: "nb", name: "Norwegian" },
  { code: "or", name: "Odia" },
  { code: "ps", name: "Pashto" },
  { code: "fa", name: "Persian" },
  { code: "pl", name: "Polish" },
  { code: "pt", name: "Portuguese" },
  { code: "pa", name: "Punjabi" },
  { code: "ro", name: "Romanian" },
  { code: "ru", name: "Russian" },
  { code: "sm", name: "Samoan" },
  { code: "sr-Cyrl", name: "Serbian (Cyrillic)" },
  { code: "sr-Latn", name: "Serbian (Latin)" },
  { code: "sk", name: "Slovak" },
  { code: "sl", name: "Slovenian" },
  { code: "es", name: "Spanish" },
  { code: "sw", name: "Swahili" },
  { code: "sv", name: "Swedish" },
  { code: "ta", name: "Tamil" },
  { code: "te", name: "Telugu" },
  { code: "th", name: "Thai" },
  { code: "to", name: "Tongan" },
  { code: "tr", name: "Turkish" },
  { code: "uk", name: "Ukrainian" },
  { code: "ur", name: "Urdu" },
  { code: "uz", name: "Uzbek" },
  { code: "vi", name: "Vietnamese" },
  { code: "cy", name: "Welsh" },
  { code: "xh", name: "Xhosa" },
  { code: "yo", name: "Yoruba" },
  { code: "zu", name: "Zulu" }
];

const sourceLangSelect = document.getElementById("sourceLang");
const targetLangSelect = document.getElementById("targetLang");
const sourceText = document.getElementById("sourceText");
const outputBox = document.getElementById("outputBox");
const charCount = document.getElementById("charCount");
const detectedLang = document.getElementById("detectedLang");
const loadingState = document.getElementById("loadingState");
const recentPairsContainer = document.getElementById("recentPairsContainer");
const historyList = document.getElementById("historyList");
const themeToggle = document.getElementById("themeToggle");
const themeLabel = document.getElementById("themeLabel");

const codeToName = new Map(languageOptions.map((lang) => [lang.code.toLowerCase(), lang.name]));

let debounceTimer;
let translationHistory = [];
let recentPairs = [];

function showToast(message) {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 2400);
}

function populateLanguageDropdowns() {
  for (const lang of languageOptions) {
    const srcOption = document.createElement("option");
    srcOption.value = lang.code;
    srcOption.textContent = lang.name;
    sourceLangSelect.appendChild(srcOption);

    if (lang.code !== "auto") {
      const targetOption = document.createElement("option");
      targetOption.value = lang.code;
      targetOption.textContent = lang.name;
      targetLangSelect.appendChild(targetOption);
    }
  }

  sourceLangSelect.value = "auto";
  targetLangSelect.value = "fr";
}

function updateCharacterCount() {
  const count = sourceText.value.length;
  charCount.textContent = `${count} / ${MAX_CHARS}`;
  charCount.classList.toggle("warning", count >= 4500);
}

function setLoading(isLoading) {
  loadingState.classList.toggle("hidden", !isLoading);
}

function persistRecentPair(source, target) {
  if (!source || !target) return;
  const pairKey = `${source}:${target}`;
  recentPairs = recentPairs.filter((item) => `${item.source}:${item.target}` !== pairKey);
  recentPairs.unshift({ source, target });
  recentPairs = recentPairs.slice(0, RECENT_PAIR_LIMIT);
  localStorage.setItem("lingoflow_recent_pairs", JSON.stringify(recentPairs));
  renderRecentPairs();
}

function renderRecentPairs() {
  recentPairsContainer.innerHTML = "";

  if (!recentPairs.length) {
    recentPairsContainer.innerHTML = '<span class="empty-hint">No recent pairs yet.</span>';
    return;
  }

  for (const pair of recentPairs) {
    const button = document.createElement("button");
    button.className = "pill";
    const sourceName = codeToName.get(pair.source.toLowerCase()) || pair.source;
    const targetName = codeToName.get(pair.target.toLowerCase()) || pair.target;
    button.textContent = `${sourceName} -> ${targetName}`;
    button.type = "button";
    button.addEventListener("click", () => {
      sourceLangSelect.value = pair.source;
      targetLangSelect.value = pair.target;
      translateWithDebounce();
    });
    recentPairsContainer.appendChild(button);
  }
}

function pushHistoryEntry(entry) {
  translationHistory.unshift(entry);
  translationHistory = translationHistory.slice(0, HISTORY_LIMIT);
  renderHistory();
}

function renderHistory() {
  historyList.innerHTML = "";
  if (!translationHistory.length) {
    historyList.innerHTML = '<li class="empty-hint">Your last 10 translations will appear here.</li>';
    return;
  }

  translationHistory.forEach((item) => {
    const li = document.createElement("li");
    li.className = "history-item";
    li.innerHTML = `
      <div><strong>${item.sourceLabel} -> ${item.targetLabel}</strong></div>
      <div>${item.sourcePreview}</div>
      <small>${item.targetPreview}</small>
    `;
    li.addEventListener("click", () => {
      sourceLangSelect.value = item.source;
      targetLangSelect.value = item.target;
      sourceText.value = item.sourceText;
      outputBox.textContent = item.translatedText;
      updateCharacterCount();
    });
    historyList.appendChild(li);
  });
}

function getFriendlyError(error, status) {
  if (status === 429) return "Rate limit reached. Please wait a moment and try again.";
  if (status === 400) return "This language pair may be unsupported. Please try a different pair.";
  if (status === 401 || status === 403) return "Invalid API credentials. Check your API key and region.";
  if (!navigator.onLine) return "You appear to be offline. Please check your internet connection.";
  return error?.message || "Unable to translate right now. Please try again shortly.";
}

async function callTranslatorApi(text, source, target) {
  if (!API_KEY || !API_REGION) {
    throw new Error("Missing API configuration. Set TRANSLATOR_API_KEY and TRANSLATOR_REGION.");
  }

  const url = new URL(`${API_BASE_URL}/translate`);
  url.searchParams.set("api-version", "3.0");
  url.searchParams.set("to", target);
  if (source !== "auto") {
    url.searchParams.set("from", source);
  }

  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Ocp-Apim-Subscription-Key": API_KEY,
      "Ocp-Apim-Subscription-Region": API_REGION
    },
    body: JSON.stringify([{ text }])
  });

  if (!response.ok) {
    const error = new Error("Translation API request failed");
    error.status = response.status;
    throw error;
  }

  return response.json();
}

async function translateText() {
  const text = sourceText.value.trim();
  const source = sourceLangSelect.value;
  const target = targetLangSelect.value;

  detectedLang.textContent = "";

  if (!text) {
    outputBox.textContent = "";
    return;
  }

  if (source !== "auto" && source === target) {
    outputBox.textContent = text;
    showToast("Source and target are identical, so output matches input.");
    return;
  }

  setLoading(true);

  try {
    const data = await callTranslatorApi(text, source, target);
    const translationNode = data?.[0];
    const translatedText = translationNode?.translations?.[0]?.text || "";
    outputBox.textContent = translatedText;

    const detectedCode = translationNode?.detectedLanguage?.language?.toLowerCase();
    if (source === "auto" && detectedCode) {
      const detectedName = codeToName.get(detectedCode) || detectedCode;
      detectedLang.textContent = `Detected: ${detectedName}`;
      showToast(`Language detected: ${detectedName}`);
    }

    const effectiveSourceCode = source === "auto" ? (detectedCode || "auto") : source;
    const sourceLabel = codeToName.get(effectiveSourceCode?.toLowerCase()) || effectiveSourceCode;
    const targetLabel = codeToName.get(target.toLowerCase()) || target;

    persistRecentPair(source, target);
    pushHistoryEntry({
      source,
      target,
      sourceLabel,
      targetLabel,
      sourceText: text,
      translatedText,
      sourcePreview: `${text.slice(0, 85)}${text.length > 85 ? "..." : ""}`,
      targetPreview: `${translatedText.slice(0, 90)}${translatedText.length > 90 ? "..." : ""}`
    });
  } catch (error) {
    const message = getFriendlyError(error, error.status);
    outputBox.textContent = "";
    showToast(message);
  } finally {
    setLoading(false);
  }
}

function translateWithDebounce() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(translateText, DEBOUNCE_MS);
}

async function copyText(text, label) {
  if (!text.trim()) {
    showToast(`Nothing to copy from ${label}.`);
    return;
  }

  try {
    await navigator.clipboard.writeText(text);
    showToast(`Copied ${label} text to clipboard!`);
  } catch {
    showToast("Clipboard access failed. Please allow clipboard permission.");
  }
}

function speakText(text, langCode) {
  if (!window.speechSynthesis) {
    showToast("Text-to-speech is not supported in this browser.");
    return;
  }
  if (!text.trim()) {
    showToast("No text available to speak.");
    return;
  }

  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = langCode === "auto" ? "en-US" : langCode;
  window.speechSynthesis.speak(utterance);
}

function swapLanguages() {
  const source = sourceLangSelect.value;
  const target = targetLangSelect.value;

  if (source === "auto") {
    showToast("Auto Detect cannot be swapped directly. Select a source language first.");
    return;
  }

  sourceLangSelect.value = target;
  targetLangSelect.value = source;

  const sourceValue = sourceText.value;
  sourceText.value = outputBox.textContent;
  outputBox.textContent = sourceValue;

  updateCharacterCount();
  translateWithDebounce();
}

function clearAll() {
  sourceText.value = "";
  outputBox.textContent = "";
  detectedLang.textContent = "";
  updateCharacterCount();
}

function applyTheme(theme) {
  document.body.classList.toggle("dark", theme === "dark");
  themeLabel.textContent = theme === "dark" ? "Light Mode" : "Dark Mode";
  localStorage.setItem("lingoflow_theme", theme);
}

function initTheme() {
  const savedTheme = localStorage.getItem("lingoflow_theme");
  const systemPrefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  applyTheme(savedTheme || (systemPrefersDark ? "dark" : "light"));
}

function initStorageData() {
  const storedPairs = localStorage.getItem("lingoflow_recent_pairs");
  if (storedPairs) {
    try {
      recentPairs = JSON.parse(storedPairs);
    } catch {
      recentPairs = [];
    }
  }
  renderRecentPairs();
}

document.getElementById("copySourceBtn").addEventListener("click", () => copyText(sourceText.value, "source"));
document.getElementById("copyTargetBtn").addEventListener("click", () => copyText(outputBox.textContent, "translated"));
document.getElementById("speakSourceBtn").addEventListener("click", () =>
  speakText(sourceText.value, sourceLangSelect.value)
);
document.getElementById("speakTargetBtn").addEventListener("click", () =>
  speakText(outputBox.textContent, targetLangSelect.value)
);
document.getElementById("swapBtn").addEventListener("click", swapLanguages);
document.getElementById("clearBtn").addEventListener("click", clearAll);

themeToggle.addEventListener("click", () => {
  const nextTheme = document.body.classList.contains("dark") ? "light" : "dark";
  applyTheme(nextTheme);
});

sourceText.addEventListener("input", () => {
  updateCharacterCount();
  translateWithDebounce();
});

sourceLangSelect.addEventListener("change", translateWithDebounce);
targetLangSelect.addEventListener("change", translateWithDebounce);

populateLanguageDropdowns();
updateCharacterCount();
initTheme();
initStorageData();
