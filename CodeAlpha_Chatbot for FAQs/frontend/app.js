const API_BASE = "http://127.0.0.1:8000";

const chatWindow = document.getElementById("chatWindow");
const messageInput = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const clearChatBtn = document.getElementById("clearChatBtn");
const suggestionChips = document.getElementById("suggestionChips");
const themeToggle = document.getElementById("themeToggle");
const welcomeTime = document.getElementById("welcomeTime");

let latestBotContext = null;

function formatTime(date = new Date()) {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function showToast(message) {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 2200);
}

function scrollToBottom() {
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function createMessageBubble({ role, text, category, confidence, showActions }) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrapper.appendChild(bubble);

  const meta = document.createElement("div");
  meta.className = "meta";
  meta.innerHTML = `<span>${formatTime()}</span>`;

  if (category) {
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = category;
    meta.appendChild(tag);
  }

  if (typeof confidence === "number") {
    const confidenceLabel = document.createElement("span");
    confidenceLabel.textContent = `${Math.round(confidence * 100)}% match`;
    meta.appendChild(confidenceLabel);
  }

  wrapper.appendChild(meta);

  if (showActions) {
    const actions = document.createElement("div");
    actions.className = "bubble-actions";

    const copyBtn = document.createElement("button");
    copyBtn.className = "icon-btn";
    copyBtn.type = "button";
    copyBtn.textContent = "Copy";
    copyBtn.addEventListener("click", async () => {
      await navigator.clipboard.writeText(text);
      showToast("Answer copied.");
    });

    const upBtn = document.createElement("button");
    upBtn.className = "icon-btn";
    upBtn.type = "button";
    upBtn.textContent = "👍";
    upBtn.addEventListener("click", () => submitFeedback("up"));

    const downBtn = document.createElement("button");
    downBtn.className = "icon-btn";
    downBtn.type = "button";
    downBtn.textContent = "👎";
    downBtn.addEventListener("click", () => submitFeedback("down"));

    actions.append(copyBtn, upBtn, downBtn);
    wrapper.appendChild(actions);
  }

  return wrapper;
}

function addMessage(messageData) {
  chatWindow.appendChild(createMessageBubble(messageData));
  scrollToBottom();
}

function renderTypingIndicator() {
  const typing = document.createElement("div");
  typing.id = "typingIndicator";
  typing.className = "message bot";
  typing.innerHTML = `
    <div class="bubble typing">
      <span></span><span></span><span></span>
    </div>
  `;
  chatWindow.appendChild(typing);
  scrollToBottom();
}

function removeTypingIndicator() {
  const typing = document.getElementById("typingIndicator");
  if (typing) typing.remove();
}

async function fetchSuggestions() {
  try {
    const response = await fetch(`${API_BASE}/suggestions`);
    const data = await response.json();
    suggestionChips.innerHTML = "";
    data.suggestions.slice(0, 6).forEach((question) => {
      const chip = document.createElement("button");
      chip.className = "chip";
      chip.type = "button";
      chip.textContent = question;
      chip.addEventListener("click", () => {
        messageInput.value = question;
        handleSend();
      });
      suggestionChips.appendChild(chip);
    });
  } catch {
    suggestionChips.innerHTML = `<span class="chip">Unable to load suggestions</span>`;
  }
}

async function submitFeedback(rating) {
  if (!latestBotContext) return;
  try {
    await fetch(`${API_BASE}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: latestBotContext.question,
        answer: latestBotContext.answer,
        rating,
      }),
    });
    showToast("Thanks for your feedback.");
  } catch {
    showToast("Could not save feedback.");
  }
}

async function handleSend() {
  const message = messageInput.value.trim();
  if (!message) return;

  addMessage({ role: "user", text: message });
  messageInput.value = "";
  renderTypingIndicator();

  try {
    const response = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });

    if (!response.ok) {
      throw new Error("Chat request failed.");
    }

    const data = await response.json();
    const corrected = data.corrected_message && data.corrected_message !== message;
    const note = corrected ? `\n\n(Interpreted as: "${data.corrected_message}")` : "";

    const delay = 800 + Math.floor(Math.random() * 400);
    setTimeout(() => {
      removeTypingIndicator();
      addMessage({
        role: "bot",
        text: `${data.answer}${note}`,
        category: data.category,
        confidence: data.confidence,
        showActions: true,
      });
      latestBotContext = { question: message, answer: data.answer };
    }, delay);
  } catch {
    removeTypingIndicator();
    addMessage({
      role: "bot",
      text: "I am having trouble connecting to the server right now. Please try again.",
      category: "Error",
      showActions: false,
    });
  }
}

function applyTheme(theme) {
  document.body.classList.toggle("dark", theme === "dark");
  themeToggle.textContent = theme === "dark" ? "Light Mode" : "Dark Mode";
  localStorage.setItem("faqbot_theme", theme);
}

function initTheme() {
  const savedTheme = localStorage.getItem("faqbot_theme");
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  applyTheme(savedTheme || (prefersDark ? "dark" : "light"));
}

function clearChat() {
  chatWindow.innerHTML = "";
  addMessage({
    role: "bot",
    text: "Chat cleared. Ask me another FAQ question anytime.",
    category: "System",
    showActions: false,
  });
}

themeToggle.addEventListener("click", () => {
  const next = document.body.classList.contains("dark") ? "light" : "dark";
  applyTheme(next);
});

sendBtn.addEventListener("click", handleSend);
clearChatBtn.addEventListener("click", clearChat);
messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    handleSend();
  }
});

welcomeTime.textContent = formatTime();
initTheme();
fetchSuggestions();
