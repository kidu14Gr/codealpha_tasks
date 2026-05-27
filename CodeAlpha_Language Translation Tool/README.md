# CodeAlpha - Language Translation Tool (LingoFlow)

LingoFlow is a portfolio-grade, responsive language translation web app built for the CodeAlpha AI Internship project. It uses the Microsoft Azure Translator API for high-quality multilingual translation and includes productivity and UX features beyond a basic translator.

## Features

- Clean two-panel translation UI (source and translated text)
- 90+ selectable languages, including `Auto Detect`
- Auto language detection with detected-language indicator
- Debounced translation requests (600ms) to reduce API spam
- Copy buttons for both source and translated text
- Text-to-speech for both source and translated text (Web Speech API)
- Language swap button (source <-> target)
- Recent language pairs (persisted in `localStorage`)
- Live character count with warning near limit (`4500/5000`)
- Clear button to reset source + output fields
- Session translation history (last 10 entries) with restore on click
- Dark/light mode toggle persisted in `localStorage`
- Loading spinner while translation is in progress
- Toast notifications for copy, detection, and errors
- Mobile-friendly responsive design

## Tech Stack

- HTML5
- CSS3 (custom modern UI)
- Vanilla JavaScript (modular functions)
- Microsoft Azure Translator Text API (`v3.0`)

## Project Structure

```text
CodeAlpha_Language Translation Tool
├── index.html
├── style.css
├── app.js
├── .env.example
├── README.md
└── assets/
```

## Setup Instructions

### 1. Clone and open the project

```bash
git clone https://github.com/kidu14Gr/codealpha_tasks.git
cd codealpha_tasks/"CodeAlpha_Language Translation Tool"
```

### 2. Create your environment file

Create `.env` from `.env.example` and fill in your Azure credentials:

```env
TRANSLATOR_API_KEY=your_key_here
TRANSLATOR_REGION=your_region_here
```

### 3. Expose env values to the browser

Because this is a static frontend app, browsers cannot read `.env` directly. Use any simple local server or build pipeline that injects these values into `window` globals before loading `app.js`. Example snippet in `index.html`:

```html
<script>
  window.TRANSLATOR_API_KEY = "YOUR_KEY";
  window.TRANSLATOR_REGION = "YOUR_REGION";
</script>
<script src="app.js"></script>
```

For real deployments, route translation requests through a backend proxy so your API key is not exposed to clients.

### 4. Run the app locally

Use any static server, for example:

```bash
python3 -m http.server 5500
```

Then open: `http://localhost:5500`

## Getting Azure Translator API Key

1. Go to the [Azure Portal](https://portal.azure.com/).
2. Create a **Translator** resource in Azure AI Services.
3. Open the resource and copy:
   - **Key** (API key)
   - **Location/Region** (e.g. `eastus`, `westeurope`)
4. Put them into your `.env` or runtime-injected config.

## Supported Languages

The app includes **90+ languages** such as English, French, Arabic, Hindi, Spanish, Chinese (Simplified/Traditional), Japanese, Korean, Portuguese, Russian, Turkish, German, and many more, plus **Auto Detect**.

## Screenshots

Add screenshots under `assets/` and reference them here:

- `assets/desktop-light.png`
- `assets/desktop-dark.png`
- `assets/mobile-view.png`

## Notes on Security and Production Readiness

- Never commit real API keys.
- Prefer a backend translation proxy in production.
- Handle API quotas/rate limits gracefully (already implemented via friendly UI errors).

## Evaluation Checklist Coverage

- [x] Text input UI
- [x] Source + target language selection (50+ languages)
- [x] Translation API integration
- [x] Output display
- [x] Copy translated text
- [x] Text-to-speech (source and output)
- [x] Auto-detect source language
- [x] Swap languages
- [x] Recent language pairs
- [x] Character count and warning
- [x] Clear fields button
- [x] Translation history panel
- [x] Copy source text
- [x] Dark/light mode with persistence
- [x] Loading feedback and toasts
- [x] Responsive polished design
- [x] Debounced translation
- [x] Graceful error handling
