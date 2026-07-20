# 🔍 Fact-Check Agent

An AI-powered truth verification tool that extracts factual claims from PDF documents and verifies each one against live web evidence.

**⚡ Live Demo:** [fact-ch-xbjxwzqyidpiympehdobps.streamlit.app](https://fact-ch-xbjxwzqyidpiympehdobps.streamlit.app/)

## ✨ Features

- 📄 **PDF Upload** — upload any research paper, report, or whitepaper
- 🔎 **Claim Extraction** — LLM identifies all verifiable factual claims (stats, dates, figures)
- 🌐 **Live Web Search** — each claim is searched against live web sources
- ✅ **AI Verdict** — classified as Verified / Inaccurate / False / Needs Review
- 📊 **Visual Dashboard** — KPI summary cards + interactive results table
- 📥 **CSV Export** — download the full verification report

## 🚀 Quick Start (Local)

```bash
# 1. Clone the repo
git clone https://github.com/arnavtalwar1/fact-check-agent.git
cd fact-check-agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml and paste your Groq or OpenRouter API key

# 4. Run
streamlit run app.py
```

## 🔑 API Key Setup

This app uses **Groq** (recommended — free) or **OpenRouter** for LLM inference.

| Provider | Get Key | Format |
|---|---|---|
| Groq | [console.groq.com/keys](https://console.groq.com/keys) | `gsk_...` |
| OpenRouter | [openrouter.ai/keys](https://openrouter.ai/keys) | `sk-or-...` |

Set the key in `.streamlit/secrets.toml`:
```toml
OPENROUTER_API_KEY = "gsk_your_key_here"
```

## ☁️ Deploy on Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo + `app.py`
4. Under **Advanced settings → Secrets**, add:
   ```toml
   OPENROUTER_API_KEY = "gsk_your_key_here"
   ```
5. Click **Deploy**

## 🗂️ Project Structure

```
├── app.py               # Main Streamlit app
├── claim_extractor.py   # LLM-based claim extraction
├── verifier.py          # LLM-based claim verification
├── web_search.py        # DuckDuckGo search (ddgs)
├── pdf_processor.py     # PDF text extraction (PyMuPDF)
├── report_generator.py  # CSV report generator
├── requirements.txt
└── .streamlit/
    └── secrets.toml.example
```

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM**: Groq (`llama-3.3-70b-versatile`) / OpenRouter
- **Web Search**: DuckDuckGo (`ddgs`)
- **PDF Parsing**: PyMuPDF
