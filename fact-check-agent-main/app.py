import streamlit as st
import os
import pandas as pd
import logging
import time

# Import modular backend components
from pdf_processor import extract_text
from claim_extractor import extract_claims
from web_search import search_claim
from verifier import verify_claim
from report_generator import generate_report_df, convert_df_to_csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up page configurations
st.set_page_config(
    page_title="Fact-Check Agent | AI-Powered Truth Verification",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stMarkdown {
    font-family: 'Outfit', sans-serif;
}

/* ── Header ── */
.header-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
    padding: 2.5rem 3rem;
    border-radius: 20px;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 12px 40px rgba(30, 58, 138, 0.25);
    border: 1px solid rgba(255,255,255,0.06);
    position: relative;
    overflow: hidden;
}
.header-container::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: rgba(96,165,250,0.08);
    pointer-events: none;
}
.header-title {
    font-size: 2.8rem;
    font-weight: 700;
    margin: 0 0 0.4rem 0;
    background: linear-gradient(to right, #60a5fa, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.03em;
}
.header-tagline {
    font-size: 1.05rem;
    font-weight: 300;
    color: #94a3b8;
    margin: 0;
}

/* ── KPI Cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}
.kpi-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 1.25rem 1rem;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 12px rgba(0,0,0,0.03);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    text-align: center;
}
.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 24px rgba(0,0,0,0.07);
}
.kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-bottom: 0.5rem;
}
.kpi-value {
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
}

/* ── Source Cards ── */
.source-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s, background 0.2s;
}
.source-card:hover {
    border-color: #93c5fd;
    background: #eff6ff;
}
.source-num {
    font-size: 0.72rem;
    font-weight: 700;
    color: #3b82f6;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.2rem;
}
.source-title {
    font-weight: 600;
    color: #1e3a8a;
    font-size: 0.93rem;
    margin-bottom: 0.3rem;
    line-height: 1.35;
}
.source-snippet {
    font-size: 0.83rem;
    color: #475569;
    margin-bottom: 0.45rem;
    line-height: 1.5;
}
.source-url {
    font-size: 0.78rem;
    color: #3b82f6;
    text-decoration: none;
    word-break: break-all;
}
.source-url:hover { text-decoration: underline; }

/* ── Correction Box ── */
.correction-box {
    background: #fffbeb;
    border-left: 4px solid #f59e0b;
    padding: 0.9rem 1rem;
    border-radius: 0 10px 10px 0;
    margin-top: 0.5rem;
}

/* ── API Status pill ── */
.api-ok   { color:#059669; font-weight:600; }
.api-miss { color:#dc2626; font-weight:600; }

/* ── Hide Streamlit default branding ── */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Session State ────────────────────────────────────────────────────────────
for key, default in [
    ("extracted_text", None),
    ("claims", None),
    ("results", None),
    ("processing_done", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── API Key (backend only) ───────────────────────────────────────────────────
env_key = os.environ.get("OPENROUTER_API_KEY", "")
secrets_key = ""
try:
    secrets_key = st.secrets.get("OPENROUTER_API_KEY", "")
except Exception:
    pass

api_key = secrets_key or env_key
is_groq_key = api_key.startswith("gsk_")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # API key status indicator
    if api_key:
        key_type = "Groq" if is_groq_key else "OpenRouter"
        masked = api_key[:6] + "••••••••" + api_key[-3:]
        st.markdown(f"🔑 **API Key:** <span class='api-ok'>✔ {key_type} ({masked})</span>", unsafe_allow_html=True)
    else:
        st.markdown("🔑 **API Key:** <span class='api-miss'>✘ Not configured</span>", unsafe_allow_html=True)
        st.caption("Set `OPENROUTER_API_KEY` as an environment variable or in `.streamlit/secrets.toml`.")

    st.divider()

    # ── Model Selection ──
    groq_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "meta-llama/llama-4-scout-17b-16e-instruct",
    ]
    openrouter_models = [
        "meta-llama/llama-3.3-70b-instruct:free",
        "qwen/qwen-2.5-coder-32b-instruct:free",
        "google/gemini-2.0-flash-exp:free",
        "qwen/qwen3-235b-a22b:free",
        "deepseek/deepseek-r1:free",
    ]

    if is_groq_key:
        model_options = groq_models + openrouter_models
        default_idx = 0
    else:
        model_options = openrouter_models + groq_models
        default_idx = 0

    model_choice = st.selectbox(
        "🤖 AI Model",
        options=model_options,
        index=default_idx,
        help="Top options match your detected key type. Groq models (gsk_...) are listed first if a Groq key is detected.",
    )

    st.divider()

    max_search_results = st.slider(
        "🔎 Search Results per Claim",
        min_value=1, max_value=8, value=4,
        help="Web sources fetched per claim for evidence."
    )
    max_claims_to_check = st.slider(
        "📋 Max Claims to Verify",
        min_value=1, max_value=30, value=10,
        help="Caps the number of claims verified (saves time & tokens)."
    )

    st.divider()
    if st.button("🗑️ Clear Results", use_container_width=True):
        for k in ["extracted_text", "claims", "results", "processing_done"]:
            st.session_state[k] = None if k != "processing_done" else False
        st.rerun()

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-container">
    <div class="header-title">🔍 Fact-Check Agent</div>
    <div class="header-tagline">AI-powered truth verification for research, marketing &amp; business documents.</div>
</div>
""", unsafe_allow_html=True)

# ── UPLOAD STAGE ─────────────────────────────────────────────────────────────
if not st.session_state.processing_done:

    st.markdown("### 📤 Upload a PDF Document")
    st.write("Upload any PDF report, whitepaper, or business document. The agent extracts factual claims and verifies each one against live web evidence.")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Max file size: 20 MB."
    )

    if uploaded_file is not None:
        if not api_key:
            st.error("❌ **API Key missing.** Set `OPENROUTER_API_KEY` as an environment variable or in `.streamlit/secrets.toml`.")
        else:
            if st.button("🚀 Verify Document Facts", use_container_width=True, type="primary"):

                prog = st.container()
                prog.markdown("#### ⚙️ Processing Document…")
                s1 = prog.empty()
                s2 = prog.empty()
                s3 = prog.empty()

                # ── Step 1: Extract PDF Text ──
                s1.info("⏳ **Step 1:** Extracting text from PDF…")
                try:
                    extracted_text = extract_text(uploaded_file)
                    st.session_state.extracted_text = extracted_text
                    s1.success(f"✅ **Step 1:** PDF text extracted ({len(extracted_text):,} characters).")
                except Exception as e:
                    s1.error(f"❌ **Step 1 Failed:** {str(e)}")
                    st.stop()

                # ── Step 2: Extract Claims ──
                s2.info(f"⏳ **Step 2:** Identifying factual claims with `{model_choice}`…")
                try:
                    claims = extract_claims(extracted_text, api_key, model_choice)
                    if not claims:
                        s2.warning("⚠️ **Step 2:** No factual claims detected in this document.")
                        st.info("The document may not contain verifiable statistics, dates, or numeric claims.")
                        st.stop()
                    st.session_state.claims = claims
                    s2.success(f"✅ **Step 2:** Found **{len(claims)}** factual claims.")
                except Exception as e:
                    s2.error(f"❌ **Step 2 Failed:** {str(e)}")
                    st.stop()

                # ── Step 3: Search & Verify ──
                s3.info("⏳ **Step 3:** Searching the web & verifying each claim…")
                results = []
                claims_to_check = claims[:max_claims_to_check]
                progress_bar = prog.progress(0.0)

                for idx, claim_obj in enumerate(claims_to_check):
                    claim = claim_obj["claim"]
                    short = claim[:70] + ("…" if len(claim) > 70 else "")
                    s3.info(f"⏳ **Step 3 ({idx+1}/{len(claims_to_check)}):** *\"{short}\"*")

                    evidence = search_claim(claim, max_results=max_search_results)
                    verdict  = verify_claim(claim, evidence, api_key, model_choice)

                    results.append({
                        "claim":        claim,
                        "status":       verdict.get("status",       "Needs Review"),
                        "confidence":   verdict.get("confidence",   0),
                        "reason":       verdict.get("reason",       ""),
                        "correct_fact": verdict.get("correct_fact", ""),
                        "evidence":     evidence,
                    })
                    progress_bar.progress((idx + 1) / len(claims_to_check))
                    # Brief pause to respect API rate limits
                    if idx < len(claims_to_check) - 1:
                        time.sleep(1.0)

                s3.success("✅ **Step 3:** Verification complete.")
                st.session_state.results = results
                st.session_state.processing_done = True
                time.sleep(0.8)
                st.rerun()

# ── RESULTS STAGE ────────────────────────────────────────────────────────────
else:
    results = st.session_state.results or []
    total         = len(results)
    verified      = sum(1 for r in results if r["status"] == "Verified")
    inaccurate    = sum(1 for r in results if r["status"] == "Inaccurate")
    false_claims  = sum(1 for r in results if r["status"] == "False")
    needs_review  = sum(1 for r in results if r["status"] == "Needs Review")
    ver_rate      = round((verified / total) * 100) if total > 0 else 0

    # ── KPI Cards (native Streamlit metrics) ──────────────────────────────
    st.markdown("### 📊 Verification Report Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Claims",   total)
    c2.metric("✅ Verified",    verified,     delta=f"{round(verified/total*100)}%" if total else None)
    c3.metric("⚠️ Inaccurate", inaccurate)
    c4.metric("❌ False",       false_claims)
    c5.metric("❓ Needs Review",needs_review)

    # accuracy progress bar
    st.markdown(f"**Accuracy Rate:** {ver_rate}%")
    st.progress(ver_rate / 100)

    st.divider()

    # ── CSV Download + Table Header ──────────────────────────────────────
    report_df  = generate_report_df(results)
    csv_bytes  = convert_df_to_csv(report_df)

    col1, col2 = st.columns([5, 1])
    col1.markdown("### 📋 Extracted Claim Verdicts")
    col2.download_button(
        "📥 Download CSV",
        data=csv_bytes,
        file_name="fact_check_report.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # ── Results Table ────────────────────────────────────────────────────
    STATUS_MAP = {
        "Verified":     "✅ Verified",
        "Inaccurate":   "⚠️ Inaccurate",
        "False":        "❌ False",
        "Needs Review": "❓ Needs Review",
    }

    table_df = pd.DataFrame([
        {
            "Claim":               r["claim"],
            "Status":              STATUS_MAP.get(r["status"], r["status"]),
            "Confidence":          r["confidence"],
            "Suggested Correction": r["correct_fact"] if r["correct_fact"] else "—",
        }
        for r in results
    ])

    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        height=min(38 + len(results) * 36, 500),
        column_config={
            "Claim": st.column_config.TextColumn("Claim", width="large"),
            "Status": st.column_config.TextColumn("Status", width="small"),
            "Confidence": st.column_config.ProgressColumn(
                "Confidence", format="%d%%", min_value=0, max_value=100, width="small"
            ),
            "Suggested Correction": st.column_config.TextColumn("Suggested Correction", width="large"),
        },
    )

    st.divider()

    # ── Evidence Inspector ───────────────────────────────────────────────
    st.markdown("### 🔍 Evidence & Reasoning Inspector")
    st.caption("Select a claim to see the AI's reasoning and the live web sources used.")

    selected_claim = st.selectbox(
        "Select claim:",
        options=[r["claim"] for r in results],
        label_visibility="collapsed",
    )

    sel = next((r for r in results if r["claim"] == selected_claim), None)

    if sel:
        col_l, col_r = st.columns([1, 1], gap="large")

        with col_l:
            st.markdown("#### 📝 Claim Details")

            # Status badge via native metric/caption approach
            status_display = STATUS_MAP.get(sel["status"], sel["status"])
            st.markdown(f"**Verdict:** {status_display} &nbsp;&nbsp; **Confidence:** {sel['confidence']}%", unsafe_allow_html=True)
            st.progress(sel["confidence"] / 100)

            st.markdown("**Original Claim:**")
            st.info(f'"{sel["claim"]}"')

            st.markdown("**Verification Reasoning:**")
            reason_text = sel["reason"] if sel["reason"] else "No reasoning returned."
            if sel["status"] == "Verified":
                st.success(reason_text)
            elif sel["status"] in ("Inaccurate", "False"):
                st.warning(reason_text)
            elif sel["status"] == "Needs Review":
                st.error(reason_text)
            else:
                st.info(reason_text)

            if sel["correct_fact"] and sel["status"] in ("Inaccurate", "False"):
                st.markdown("**💡 Suggested Correction:**")
                st.warning(sel["correct_fact"])

        with col_r:
            st.markdown("#### 🌐 Retrieved Web Evidence")
            evidence_list = sel["evidence"]

            if not evidence_list:
                st.warning("No web evidence was retrieved for this claim. This may indicate a network issue or rate-limiting by the search provider.")
            else:
                for idx, ev in enumerate(evidence_list):
                    with st.container(border=True):
                        st.markdown(f"**Source {idx+1}:** {ev.get('title', 'Untitled')}")
                        st.caption(ev.get("snippet", "No description available."))
                        url = ev.get("url", "")
                        if url:
                            st.markdown(f"[🔗 {url}]({url})")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<br><br><div style='text-align:center;color:#94a3b8;font-size:0.82rem;'>"
    "Fact-Check Agent © 2026 · AI-Powered Truth Verification"
    "</div>",
    unsafe_allow_html=True,
)
