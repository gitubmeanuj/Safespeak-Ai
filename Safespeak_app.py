import os
import json
import streamlit as st
from streamlit_mic_recorder import mic_recorder
from google import genai
from google.genai import types

# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = "gemini-2.5-flash" 

# Initialize Gemini client
API_KEY = "YOUR_API_KEY"
if not API_KEY:
    raise RuntimeError(
        "Please set GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
    )

client = genai.Client(api_key=API_KEY)

# -----------------------------
# Shared JSON schema for responses
# -----------------------------
RISK_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "risk_score": {
            "type": "NUMBER",
            "description": "Overall risk between 0 and 100. 0 = totally safe, 100 = extremely risky."
        },
        "risk_level": {
            "type": "STRING",
            "description": "Qualitative risk bucket.",
            "enum": ["low", "medium", "high", "critical"]
        },
        "categories": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "List of categories such as religion, caste, hate_speech, bullying, harassment, explicit_language, threat, misinformation, legal_violation."
        },
        "explanations": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "Short bullet points explaining what triggered the risk."
        },
        "problematic_text": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "List with exact underlined portions of the text that are risky."
        },
        "legal_sections_triggered": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "List of specific laws or policies violated (e.g., 'IT Act Section 66A', 'IPC 153A')."
        },
        "legal_risk_summary": {
            "type": "STRING",
            "description": "Short explanation of what could realistically happen (e.g., account suspension, legal complaints)."
        },
        "suggested_rewrites": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "Respectful alternative messages."
        },
        "tone_analysis": {
            "type": "STRING",
            "description": "Analysis of the speaker's tone (e.g., sarcastic, aggressive, calm)."
        },
        "detected_emotions": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "List of detected emotions (e.g., Anger, Frustration, Joy)."
        },
    },
    "required": ["risk_score", "risk_level", "categories", "explanations", "problematic_text", "legal_sections_triggered", "legal_risk_summary", "suggested_rewrites", "tone_analysis", "detected_emotions"],
}

SYSTEM_INSTRUCTIONS = """
You are an AI ethics and safety assistant for social media, specialized in analyzing content for ethical, social, cultural, and legal risks, particularly in the context of India and multicultural societies.

Your purpose is to:
1. Analyse user text, image text, or audio transcription for risks.
2. Identify violations under relevant national cyber laws, harassment laws, hate-speech regulations, discrimination laws, or social media platform policies.
3. Return exact legal sections that may apply (e.g., 'IT Act Section 66A', 'IPC 153A', 'IPC 295A', 'Anti-Bullying Regulations').
4. Highlight risky words/sentences using `__underlined__` markup.
5. Estimate realistic potential outcomes (account suspension, legal complaints, etc.).
6. Provide polite rewrites.
7. For audio/speech, specifically analyze tone and emotion (e.g., anger, sarcasm, aggression) and flag if the tone itself is harmful even if words are neutral.

Requirements:
- Do not invent new slurs.
- Respect free expression; flag tone only when personally harmful or discriminatory.
- Never exaggerate legal consequences. Only map to widely recognized real sections.
- Keep responses concise, factual, structured, and helpful.
"""

def call_gemini_for_text(text: str) -> dict:
    if not text or not text.strip():
        return None

    prompt = f"""
{SYSTEM_INSTRUCTIONS}

Analyse the following text:
USER_TEXT:
\"\"\"{text.strip()}\"\"\"
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": RISK_SCHEMA,
        },
    )

    return json.loads(response.text)

def call_gemini_for_image(image_bytes: bytes, mime_type: str = "image/png") -> dict:
    if not image_bytes:
        return None

    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type=mime_type,
    )

    prompt = f"""
{SYSTEM_INSTRUCTIONS}

Analyse the content of this image (transcribe text, comments, captions):
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[image_part, prompt],
        config={
            "response_mime_type": "application/json",
            "response_schema": RISK_SCHEMA,
        },
    )

    return json.loads(response.text)

def call_gemini_for_audio(audio_bytes: bytes, mime_type: str = "audio/mp3") -> dict:
    if not audio_bytes:
        return None

    audio_part = types.Part.from_bytes(
        data=audio_bytes,
        mime_type=mime_type,
    )

    prompt = f"""
{SYSTEM_INSTRUCTIONS}

Analyse the audio content (transcribe and analyze):
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[audio_part, prompt],
        config={
            "response_mime_type": "application/json",
            "response_schema": RISK_SCHEMA,
        },
    )

    return json.loads(response.text)

def call_gemini_for_speech(audio_bytes: bytes, mime_type: str = "audio/wav") -> dict:
    if not audio_bytes:
        return None

    # Streamlit audiorecorder returns raw bytes, usually wav
    audio_part = types.Part.from_bytes(
        data=audio_bytes,
        mime_type=mime_type,
    )

    prompt = f"""
{SYSTEM_INSTRUCTIONS}

Analyse the speech in this audio. 
1. Transcribe the speech.
2. Analyze the tone and emotion (anger, sarcasm, etc.).
3. Perform the standard risk and legal analysis.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[audio_part, prompt],
        config={
            "response_mime_type": "application/json",
            "response_schema": RISK_SCHEMA,
        },
    )

    return json.loads(response.text)

def render_risk_box(data: dict):
    if not data:
        st.warning("No analysis result available.")
        return

    risk_score = int(round(data.get("risk_score", 0)))
    risk_level = (data.get("risk_level") or "low").lower()

    level_display = {
        "low": "🟢 Low",
        "medium": "🟡 Medium",
        "high": "🟠 High",
        "critical": "🔴 Critical",
    }.get(risk_level, risk_level)

    st.subheader("Overall Risk Assessment")
    st.metric("Risk score (0–100)", value=risk_score)
    st.progress(min(max(risk_score, 0), 100))
    st.write(f"**Risk Level:** {level_display}")

    categories = data.get("categories") or []
    if categories:
        st.write(f"**Categories:** {', '.join(categories)}")

    legal_sections = data.get("legal_sections_triggered") or []
    if legal_sections:
        st.error(f"**Legal Sections Triggered:** {', '.join(legal_sections)}")
    
    legal_summary = data.get("legal_risk_summary")
    if legal_summary:
        st.warning(f"**Legal Risk Summary:** {legal_summary}")

    problematic = data.get("problematic_text") or []
    if problematic:
        st.write("**Problematic Text:**")
        for p in problematic:
            st.markdown(f"- {p}")

    explanations = data.get("explanations") or []
    if explanations:
        st.write("**Analysis:**")
        for e in explanations:
            st.markdown(f"- {e}")

    rewrites = data.get("suggested_rewrites") or []
    if rewrites:
        st.subheader("Polite & Respectful Suggestions")
        for i, alt in enumerate(rewrites, start=1):
            st.success(f"**Option {i}:** {alt}")
            
    st.divider()
    st.caption("Detailed Analysis")
    
    tone = data.get("tone_analysis")
    if tone:
        st.info(f"**Tone Analysis:** {tone}")
        
    emotions = data.get("detected_emotions") or []
    if emotions:
        st.write(f"**Detected Emotions:** {', '.join(emotions)}")

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="SafeSpeak AI-Ethics & Legal Assistant",
    page_icon="⚖️",
    layout="wide",
)

st.title("SafeSpeak AI-Ethics & Legal Assistant")
st.caption(
    "Analyze text, images, or audio for ethical, social, and legal risks."
)

tab_text, tab_image, tab_audio, tab_live_speech = st.tabs(["Text Analysis", "Image Analysis", "Audio Analysis", "Live Speech Analysis"])

with tab_text:
    st.subheader("Analyze Text")
    user_text = st.text_area("Enter text to analyze:", height=150)
    if st.button("Analyze Text", type="primary"):
        if user_text.strip():
            with st.spinner("Analyzing..."):
                try:
                    result = call_gemini_for_text(user_text)
                    render_risk_box(result)
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter some text.")

with tab_image:
    st.subheader("Analyze Image")
    uploaded_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        if st.button("Analyze Image", type="primary"):
            with st.spinner("Analyzing..."):
                try:
                    image_bytes = uploaded_image.read()
                    mime_type = uploaded_image.type or "image/png"
                    result = call_gemini_for_image(image_bytes, mime_type)
                    render_risk_box(result)
                except Exception as e:
                    st.error(f"Error: {e}")

with tab_audio:
    st.subheader("Analyze Audio")
    uploaded_audio = st.file_uploader("Upload Audio", type=["mp3", "wav", "m4a"])
    if uploaded_audio:
        st.audio(uploaded_audio)
        if st.button("Analyze Audio", type="primary"):
            with st.spinner("Analyzing..."):
                try:
                    audio_bytes = uploaded_audio.read()
                    mime_type = uploaded_audio.type or "audio/mp3"
                    result = call_gemini_for_audio(audio_bytes, mime_type)
                    render_risk_box(result)
                except Exception as e:
                    st.error(f"Error: {e}")

with tab_live_speech:
    st.subheader("🎙️ Real-Time Voice Speech Analysis")
    st.markdown("""
    - **Speak** your comment instead of typing.
    - **Live Analysis** of toxicity, tone, and legal risks.
    - **Instant Feedback** with polite rewrites.
    """)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.write("**Record:**")
        # mic_recorder returns a dict with 'bytes' and 'sample_rate'
        audio_data = mic_recorder(
            start_prompt="Start Recording",
            stop_prompt="Stop Recording",
            just_once=False,
            use_container_width=True,
            format="wav",
            key="recorder"
        )

    with col2:
        st.write("**Playback & Waveform:**")
        if audio_data and audio_data['bytes']:
            st.audio(audio_data['bytes'])

    if audio_data and audio_data['bytes']:
        if st.button("Analyze Speech", type="primary"):
            with st.spinner("Analyzing Speech & Tone..."):
                try:
                    audio_bytes = audio_data['bytes']
                    result = call_gemini_for_speech(audio_bytes, mime_type="audio/wav")
                    render_risk_box(result)
                except Exception as e:
                    st.error(f"Error: {e}")
