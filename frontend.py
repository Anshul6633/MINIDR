# app.py
import os
import base64
from typing import List, Dict, Any

import streamlit as st
from groq import Groq
from dotenv import load_dotenv

# ---------- Config ----------
load_dotenv()
GROQ_API_KEY= "replace ke"
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is missing. Create a .env file (see .env.example) and restart.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

APP_TITLE = "MiniDr — AI Health Assistant"
APP_TAGLINE = "General chat (Llama), medical chat (Gemma/Llama), and vision (Llama 4 Scout/Maverick)."
DISCLAIMER = (
    "⚠️ MiniDr is **not a medical device** and does not provide diagnosis or treatment. "
    "For emergencies or concerning symptoms, contact a qualified healthcare professional."
)

# ---------- Model Catalog (current Groq docs & fallbacks) ----------
MODELS = {
    "general_text_primary": "llama-3.1-8b-instant",  # fast, general chat
    "general_text_alt": "llama-3.3-70b-versatile",   # higher quality, slower

    # Medical: Gemma 2 has a deprecation notice (Oct 8, 2025). We keep it as best-effort and fallback.
    "medical_primary": "gemma2-9b-it",               # will fallback if unavailable
    "medical_fallback": "llama-3.1-8b-instant",      # recommended replacement in Groq deprecations

    # Vision: per Groq vision docs (preview models)
    "vision_primary": "meta-llama/llama-4-scout-17b-16e-instruct",
    "vision_alt": "meta-llama/llama-4-maverick-17b-128e-instruct",
}

# ---------- Helpers ----------
def infer_language_directive(user_text: str) -> str:
    """
    Heuristic: tell the model to reply in the same language style (Hindi/English/Hinglish).
    We avoid heavyweight language detection; LLM handles multilingual well.
    """
    # Simple hinting based on Devanagari chars
    has_devanagari = any('\u0900' <= ch <= '\u097F' for ch in user_text)
    if has_devanagari:
        return "User may be writing in Hindi or Hinglish. Reply in the same language and script as the user."
    # Hinglish is often English letters with Hindi words; just say 'same language'
    return "Reply in the user's language and tone (English or Hinglish)."

def stream_or_return_text(model: str, messages: List[Dict[str, Any]], temperature: float = 0.2) -> str:
    """Call Groq Chat Completions and return text content."""
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=1024,
            stream=False,
        )
        return completion.choices[0].message.content
    except Exception as e:
        # simple model fallback logic for medical + vision + general
        if model == MODELS["medical_primary"]:
            # fall back to recommended model
            try:
                completion = client.chat.completions.create(
                    model=MODELS["medical_fallback"],
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=1024,
                    stream=False,
                )
                return ("[FYI: Gemma2 unavailable; using Llama 3.1 8B fallback]\n\n"
                        + completion.choices[0].message.content)
            except Exception as e2:
                return f"Error calling Groq (fallback failed): {e2}"
        elif model == MODELS["vision_primary"]:
            # try alternate vision model
            try:
                completion = client.chat.completions.create(
                    model=MODELS["vision_alt"],
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=1024,
                    stream=False,
                )
                return ("[FYI: Scout unavailable; using Maverick vision fallback]\n\n"
                        + completion.choices[0].message.content)
            except Exception as e2:
                return f"Error calling Groq vision (fallback failed): {e2}"
        elif model == MODELS["general_text_primary"]:
            try:
                completion = client.chat.completions.create(
                    model=MODELS["general_text_alt"],
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=1024,
                    stream=False,
                )
                return ("[FYI: 8B instant unavailable; using 70B versatile fallback]\n\n"
                        + completion.choices[0].message.content)
            except Exception as e2:
                return f"Error calling Groq general (fallback failed): {e2}"
        else:
            return f"Error calling Groq: {e}"

def encode_image_to_data_url(file) -> str:
    """Encode uploaded image as base64 data URL."""
    mime = file.type or "image/jpeg"
    b64 = base64.b64encode(file.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

# ---------- Streamlit UI ----------
st.set_page_config(page_title="MiniDr", page_icon="🩺", layout="wide")
st.title(APP_TITLE)
st.caption(APP_TAGLINE)
st.info(DISCLAIMER)

with st.sidebar:
    st.header("⚙️ Settings")
    st.write("Pick how MiniDr routes your prompts:")
    routing_mode = st.radio(
        "Routing",
        options=["Auto (recommended)", "Force General (Llama)", "Force Medical (Gemma/Llama)"],
        index=0,
    )
    default_general_model = st.selectbox(
        "General model",
        options=[MODELS["general_text_primary"], MODELS["general_text_alt"]],
        index=0,
        help="Primary = fast; Alternate = higher quality."
    )
    default_vision_model = st.selectbox(
        "Vision model",
        options=[MODELS["vision_primary"], MODELS["vision_alt"]],
        index=0,
        help="Both are preview per Groq Vision docs."
    )
    st.divider()
    st.write("Tip: MiniDr will answer in Hindi/English/Hinglish matching your input.")

# Tabs for Chat & Vision
tab_chat, tab_vision = st.tabs(["💬 Chat", "🖼️ Vision (CT/X-ray/etc.)"])

# ---------- CHAT TAB ----------
with tab_chat:
    st.subheader("Chat with MiniDr")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # System prompts (kept concise; model handles multilingual)
    BASE_SYSTEM = (
        "You are MiniDr, a polite AI Health Assistant. "
        "Always include a brief safety note when medical advice is requested. "
        "Never claim to diagnose. Keep answers clear and actionable."
    )

    MEDICAL_SYSTEM = (
        BASE_SYSTEM
        + " If the user asks a medical question, structure answers with: 'What it could be', "
          "'What to do now', and 'When to seek care'. Prefer Indian context (₹, mg, common brands) when relevant."
    )

    user_input = st.chat_input("Type in Hindi, English, or Hinglish…")

    # Render history
    for role, content in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(content)

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append(("user", user_input))

        # Routing
        user_lower = user_input.lower()
        looks_medical = any(
            kw in user_lower
            for kw in [
                "symptom", "doctor", "pain", "fever", "bp", "diabetes", "sugar",
                "tablet", "medicine", "dose", "treatment", "allergy", "side effect",
                "ct", "x-ray", "mri", "scan", "infection", "cough", "cold", "flu",
                "throat", "stomach", "vomit", "dizziness", "period", "mens", "pregnancy",
                "bp", "bp high", "bp low", "cholesterol", "asthma", "covid"
            ]
        )

        if routing_mode == "Force General (Llama)":
            target_model = default_general_model
            system_msg = BASE_SYSTEM
        elif routing_mode == "Force Medical (Gemma/Llama)":
            target_model = MODELS["medical_primary"]
            system_msg = MEDICAL_SYSTEM
        else:
            # Auto
            if looks_medical:
                target_model = MODELS["medical_primary"]
                system_msg = MEDICAL_SYSTEM
            else:
                target_model = default_general_model
                system_msg = BASE_SYSTEM

        lang_directive = infer_language_directive(user_input)

        messages = [
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": (
                    f"{user_input}\n\nInstruction: {lang_directive} "
                    "Keep responses concise unless I ask for more detail."
                ),
            },
        ]

        with st.chat_message("assistant"):
            reply = stream_or_return_text(target_model, messages, temperature=0.2)
            st.markdown(reply)
        st.session_state.chat_history.append(("assistant", reply))

# ---------- VISION TAB ----------
with tab_vision:
    st.subheader("Analyze a medical image (e.g., CT, X-ray, skin photo)")
    st.caption("Use JPG/PNG. The model will describe features and suggest next steps. **Not diagnostic**.")

    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    vision_prompt = st.text_area(
        "Prompt",
        value="Explain key findings and possible differentials in plain language. "
              "Highlight red flags. Reply in the same language as my prompt.",
        height=100,
    )
    analyze_btn = st.button("Analyze Image", type="primary", disabled=uploaded is None)

    if analyze_btn and uploaded:
        data_url = encode_image_to_data_url(uploaded)
        # Keep user prompt language hint
        lang_directive = infer_language_directive(vision_prompt)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": (vision_prompt + "\n\n" + lang_directive +
                              "\nIMPORTANT: You are not a doctor. "
                              "Do not give a diagnosis. Provide observations and safety guidance only.")
                     },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]

        # Try primary vision model; if fails, fallback logic inside helper will try alt.
        reply = stream_or_return_text(
            model=default_vision_model,
            messages=messages,
            temperature=0.1
        )

        st.markdown("### Result")
        st.markdown(reply)
        st.markdown(
            "> ⚠️ This is an AI-generated observation and **not** a medical diagnosis. "
            "Consult a licensed clinician for interpretation."
        )

# ---------- Footer ----------
st.divider()
st.caption("MiniDr © 2025 • Built on Groq • Streamlit UI")