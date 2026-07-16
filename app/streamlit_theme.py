"""Tema visual con colores celeste, blanco y dorado para la app Streamlit."""

ARGENTINA_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');

:root {
    --arg-celeste: #74acdf;
    --arg-celeste-dark: #2b5f8c;
    --arg-gold: #fcd116;
    --arg-white: #ffffff;
}

.stApp {
    background:
        linear-gradient(
            135deg,
            rgba(116, 172, 223, 0.35) 0%,
            rgba(255, 255, 255, 0.9) 35%,
            rgba(255, 255, 255, 0.9) 65%,
            rgba(116, 172, 223, 0.35) 100%
        ),
        repeating-linear-gradient(
            -45deg,
            rgba(116, 172, 223, 0.06) 0 18px,
            rgba(255, 255, 255, 0.08) 18px 36px
        );
    background-attachment: fixed;
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background:
        radial-gradient(circle at 12% 18%, rgba(252, 209, 22, 0.18) 0%, transparent 22%),
        radial-gradient(circle at 88% 82%, rgba(116, 172, 223, 0.22) 0%, transparent 28%);
    pointer-events: none;
    z-index: 0;
}

.main .block-container {
    position: relative;
    z-index: 1;
    max-width: 52rem;
    background: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(116, 172, 223, 0.35);
    border-radius: 18px;
    box-shadow: 0 10px 40px rgba(43, 95, 140, 0.12);
    padding: 1.5rem 2rem 2rem;
}

h1, h2, h3, p, label, .stMarkdown, .stCaption {
    font-family: "Montserrat", sans-serif !important;
}

h1 {
    color: var(--arg-celeste-dark) !important;
    font-weight: 700 !important;
}

[data-testid="stCaptionContainer"] p {
    color: #3d5a73 !important;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        rgba(116, 172, 223, 0.22) 0%,
        rgba(255, 255, 255, 0.96) 45%,
        rgba(252, 209, 22, 0.08) 100%
    );
    border-right: 2px solid rgba(252, 209, 22, 0.45);
}

section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding-top: 1rem;
}

.stButton > button {
    background: linear-gradient(90deg, var(--arg-celeste-dark), var(--arg-celeste));
    color: white;
    border: none;
    border-radius: 10px;
    font-family: "Montserrat", sans-serif;
    font-weight: 600;
}

.stButton > button:hover {
    border: none;
    color: white;
    background: linear-gradient(90deg, #1f3550, var(--arg-celeste-dark));
}

[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.72);
    border: 1px solid rgba(116, 172, 223, 0.2);
    border-radius: 12px;
}

[data-testid="stChatInput"] {
    border-color: rgba(116, 172, 223, 0.55);
    background: rgba(255, 255, 255, 0.95);
}

[data-testid="stChatInput"]:focus-within {
    border-color: var(--arg-celeste);
    box-shadow: 0 0 0 2px rgba(116, 172, 223, 0.25);
}

#MainMenu, footer, header[data-testid="stHeader"] {
    background: rgba(255, 255, 255, 0.55);
    backdrop-filter: blur(6px);
}
</style>
"""

def inject_argentina_theme() -> None:
    import streamlit as st

    st.markdown(ARGENTINA_THEME_CSS, unsafe_allow_html=True)
