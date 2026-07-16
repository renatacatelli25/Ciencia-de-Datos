import os

import streamlit as st
from dotenv import load_dotenv

from chat_service import ChatService

load_dotenv()

st.set_page_config(
    page_title="IA de Reseñas Amazon",
    page_icon="🍕",
    layout="centered",
)

backend = os.getenv("CHAT_BACKEND", "mock").lower()

title_col, status_col = st.columns([11, 1])
with title_col:
    st.title("🍕 Asistente de Reseñas Amazon")
with status_col:
    if backend == "api":
        st.markdown(
            '<p title="OpenAI API activa" style="margin:1.4rem 0 0;text-align:right;'
            'font-size:1.1rem;line-height:1;color:#22c55e;">●</p>',
            unsafe_allow_html=True,
        )

st.caption("Interfaz pública del proyecto Ciencia de Datos.")

with st.sidebar:
    if st.button("Reiniciar conversación"):
        st.session_state.messages = []
        st.rerun()

if "chat_service" not in st.session_state:
    st.session_state.chat_service = ChatService()

if "messages" not in st.session_state:
    if backend == "mock":
        welcome = (
            "¡Hola! Soy el asistente del proyecto. "
            "Todavía estoy en modo **mock**, pero ya podés probar la interfaz.\n\n"
            "Ejemplos:\n"
            "- ¿Cuál es el score promedio?\n"
            "- ¿Qué productos tienen más reseñas?\n"
            "- ¿Cómo se comporta el helpfulness?"
        )
    else:
        welcome = (
            "¡Hola! Soy el asistente de productos sobre reseñas de Amazon.\n\n"
            "Podés preguntarme por productos, marcas, precios o características. "
            "Ejemplo: *¿Hay snacks saludables sin gluten?*"
        )

    st.session_state.messages = [{"role": "assistant", "content": welcome}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Escribí tu pregunta..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
                if m["role"] in ("user", "assistant")
            ]
            answer = st.session_state.chat_service.reply(prompt, history)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
