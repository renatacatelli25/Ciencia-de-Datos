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

st.title("🍕 Asistente de Reseñas Amazon")
st.caption(
    "Interfaz pública del proyecto Ciencia de Datos. "
    f"Backend activo: **{backend}**."
)

with st.sidebar:
    st.header("Estado del sistema")
    if backend == "mock":
        st.success("Backend: MOCK")
        st.info("LLM: no conectado")
        st.info("Modelo ML: no conectado")
        st.info("RAG: no conectado")
    elif backend in ("rag", "local"):
        st.success("Backend: RAG (local)")
        st.info("Embeddings: sentence-transformers")
        st.info("LLM: heurístico / CrossEncoder")
        st.success("RAG: conectado")
    elif backend == "api":
        st.success("Backend: RAG (api)")
        st.info("Embeddings: OpenAI")
        st.warning("LLM: OpenAI (requiere API key)")
        st.success("RAG: conectado")
    else:
        st.warning(f"Backend: {backend}")

    st.caption("Cambiá `CHAT_BACKEND` en `.env` (mock | local | api | rag).")
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
