import streamlit as st

from chat_service import ChatService

st.set_page_config(
    page_title="IA de Reseñas Amazon",
    page_icon="🍕",
    layout="centered",
)

st.title("🍕 Asistente de Reseñas Amazon")
st.caption(
    "Interfaz pública del proyecto Ciencia de Datos. "
    "Por ahora usa un backend simulado (MockProvider)."
)

with st.sidebar:
    st.header("Estado del sistema")
    st.success("Backend: MOCK")
    st.info("LLM: no conectado")
    st.info("Modelo ML: no conectado")
    st.info("RAG: no conectado")
    if st.button("Reiniciar conversación"):
        st.session_state.messages = []
        st.rerun()

if "chat_service" not in st.session_state:
    st.session_state.chat_service = ChatService()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "¡Hola! Soy el asistente del proyecto. "
                "Todavía estoy en modo **mock**, pero ya podés probar la interfaz.\n\n"
                "Ejemplos:\n"
                "- ¿Cuál es el score promedio?\n"
                "- ¿Qué productos tienen más reseñas?\n"
                "- ¿Cómo se comporta el helpfulness?"
            ),
        }
    ]

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
