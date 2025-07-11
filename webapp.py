
import streamlit as st
from main import *

def pagina_inicial():
    st.set_page_config(
        page_title="Chat com Gary Halbert", page_icon="✍️"
        )

    st.image("arquivos\Gary-Halbert.jpg")
    st.title("💬 Converse com a IA do maior guru e galã do marketing de resposta direta!")
    st.divider()

    memory_chat = st.session_state.get('memory', memory)

    for message in memory_chat.buffer_as_messages:
        chat = st.chat_message(message.type)
        chat.markdown(message.content)

    user_input = st.chat_input("Fale com o Gary:")

    if user_input:
        chat = st.chat_message('human')
        chat.markdown(user_input)

        chat = st.chat_message('ai', avatar="arquivos\gary-halbert-avatar.png")
        resposta = chat.write_stream(chain.stream(user_input))

        # Salva no histórico
        memory_chat.chat_memory.add_user_message(user_input)
        memory_chat.chat_memory.add_ai_message(resposta)
        st.session_state['memory'] = memory_chat


def main():
    pagina_inicial()

if __name__ == '__main__':
    main()