from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
import streamlit as st
from main import documents


#VECTORDB AND EMBEDDINGS
#############################################################
directory = 'chroma_retrival_bd'
embeddings_model = OpenAIEmbeddings()

vectordb = Chroma.from_documents(
    documents=documents,
    embedding=embeddings_model,
    persist_directory=directory
)


#RETRIEVER
#############################################################
prompt = ChatPromptTemplate.from_template(
    '''
    Converse como se fosse o Gary Halbert, um dos maiores copywriters da história.
    Saiba que Gary morreu em 2008.
    Fale com o usuário de maneira pessoal, da mesma forma que o Gary escreve em suas cartas.
    Mantenha o mesmo tom, características, humor, ironia e forma de escrever do Gary.
    Responda as perguntas se baseando no contexto fornecido.
    Se não houver dados suficientes para a pergunta e contexto, apenas diga que não sabe.
    Você é um chatbot, não precisa se despedir ao final de cada resposta. Você não está escrevendo cartas, mas sim conversando através de um chat.
    Não invente histórias. Não se baseie em informações que não estão disponíveis. Não invente informações.

    contexto: {contexto}

    pergunta: {pergunta}
    '''
)

retriever = vectordb.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 5, 'fetch_k': 25}
    )

def join_documents(input):
    input['contexto'] = '\n\n'.join([c.page_content for c in input['contexto']])
    return input

chain = RunnableParallel({
    'pergunta': RunnablePassthrough(),
    'contexto': retriever,
}) | join_documents | prompt | ChatOpenAI(model='gpt-4o')

MEMORY = ConversationBufferMemory()


#------ FRONT-END COM STREAMLIT -------
#############################################################

def pagina_inicial():
    st.set_page_config(
        page_title="Chat com Gary Halbert", page_icon="✍️"
        )

    st.image("arquivos\Gary-Halbert.jpg")
    st.title("💬 Converse com a IA do maior guru e galã do marketing de resposta direta!")
    st.divider()

    memory = st.session_state.get('memory', MEMORY)

    for message in memory.buffer_as_messages:
        chat = st.chat_message(message.type)
        chat.markdown(message.content)

    user_input = st.chat_input("Fale com o Gary:")

    if user_input:
        chat = st.chat_message('human')
        chat.markdown(user_input)

        chat = st.chat_message('ai', avatar="arquivos\gary-halbert-avatar.png")
        resposta = chat.write_stream(chain.stream(user_input))

        # Salva no histórico
        memory.chat_memory.add_user_message(user_input)
        memory.chat_memory.add_ai_message(resposta)
        st.session_state['memory'] = memory


def main():
    pagina_inicial()

if __name__ == '__main__':
    main()