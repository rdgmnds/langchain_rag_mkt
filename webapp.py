from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
import streamlit as st
from main import documents


#VECTORSTORE AND EMBEDDINGS
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
chat = ChatOpenAI(model='gpt-4o')
prompt = ChatPromptTemplate.from_template(
    '''
    Converse como se fosse o Gary Halbert, um dos maiores copywriters da história.
    Saiba que Gary morreu em 2008.
    Fale com o usuário de maneira pessoal, da mesma forma que o Gary escreve em suas cartas.
    Mantenha o mesmo tom, características, humor, ironia e forma de escrever do Gary.
    Responda as perguntas se baseando no contexto fornecido.
    Se não houver dados suficientes para a pergunta e contexto, apenas diga que não sabe.
    Você é um chatbot, não precisa se despedir ao final de cada resposta.

    contexto: {contexto}

    pergunta: {pergunta}
    '''
)

retriever = vectordb.as_retriever(search_type='mmr', search_kwargs={'k': 5, 'fetch_k': 25})

setup = RunnableParallel({
    'pergunta': RunnablePassthrough(),
    'contexto': retriever,
})

def join_documents(input):
    input['contexto'] = '\n\n'.join([c.page_content for c in input['contexto']])
    return input

setup = RunnableParallel({
    'pergunta': RunnablePassthrough(),
    'contexto': retriever,
}) | join_documents

chain = setup | prompt | chat


#------ FRONT-END COM STREAMLIT -------
#############################################################

st.set_page_config(
    page_title="Chat com Gary Halbert", page_icon="✍️"
    )

st.image("arquivos\Gary-Halbert.jpg")
st.title("💬 Converse com a IA do maior guru e galã do marketing de resposta direta!")

if "history" not in st.session_state:
    st.session_state.history = []

pergunta = st.text_input("Faça sua pergunta:", placeholder="Ex: Qual o segredo para uma boa headline?")

if st.button("Enviar") and pergunta:
    with st.spinner("Gary está pensando na resposta..."):
        resposta = chain.invoke(pergunta).content

    # Salva no histórico
    st.session_state.history.append((pergunta, resposta))

# Mostra histórico
for i, (pergunta, resposta) in enumerate(reversed(st.session_state.history)):
    st.markdown(f"**Você:** {pergunta}")
    st.markdown(f"**Gary Halbert:** {resposta}")
    st.markdown("---")
