from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
import streamlit as st
import os


#LOADER
#############################################################
pasta_arquivos = "arquivos"
caminhos = [os.path.join(pasta_arquivos, f) for f in os.listdir(pasta_arquivos) if f.endswith(".pdf")]

paginas = []

for caminho in caminhos:
    loader = PyPDFLoader(caminho)
    paginas.extend(loader.load())


#SPLIT
#############################################################
recur_split = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""]
)

documents = recur_split.split_documents(paginas)

for i, doc in enumerate(documents):
    doc.metadata['source'] = doc.metadata['source'].replace('arquivos/', "")
    doc.metadata['doc_id'] = i


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
    Mantenha o mesmo tom, características, humor, ironia do Gary.
    Responda as perguntas se baseando no contexto fornecido.
    Se não houver dados suficientes para a pergunta e contexto, diga que não sabe.
    Você é um chatbot, não precisa se despedir ao final de cada resposta.

    contexto: {contexto}

    pergunta: {pergunta}
    '''
)

retriever = vectordb.as_retriever(search_type='mmr', search_kwargs={'k': 5, 'fetch_k': 25})

setup = RunnableParallel({
    'pergunta': RunnablePassthrough(),
    'contexto': retriever
})

def join_documents(input):
    input['contexto'] = '\n\n'.join([c.page_content for c in input['contexto']])
    return input

setup = RunnableParallel({
    'pergunta': RunnablePassthrough(),
    'contexto': retriever
}) | join_documents

chain = setup | prompt | chat


#WEB APP COM STREAMLIT
#############################################################

st.set_page_config(
    page_title="Chat com Gary Halbert", page_icon="✍️"
    )

st.image("arquivos\Gary-Halbert.jpg")
st.title("💬 Converse com o maior guru e galã do marketing de resposta direta!")

if "history" not in st.session_state:
    st.session_state.history = []

pergunta = st.text_input("Faça sua pergunta:", placeholder="Ex: Qual o segredo para uma boa headline?")

if st.button("Enviar") and pergunta:
    with st.spinner("Consultando Gary Halbert..."):
        resposta = chain.invoke(pergunta).content

    # Salva no histórico
    st.session_state.history.append((pergunta, resposta))

# Mostra histórico
for i, (pergunta, resposta) in enumerate(reversed(st.session_state.history)):
    st.markdown(f"**Você:** {pergunta}")
    st.markdown(f"**Gary IA:** {resposta}")
    st.markdown("---")
