from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


#LOADER
#############################################################
caminhos = [
    "arquivos/Gary Halbert - Big Idea.pdf",
    "arquivos/AIDA.pdf",
    "arquivos/Conheça seu cliente.pdf",
    "arquivos\Anúncio da água.pdf",
    "arquivos\O Lado Negro do Sucesso.pdf",
    "arquivos\Ensinamentos para a vida.pdf",
    "arquivos\Como ganharia dinheiro com um PC.pdf"
]

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
directory = 'arquivos/chroma_retrival_bd'
embeddings_model = OpenAIEmbeddings()

vectordb = Chroma.from_documents(
    documents=documents,
    embedding=embeddings_model,
    persist_directory=directory
)


#RETRIEVER
#############################################################
chat = ChatOpenAI(model='gpt-3.5-turbo-0125')
prompt = ChatPromptTemplate.from_template(
    '''
    Você é um assistente de marketing de resposta direta.
    Seu maior mentor é o Gary Halbert, um dos maiores copywriters da história.
    Responda as perguntas se baseando no contexto fornecido.
    Se não houver dados suficientes para a pergunta e contexto, diga que não há informações suficientes.

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
#chain.invoke('Qual o fator mais importante para criar uma campanha de sucesso, segundo Gary Halbert?')
