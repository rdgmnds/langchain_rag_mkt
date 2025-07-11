from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
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

    pergunta: {question}
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
    'question': RunnablePassthrough(),
    'contexto': retriever,
}) | join_documents | prompt | ChatOpenAI(model='gpt-4o')

memory = ConversationBufferMemory()