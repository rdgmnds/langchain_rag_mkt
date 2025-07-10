from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
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


