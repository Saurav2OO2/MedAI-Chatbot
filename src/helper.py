from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List

def load_pdf_files(data):
    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
    document = loader.load()
    return document

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    '''Given a List of Document objects, return a new List of Documents objects
    Containing only 'Source' in metadata and the original page_content'''

    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get('source')
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={'source': src}
            )
        )
    return minimal_docs

# Split the documents in to smaller chunks
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    text_chunk = text_splitter.split_documents(minimal_docs)
    return text_chunk

# Downloading sentence-transformer model
def download_embeddings():
    '''Download and return the HuggingFace embedding model'''
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    embeddings = HuggingFaceEmbeddings(
        model_name = model_name,
    )
    return embeddings

embeddings = download_embeddings()