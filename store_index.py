import os
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_embeddings


load_dotenv()
PINECONE_API_KEY=os.getenv('PINECONE_API_KEY')
GROQ_API_KEY=os.getenv('GROQ_API_KEY')

os.environ['PINECONE_API_KEY']=PINECONE_API_KEY
os.environ['GROQ_API_KEY']=GROQ_API_KEY

extracted_data = load_pdf_files('data/')
filter_data = filter_to_minimal_docs(extracted_data)
text_chunk = text_split(filter_data)

embeddings = download_embeddings()

pinecone_api_key = PINECONE_API_KEY
pinecone_client = Pinecone(api_key=pinecone_api_key)

index_name = 'medial-chatbot'

if not pinecone_client.has_index(index_name):
    pinecone_client.create_index(
        name = index_name,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

index = pinecone_client.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunk,
    embedding=embeddings,
    index_name=index_name
)