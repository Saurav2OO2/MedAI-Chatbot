from flask import Flask, render_template, jsonify, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.getenv('PINECONE_API_KEY')
GROQ_API_KEY=os.getenv('GROQ_API_KEY')

os.environ['PINECONE_API_KEY']=PINECONE_API_KEY
os.environ['GROQ_API_KEY']=GROQ_API_KEY


embeddings = download_embeddings()

index_name = 'medial-chatbot'

docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name
)

retriever = docsearch.as_retriever(search_type='similarity', search_kwargs={'k':3})

# Model
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system_prompt),
        ('human', '{input}')
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

chat_history = []

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/get', methods=['GET', 'POST'])
def chat():
    msg = request.form['msg']
    print(f'User: {msg}')

    # Append user msg
    chat_history.append(('user', msg))

    # Buld convo context
    past_conversation = '\n'.join([f'{speaker}: {text}' for speaker, text in chat_history[-5:]])

    # Feed context into the model
    response = rag_chain.invoke({
        'input': f'The conversation so far: \n{past_conversation}\n User: {msg}'
    })

    answer = response['answer']

    # Append assistant reply to memory
    chat_history.append(('assistant', answer))

    print('Response', answer)
    return str(answer)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)