import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

#Load the Envs
from dotenv import load_dotenv
load_dotenv()

# Load the GROQ API Key
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY') # type: ignore

# Import LLM
groq_api_key = os.getenv('GROQ_API_KEY')
llm = ChatGroq(groq_api_key = groq_api_key, model = 'gemma2-9b-it') # type: ignore

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question : {input}
    
    """
)

def create_vector_embeddings():
    if 'vectors' in st.session_state:
        return
    
    st.session_state.embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
    st.session_state.loader = PyPDFDirectoryLoader('Research_Papers')

    st.session_state.docs = st.session_state.loader.load()
    if not st.session_state.docs:
        st.error("No PDF documents found. Please check the folder path.")
        return

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])

    if not st.session_state.final_documents:
        st.error("Text splitting failed. No documents to embed.")
        return

    try:
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings
        )
        st.success("Vector database created successfully.")
    except Exception as e:
        st.error(f"Error creating vector database: {e}")
        
user_prompt = st.text_input('Enter your query from reseach paper')

if st.button('Document Embedding'):
    create_vector_embeddings()
    
import time

if user_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever, document_chain)
    
    start = time.process_time()
    response = retriever_chain.invoke({'input' : user_prompt})
    print(f'Response time : {time.process_time() - start}')
    
    st.write(response['answer'])
    
    # With a streamlit expander
    with st.expander('Document Similarity Search'):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('-----------------')