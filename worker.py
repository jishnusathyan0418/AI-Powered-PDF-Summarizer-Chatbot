import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

conversation_retrieval_chain = None
chat_history = []

def init_llm():
    global llm_hub, embeddings
    
    print("Initializing Groq LLM and Embeddings......")
    
    MODEL_NAME = "llama-3.3-70b-versatile"  # Recommended replacement
    
    model_parameters = {
        "temperature":0.1,
        "max_tokens": 256,
    }
    
    llm_hub = ChatGroq(
        api_key=GROQ_API_KEY,
        model=MODEL_NAME,
        **model_parameters
    )
    
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    embeddings = HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL,
        model_kwargs = {"device":"cuda" if torch.cuda.is_available() else "cpu"}
    )
    
    return llm_hub, embeddings

# Function to process a PDF document using PYPDFLoader

def process_document(document_path):
    global conversation_retrieval_chain
    
    #load the document
    loader = PyPDFLoader(document_path)
    documents = loader.load()
    
    ##pslit the documents into chunks
    text_splitters = RecursiveCharacterTextSplitter(chunk_size=1064, chunk_overlap=160)
    texts = text_splitters.split_documents(documents)
    
    #vectorize the documents using Chroma vector store
    db = Chroma.from_documents(texts, embedding=embeddings)
    
    prompt = ChatPromptTemplate.from_template("""
                    You are a helpful assistant analyzing PDF documents.
                    Use ONLY the provided context to answer. Be concise.

                    Context: {context}
                    Question: {input}
                    Answer:""")
    
    #stuff chain
    stuff_chain = create_stuff_documents_chain(llm=llm_hub, prompt=prompt)
    
    #create RAG Chain
    conversation_retrieval_chain = create_retrieval_chain(
        retriever = db.as_retriever(search_type='mmr', search_kwargs={"k":6}),
        combine_docs_chain = stuff_chain,
    )
    
def process_prompt(prompt):
    global conversation_retrieval_chain
    global chat_history
    
    #query the model using the new .invoke() method
    
    output = conversation_retrieval_chain.invoke({"input": prompt, "chat_history":chat_history})
    answer = output["answer"]
    
    #update chat_history
    
    chat_history.append((prompt, answer))
    return answer

init_llm()

    
    
    
    
    
    
    
    
    
    
    
    
