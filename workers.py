import os
from chromadb import logger
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
import torch

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

conversation_retrieval_chain = None
chat_history = []

def init_llm():
    global llm_hub, embeddings
    
    print("🚀 Initializing Groq LLM and embeddings...")
    
    MODEL_NAME = "llama-3.3-70b-versatile"  # Recommended replacement
    
    model_parameters = {
        "temperature":0.1,
        "max_tokens":256,
    }
    
    llm_hub = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model=MODEL_NAME,
        **model_parameters
    )
    
    print(f"✅ Groq LLM initialized: {MODEL_NAME}")
    
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    embeddings = HuggingFaceEmbeddings(
        model_name= EMBEDDING_MODEL,
        model_kwargs = {"device":"cuda" if torch.cuda.is_available() else "cpu"}
    )
    
    print(f"✅ HuggingFace Embeddings initialized: {EMBEDDING_MODEL}")
    
    return llm_hub, embeddings

# Function to process a PDF document using PYPDFLoader

def process_document(document_path):
    
    global conversation_retrieval_chain
    
    logger.info(f"Loading documents info, {document_path}")
    
    #Load the document using PYPDFLoader
    loader = PyPDFLoader(document_path)
    documents = loader.load()   
    logger.debug(f"Loaded documents length {len(documents)}")
    
    #Split the document into chunks
    text_splitters = RecursiveCharacterTextSplitter(chunk_size=1064, chunk_overlap=160)
    texts = text_splitters.split_documents(documents)
    logger.debug(f"Split document into {len(texts)} chunks")
    
    
    #Create a Chroma vector store from the document chunks
    logger.info("Initializing Chroma vector store from document chunks")
    db = Chroma.from_documents(texts, embedding=embeddings)
    logger.debug("Chroma vector store initialized")
    
    #Optional: Log available collections if availble 
    try:
        collections = db._client.list_collections()
        logger.debug(f"Available collection in Chroma {collections}")
    except Exception as e:
        logger.warning(f"Could not retrieve the collection from Chroma: {e}")
    
    prompt = ChatPromptTemplate.from_template("""
                    You are a helpful assistant analyzing PDF documents.
                    Use ONLY the provided context to answer. Be concise.

                    Context: {context}
                    Question: {input}
                    Answer:""")
    
    
    #Create a simple chain that stuffs document
    stuff_chain = create_stuff_documents_chain(llm=llm_hub, prompt=prompt) 
    
    #Create a full RAG Chain
    conversation_retrieval_chain = create_retrieval_chain(
        retriever = db.as_retriever(search_type="mmr", search_kwargs={'k':6}),
        combine_docs_chain = stuff_chain,
    )
    
    logger.info("RAG Chain created successfully.")
    
def process_prompt(prompt):
    global conversation_retrieval_chain
    global chat_history
    
    logger.info(f"Processing Prompt: {prompt}")
    
    
    if conversation_retrieval_chain is None:
        logger.warning("⚠️ No document processed yet!")
        return "Error: Please upload a PDF document first before asking questions."
    
    #Query the model using the new .invoke() method
    output = conversation_retrieval_chain.invoke({"input": prompt, "chat_history": chat_history})
    answer = output["answer"]
    logger.info(f"Generated answer: {answer}")
    
    #Update the chat history
    chat_history.append((prompt, answer))
    logger.debug(f"Chat history updated. Total exchanges: {len(chat_history)}")
    
    return answer

#Initialize the language model
init_llm()
logger.info("LLM and embeddings initialized successfully completed.")
