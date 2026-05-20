import os
from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationSummaryMemory
from langchain_community.vectorstores import Chroma
from src.ingestion import load_embeddings

def initialize_llm(api_key, model_name="llama-3.1-8b-instant"):
    """Initializes the ChatGroq LLM framework."""
    return ChatGroq(
        model=model_name,
        temperature=0,  # Keep it at 0 for factual accuracy
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=api_key
    )

def create_qa_chain(api_key, db_path="./chroma_db"):
    """Creates the ConversationalRetrievalChain using ChromaDB and ChatGroq."""
    llm = initialize_llm(api_key)
    embeddings = load_embeddings()
    
    # Load the existing persisted database
    vocab_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    # Setup conversation summary memory
    memory = ConversationSummaryMemory(
        llm=llm, 
        memory_key="chat_history", 
        return_messages=True
    )
    
    # Build conversational QA chain (Using MMR for diversified search)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vocab_db.as_retriever(search_type="mmr", search_kwargs={"k": 8}),
        memory=memory
    )
    return qa_chain

def close_db_connection(qa_chain):
    """
    Forces the underlying ChromaDB client to close completely, 
    releasing all SQLite file locks.
    """
    try:
        if qa_chain and hasattr(qa_chain, 'retriever') and hasattr(qa_chain.retriever, 'vectorstore'):
            db = qa_chain.retriever.vectorstore
            
            # 1. Check if the client has an explicit close method
            if hasattr(db, '_client') and hasattr(db._client, 'close'):
                db._client.close()
                
            # 2. Break LangChain's internal pointers
            qa_chain.retriever.vectorstore = None
    except Exception as e:
        print(f"Silent database close error: {e}")