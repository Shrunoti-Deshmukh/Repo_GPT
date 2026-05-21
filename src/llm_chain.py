import gc
import time

from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationSummaryMemory
from langchain_community.vectorstores import Chroma

from src.ingestion import load_embeddings

def initialize_llm(
    api_key,
    model_name="llama-3.1-8b-instant"
):

    return ChatGroq(
        model=model_name,
        temperature=0,
        api_key=api_key
    )

def create_qa_chain(api_key, db_path):

    llm = initialize_llm(api_key)

    embeddings = load_embeddings()

    vector_db = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )

    memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8}
        ),
        memory=memory
    )

    return qa_chain

def close_db_connection(qa_chain):

    try:

        if qa_chain:

            if hasattr(qa_chain, "retriever"):

                retriever = qa_chain.retriever

                if hasattr(retriever, "vectorstore"):

                    vectorstore = retriever.vectorstore

                    retriever.vectorstore = None

                    del vectorstore

            del qa_chain

    except Exception as e:
        print(f"Cleanup warning: {e}")

    finally:

        gc.collect()

        time.sleep(1)