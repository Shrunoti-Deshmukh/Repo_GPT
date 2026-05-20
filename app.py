import streamlit as st
import os
import shutil
import stat
from dotenv import load_dotenv
from src.ingestion import create_vector_db
from src.llm_chain import create_qa_chain

# Load environment variables from .env file if it exists
load_dotenv()

def remove_readonly(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)
    
# Streamlit App Configurations
st.set_page_config(page_title="GitHub Source Code Expert Chatbot", page_icon="💻", layout="wide")

st.title("💻 GitHub Repository Q&A Chatbot")
st.write("Clone any public GitHub Python repository, index its code structure into an embedding DB, and chat with it in real-time.")

# ----------------- SIDEBAR CONFIGURATIONS -----------------
st.sidebar.header("🔧 Configuration Setup")

# API Key input (falls back to system environmental variable if present)
groq_api_key = st.sidebar.text_input(
    "Groq API Key", 
    value=os.getenv("GROQ_API_KEY", ""), 
    type="password",
    help="Grab your API Key from the Groq Console."
)

# Repo Input
repo_url = st.sidebar.text_input(
    "GitHub Repository URL", 
    placeholder="https://github.com/username/repository-name"
)

DB_PATH = "./chroma_db"
REPO_PATH = "test_repo1"

# Button Trigger for Repository Processing
if st.sidebar.button("⚙️ Process & Index Repository"):
    if not groq_api_key:
        st.sidebar.error("Missing Groq API Key! Please enter it to continue.")
    elif not repo_url:
        st.sidebar.error("Missing Repository URL!")
    else:
        with st.sidebar.status("Processing codebase architecture...", expanded=True) as status:
            try:
                # Clean slate processing to prevent database stacking issues
                if os.path.exists(DB_PATH):
                    shutil.rmtree(DB_PATH, onerror=remove_readonly) # <-- Updated
                if os.path.exists(REPO_PATH):
                    shutil.rmtree(REPO_PATH, onerror=remove_readonly) # <-- Updated
                
                st.write("📥 Cloning remote repository structure...")
                create_vector_db(repo_url, target_path=REPO_PATH, db_path=DB_PATH)
                
                st.write("🧠 Constructing Conversational RAG Engine...")
                st.session_state.qa_chain = create_qa_chain(groq_api_key, db_path=DB_PATH)
                st.session_state.chat_history = []
                st.session_state.repo_processed = True
                
                status.update(label="✅ Ready! Ask your questions.", state="complete", expanded=False)
                st.sidebar.success("Repository completely indexed!")
            except Exception as e:
                status.update(label="❌ Ingestion Failed!", state="error", expanded=True)
                st.sidebar.error(f"Error logs: {str(e)}")

# Clear History Operations
if st.sidebar.button("🧹 Clear Chat History"):
    st.session_state.chat_history = []
    if "qa_chain" in st.session_state and groq_api_key:
        # Reset chain memory buffers completely
        st.session_state.qa_chain = create_qa_chain(groq_api_key, db_path=DB_PATH)
    st.rerun()

# ----------------- CHAT STATES INTERFACE -----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "repo_processed" not in st.session_state:
    st.session_state.repo_processed = False

# Auto-reconnect if DB already exists locally and API Key is initialized
if not st.session_state.repo_processed and os.path.exists(DB_PATH) and groq_api_key:
    try:
        st.session_state.qa_chain = create_qa_chain(groq_api_key, db_path=DB_PATH)
        st.session_state.repo_processed = True
    except Exception:
        pass

# Render interface viewports
if st.session_state.repo_processed:
    # Render historical logs using Streamlit's chat elements
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Capturing New User Input
    if user_query := st.chat_input("Ask about functions, classes, dependencies, or logic flow..."):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # Process through model chain pipeline
        with st.chat_message("assistant"):
            with st.spinner("Analyzing code vectors..."):
                try:
                    response = st.session_state.qa_chain({"question": user_query})
                    answer = response.get("answer", "No viable resolution found inside document embeddings.")
                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"An exception occurred during lookup execution: {str(e)}")
else:
    st.info("👈 Fill out your **Groq API Key** and **GitHub Repository URL** inside the sidebar panel and click **Process & Index Repository** to start exploring code bases.")