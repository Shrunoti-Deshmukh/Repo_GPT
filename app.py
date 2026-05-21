import streamlit as st
import os
import shutil
import stat
import uuid
import glob
from dotenv import load_dotenv

from src.ingestion import create_vector_db
from src.llm_chain import create_qa_chain, close_db_connection

# =========================
# ENV VARIABLES
# =========================
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY", "")

# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(
    page_title="GitHub Source Code Expert Chatbot",
    page_icon="💻",
    layout="wide"
)

st.title("💻 GitHub Repository Q&A Chatbot")

st.write(
    "Clone any public GitHub repository, index it into a vector DB, "
    "and chat with the codebase."
)

# =========================
# HELPERS
# =========================
def remove_readonly(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)

BASE_DB_PATH = "./chroma_db"
BASE_REPO_PATH = "./repos"

os.makedirs(BASE_DB_PATH, exist_ok=True)
os.makedirs(BASE_REPO_PATH, exist_ok=True)

def cleanup_old_databases(base_path=BASE_DB_PATH, keep_latest=3):

    dbs = sorted(
        glob.glob(f"{base_path}/*"),
        key=os.path.getmtime,
        reverse=True
    )

    for old_db in dbs[keep_latest:]:
        try:
            shutil.rmtree(old_db, ignore_errors=True)
        except:
            pass

# =========================
# SESSION STATE
# =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "repo_processed" not in st.session_state:
    st.session_state.repo_processed = False

if "db_path" not in st.session_state:
    st.session_state.db_path = None

if "repo_path" not in st.session_state:
    st.session_state.repo_path = None

# Cleanup old DBs automatically
cleanup_old_databases()

# =========================
# SIDEBAR
# =========================
st.sidebar.header("🔧 Configuration")

repo_url = st.sidebar.text_input(
    "GitHub Repository URL",
    placeholder="https://github.com/user/repo"
)

# =========================
# PROCESS REPO
# =========================
if st.sidebar.button("⚙️ Process & Index Repository"):

    if not groq_api_key:
        st.sidebar.error("Missing GROQ_API_KEY")
    
    elif not repo_url:
        st.sidebar.error("Please enter a repository URL")

    else:

        with st.sidebar.status(
            "Processing repository...",
            expanded=True
        ) as status:

            try:

                # Close previous chain
                if "qa_chain" in st.session_state:
                    close_db_connection(
                        st.session_state.qa_chain
                    )

                    del st.session_state.qa_chain

                # Create unique repo/db paths
                unique_id = str(uuid.uuid4())

                DB_PATH = os.path.join(
                    BASE_DB_PATH,
                    unique_id
                )

                REPO_PATH = os.path.join(
                    BASE_REPO_PATH,
                    unique_id
                )

                st.session_state.db_path = DB_PATH
                st.session_state.repo_path = REPO_PATH

                st.write("📥 Cloning repository...")

                create_vector_db(
                    repo_url=repo_url,
                    target_path=REPO_PATH,
                    db_path=DB_PATH
                )

                st.write("🧠 Creating QA chain...")

                st.session_state.qa_chain = create_qa_chain(
                    groq_api_key,
                    db_path=DB_PATH
                )

                st.session_state.chat_history = []
                st.session_state.repo_processed = True

                status.update(
                    label="✅ Ready",
                    state="complete",
                    expanded=False
                )

                st.sidebar.success("Repository indexed successfully!")

            except Exception as e:

                status.update(
                    label="❌ Failed",
                    state="error",
                    expanded=True
                )

                st.sidebar.error(f"Error logs: {str(e)}")

# =========================
# DELETE CURRENT REPO
# =========================
if st.sidebar.button("🗑️ Remove Current Repository"):

    try:

        # Close active chain
        if "qa_chain" in st.session_state:

            close_db_connection(
                st.session_state.qa_chain
            )

            del st.session_state.qa_chain

        # Delete repo folder ONLY
        repo_path = st.session_state.get("repo_path")

        if repo_path and os.path.exists(repo_path):
            shutil.rmtree(
                repo_path,
                onerror=remove_readonly
            )

        # DO NOT DELETE ACTIVE CHROMA DB
        # Windows locks mmap files internally

        st.session_state.repo_processed = False
        st.session_state.chat_history = []

        st.sidebar.success(
            "Repository removed successfully!"
        )

        st.rerun()

    except Exception as e:
        st.sidebar.error(str(e))

# =========================
# CLEAR CHAT
# =========================
if st.sidebar.button("🧹 Clear Chat History"):

    st.session_state.chat_history = []

    if (
        "qa_chain" in st.session_state
        and st.session_state.db_path
    ):

        close_db_connection(
            st.session_state.qa_chain
        )

        st.session_state.qa_chain = create_qa_chain(
            groq_api_key,
            db_path=st.session_state.db_path
        )

    st.rerun()

# =========================
# CHAT UI
# =========================
if st.session_state.repo_processed:

    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input(
        "Ask questions about the repository..."
    ):

        with st.chat_message("user"):
            st.markdown(user_query)

        st.session_state.chat_history.append({
            "role": "user",
            "content": user_query
        })

        with st.chat_message("assistant"):

            with st.spinner("Analyzing repository..."):

                try:

                    response = st.session_state.qa_chain({
                        "question": user_query
                    })

                    answer = response.get(
                        "answer",
                        "No answer found."
                    )

                    st.markdown(answer)

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:
                    st.error(str(e))

else:

    st.info(
        "👈 Enter a GitHub repository URL "
        "and click Process & Index Repository."
    )