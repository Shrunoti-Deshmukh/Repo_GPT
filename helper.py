import os
from git import Repo
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

LANGUAGE_EXTENSIONS = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".ts": Language.TS,
    ".java": Language.JAVA,
    ".cpp": Language.CPP,
    ".c": Language.C,
    ".cs": Language.CSHARP,
    ".go": Language.GO,
    ".rb": Language.RUBY,
    ".php": Language.PHP,
    ".rs": Language.RUST,
    ".kt": Language.KOTLIN,
    ".scala": Language.SCALA,
    ".swift": Language.SWIFT,
    ".html": Language.HTML,
    ".sol": Language.SOL,
    ".lua": Language.LUA,
}

def repo_injection(repo_url, target_path="test_repo1"):
    """Clones a remote GitHub repository to a local path using GitPython."""
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    Repo.clone_from(repo_url, to_path=target_path)
    return target_path

def load_repo(repo_path):
    """
    Load all supported source code files from repository.
    """

    all_documents = []

    for extension, language in LANGUAGE_EXTENSIONS.items():

        try:
            loader = GenericLoader.from_filesystem(
                repo_path,
                glob="**/*",
                suffixes=[extension],
                parser=LanguageParser(
                    language=language,
                    parser_threshold=500
                )
            )

            documents = loader.load()

            for doc in documents:
                doc.metadata["language"] = language.name

            all_documents.extend(documents)

            print(f"Loaded {len(documents)} files for {extension}")

        except Exception as e:
            print(f"Skipping {extension}: {e}")

    return all_documents

def text_splitter(documents):

    final_chunks = []

    for doc in documents:

        language_name = doc.metadata.get("language")

        try:
            language_enum = getattr(Language, language_name)

            splitter = RecursiveCharacterTextSplitter.from_language(
                language=language_enum,
                chunk_size=2000,
                chunk_overlap=200
            )

        except Exception:

            # Fallback splitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200
            )

        chunks = splitter.split_documents([doc])
        final_chunks.extend(chunks)

    return final_chunks

def load_embeddings():
    """Initializes the OpenAI embedding model framework."""
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

def create_vector_db(repo_url):
    # Step 1: Inject / Clone Repo
    repo_path = repo_injection(repo_url)
    print("# Step 1: Inject / Clone Repo")

    # Step 2: Extract & Structural Parsing
    docs = load_repo(repo_path)
    print("# Step 2: Extract & Structural Parsing")

    # Step 3: Context-aware Chunking
    chunks = text_splitter(docs)
    print("# Step 3: Context-aware Chunking")
    
    # Step 4: Get Embeddings Model
    embeddings = load_embeddings()
    print("# Step 4: Get Embeddings Model")
    
    # Step 5: Save & Persist Chunks in Chroma DB
    print("DB started")
    persist_directory = "DB"
    db = Chroma.from_documents(
    documents = chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
    )
    db.persist()
    print("Vector database created and persisted successfully!")