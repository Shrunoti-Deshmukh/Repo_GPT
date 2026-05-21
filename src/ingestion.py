import os
import shutil
import stat
import gc
import time

from git import Repo

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser

from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter
)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def remove_readonly(func, path, excinfo):

    os.chmod(path, stat.S_IWRITE)
    func(path)

def repo_injection(repo_url, target_path):

    if os.path.exists(target_path):

        shutil.rmtree(
            target_path,
            onerror=remove_readonly
        )

    os.makedirs(target_path)

    Repo.clone_from(
        repo_url,
        to_path=target_path
    )

    return target_path

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
    ".ipynb": Language.PYTHON,
}

def load_repo(repo_path):

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

        except Exception as e:
            print(f"Skipping {extension}: {e}")

    return all_documents

def text_splitter(documents):

    final_chunks = []

    for doc in documents:

        language_name = doc.metadata.get("language")

        try:

            language_enum = getattr(
                Language,
                language_name
            )

            splitter = RecursiveCharacterTextSplitter.from_language(
                language=language_enum,
                chunk_size=2000,
                chunk_overlap=200
            )

        except Exception:

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200
            )

        chunks = splitter.split_documents([doc])

        final_chunks.extend(chunks)

    return final_chunks

def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def create_vector_db(
    repo_url,
    target_path,
    db_path
):

    repo_path = repo_injection(
        repo_url,
        target_path=target_path
    )

    docs = load_repo(repo_path)

    chunks = text_splitter(docs)

    embeddings = load_embeddings()

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path
    )

    db.persist()

    # Cleanup refs
    del db
    del docs
    del chunks
    del embeddings

    gc.collect()

    time.sleep(1)

    return True