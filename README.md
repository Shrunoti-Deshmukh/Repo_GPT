💻 GitHub Repository Q&A Chatbot (RAG System)

An AI-powered system that allows you to enter any GitHub repository URL and chat with the codebase using Retrieval-Augmented Generation (RAG).
It indexes source code into embeddings using ChromaDB and enables natural language querying via a conversational LLM.

🚀 Features

🔗 Input any public GitHub repository URL

📦 Automatic repo cloning & parsing

🧠 Code-aware chunking (language-specific)

🔎 Vector search using ChromaDB

🤖 Conversational AI powered by Groq LLM

💬 Chat history memory support

🔄 Supports multiple repositories (unique DB per run)

🧹 Cleanup utilities for safe reruns

⚡ Streamlit UI for interactive experience

![Alt text](mermaid-diagram.png)

⚠️ Known Limitations
Large repositories may take time to embed
Windows may still impose file locking constraints
No incremental indexing yet
No AST-level parsing (future improvement)

🔮 Future Improvements
⚡ Incremental indexing (only changed files)
🌳 AST-based parsing (Tree-sitter)
🔍 Hybrid search (BM25 + Vector)
📊 Repo analytics dashboard
🤖 Multi-agent code reviewer system
☁️ Cloud deployment (Docker + AWS)
