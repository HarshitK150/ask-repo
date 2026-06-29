# AskRepo 💬

Chat with any GitHub repository. Understand codebases without manually reading thousands of lines of code.

## Demo

🚀 **Live app: [ask-repo.streamlit.app](https://ask-repo.streamlit.app)**

> Enter any public GitHub repo URL → Ask questions in natural language → Get answers with source references

## Problem

Developers joining a new project often spend days understanding:

- Project architecture
- Folder structure
- API flows
- Business logic
- Dependencies

AskRepo solves this by letting you chat with the codebase directly.

## How It Works

1. **Ingest** — Fetches the repository as a zip via the GitHub API and extracts relevant source files (`.py`, `.js`, `.ts`, `.tsx`, `.jsx`, `.md`, `.yml`)
2. **Chunk & Embed** — Splits files into chunks using LangChain's `RecursiveCharacterTextSplitter` and embeds them with OpenAI's `text-embedding-3-small`
3. **Retrieve & Answer** — On each question, retrieves the top 5 most relevant chunks and passes them as context to `gpt-4o-mini` via an LCEL chain
4. **Chat History** — Maintains conversation history so follow-up questions work naturally

## Tech Stack

- **LangChain** — Chunking, embeddings, LCEL chain
- **OpenAI** — `gpt-4o-mini` for answers, `text-embedding-3-small` for embeddings
- **ChromaDB** — In-memory vector store (session-isolated)
- **Streamlit** — UI and session management
- **GitHub API** — Repository ingestion

## Getting Started

### Prerequisites

- Python 3.10+
- OpenAI API key
- GitHub personal access token (public_repo scope only)

### Installation

```bash
git clone https://github.com/HarshitK150/ask-repo.git
cd ask-repo
uv install
```

### Environment Variables

Create a `.env` file:

```
OPENAI_API_KEY=sk-...
GITHUB_TOKEN=ghp_...
```

### Run

```bash
uv run streamlit run app.py
```

## Project Structure

```
ask-repo/
├── app.py              # Streamlit UI
├── ingest_repos.py     # GitHub fetch and file extraction
├── vector_store.py     # Chunking, embedding, vector store
├── rag_chain.py        # LCEL RAG chain with chat history
├── .env                # API keys (never commit)
└── pyproject.toml
```

## Limitations & Future Work

- Only public repositories are supported (private repos require broader GitHub token scope)
- Large repositories take 10-30 seconds to index on first load
- A production version would cache embeddings in a persistent vector store (e.g. Pinecone) so the same repo isn't re-indexed per session
- Could be extended with a LangGraph agent for multi-step reasoning (e.g. "compare how auth is handled across these two repos")
