import streamlit as st
from ingest_repos import get_repo_files
from vector_store import build_vectorstore
from rag_chain import build_qa_chain

st.set_page_config(page_title="RepoChat", page_icon="💬")
st.title("💬 RepoChat")
st.caption("Chat with any GitHub repository")

# --- Sidebar: repo loader ---
with st.sidebar:
    st.header("Load a Repository")
    repo_url = st.text_input("GitHub URL", placeholder="https://github.com/owner/repo")
    load_btn = st.button("Load Repo")

    if load_btn and repo_url:
        try:
            parts = repo_url.strip("/").split("/")
            owner, repo = parts[-2], parts[-1]

            with st.spinner(f"Fetching {owner}/{repo}..."):
                files = get_repo_files(owner, repo)
            st.success(f"✅ Fetched {len(files)} files")

            with st.spinner("Building vector store..."):
                vs = build_vectorstore(files)
            
            chain, retriever = build_qa_chain(vs)

            st.session_state.chain = chain
            st.session_state.retriever = retriever
            st.session_state.repo_name = f"{owner}/{repo}"
            st.session_state.messages = []
            st.success("Ready! Ask anything below.")

        except Exception as e:
            st.error(f"Error: {e}")

# --- Main: chat interface ---
if "chain" not in st.session_state:
    st.info("👈 Enter a GitHub repo URL in the sidebar to get started")
else:
    st.subheader(f"Chatting with `{st.session_state.repo_name}`")

    # display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # chat input
    if question := st.chat_input("Ask anything about this repo..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.chain.invoke(question)
                docs = st.session_state.retriever.invoke(question)
                sources = list(set(d.metadata["source"] for d in docs))

            st.markdown(answer.content)
            with st.expander("📁 Sources"):
                for s in sources:
                    st.code(s)

        st.session_state.messages.append({"role": "assistant", "content": answer.content})