from operator import itemgetter

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

# Prompt
PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert code assistant helping developers understand a GitHub repository.

Use the provided context to answer the user's question.
Always mention which file the information comes from.
If you don't know the answer, say so — don't make things up.
     
Context:
{context}
"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])


def build_qa_chain(vectorstore):
    # Create retriever (top 5 chunks)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Chat model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Format retrieved docs into a single text block
    def format_docs(docs):
        return "\n\n".join(
            f"SOURCE: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
            for doc in docs
        )

    # LCEL pipeline
    chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | PROMPT
        | llm
    )

    return chain, retriever


# ------ test -------
if __name__ == "__main__":
    from ingest_repos import get_repo_files
    from vector_store import build_vectorstore

    # Load repo
    files = get_repo_files("tiangolo", "fastapi")

    # Build vector store
    vs = build_vectorstore(files)

    # Build chain
    chain, retriever = build_qa_chain(vs)

    questions = [
        "How does routing work?",
        "What is the folder structure of this repo?",
        "How does FastAPI handle authentication?",
    ]

    for q in questions:
        print(f"\nQ: {q}")

        result = chain.invoke(q)
        print(f"A: {result.content}")

        docs = retriever.invoke(q)
        print("Sources:", [d.metadata["source"] for d in docs])
        print("---")