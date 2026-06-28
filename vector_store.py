import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
from dotenv import load_dotenv

load_dotenv()

def build_vectorstore(files):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    docs = []
    for f in files:
        chunks = splitter.create_documents(
            texts=[f["content"]],
            metadatas=[{"source": f["path"]}]
        )
        docs.extend(chunks)

    print(f"Total chunks: {len(docs)}")

    vectorstore = Chroma.from_documents(
         documents=docs,
         embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
         collection_name=f"repo_{uuid.uuid4().hex}",
         client_settings=chromadb.config.Settings(is_persistent=False),
    )
    return vectorstore


# ------ test -------
if __name__ == "__main__":
    from ingest_repos import get_repo_files

    files = get_repo_files("tiangolo", "fastapi")
    vs = build_vectorstore(files)

    # quick retrieval test
    results = vs.similarity_search("how does routing work?", k=3)
    for r in results:
        print(r.metadata["source"])
        print(r.page_content)
        print("---")