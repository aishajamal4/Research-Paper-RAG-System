from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

PAPERS_DIR = "papers"
DB_DIR = "vector_db"

def load_papers():
    documents = []

    for file in os.listdir(PAPERS_DIR):
        if file.endswith(".pdf"):
            path = os.path.join(PAPERS_DIR, file)
            loader = PyPDFLoader(path)
            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = file

            documents.extend(docs)

    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    return splitter.split_documents(documents)

def create_vector_db(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    db.persist()
    print("Vector database created successfully.")

if __name__ == "__main__":
    documents = load_papers()
    chunks = split_documents(documents)
    create_vector_db(chunks)