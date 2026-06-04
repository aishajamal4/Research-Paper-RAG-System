import streamlit as st

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM

DB_DIR = "vector_db"

st.set_page_config(
    page_title="AI Research Paper RAG Assistant",
    page_icon="📚"
)

st.title("📚 AI Research Paper RAG Assistant")
st.write("Ask questions based on your uploaded research papers.")

@st.cache_resource
def load_rag():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )

    llm = OllamaLLM(
        model="llama3.2:3b"
    )

    return db, llm

db, llm = load_rag()

question = st.text_input("Ask a question:")

if st.button("Get Answer"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching papers and generating answer..."):
            docs = db.similarity_search(question, k=5)

            context = "\n\n".join([
                f"Source: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')}\n{doc.page_content}"
                for doc in docs
            ])

            prompt = f"""
You are a research assistant.

Answer the question using ONLY the context below.
If the answer is not found in the context, say:
"The answer is not available in the provided papers."

Question:
{question}

Context:
{context}

Answer:
"""

            answer = llm.invoke(prompt)

        st.subheader("Final Answer")
        st.write(answer)

        st.subheader("Sources")
        seen = set()

        for doc in docs:
            source = doc.metadata.get("source")
            page = doc.metadata.get("page")
            key = (source, page)

            if key not in seen:
                st.write(f"- {source} | Page {page}")
                seen.add(key)