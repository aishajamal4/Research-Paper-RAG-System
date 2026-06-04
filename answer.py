from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM

DB_DIR = "vector_db"

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

question = input("Ask a question: ")

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

print("\nFinal Answer:\n")
print(answer)

print("\nSources:\n")

seen = set()

for doc in docs:
    source = doc.metadata.get("source")
    page = doc.metadata.get("page")
    key = (source, page)

    if key not in seen:
        print(f"- {source} | Page {page}")
        seen.add(key)