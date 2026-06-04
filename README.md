# AI Research Paper RAG Assistant

## Overview

AI Research Paper RAG Assistant is an end-to-end Retrieval-Augmented Generation (RAG) system designed to answer questions based on a collection of AI research papers.

Instead of relying solely on the knowledge stored in a language model, the system retrieves relevant information from uploaded research papers and uses it as context for generating accurate, source-grounded responses.

The project demonstrates the complete RAG workflow used in modern AI applications.

---

## Features

* Research paper ingestion from PDF files
* Automatic document parsing and text extraction
* Document chunking using recursive text splitting
* Embedding generation using Sentence Transformers
* Vector storage using ChromaDB
* Semantic similarity search
* Context retrieval pipeline
* Local LLM inference using Ollama
* Source-grounded answer generation
* Source citation display
* Interactive Streamlit web interface

---

## System Architecture

User Question

↓

Embedding Generation

↓

Vector Search (ChromaDB)

↓

Top-K Relevant Chunks

↓

LLM (Ollama)

↓

Final Answer + Sources

---

## Tech Stack

### AI / NLP

* LangChain
* Sentence Transformers
* Retrieval-Augmented Generation (RAG)
* Semantic Search
* Embeddings

### Vector Database

* ChromaDB

### LLM

* Ollama
* Llama 3.2

### Frontend

* Streamlit

### Backend

* Python

---

## Project Workflow

### 1. Document Ingestion

Research papers are loaded from PDF files and converted into text.

### 2. Text Chunking

Documents are divided into smaller overlapping chunks to improve retrieval accuracy.

### 3. Embedding Generation

Each chunk is converted into a dense vector representation using:

all-MiniLM-L6-v2

### 4. Vector Storage

Embeddings are stored in ChromaDB for efficient similarity search.

### 5. Retrieval

When a user submits a question, the system retrieves the most relevant document chunks using semantic similarity.

### 6. Answer Generation

Retrieved context is sent to a local LLM through Ollama to generate a grounded answer.

### 7. Source Citation

The system displays the source paper and page number used to generate the answer.

---

## Example Query

Question:

What is Retrieval-Augmented Generation?

Answer:

Retrieval-Augmented Generation (RAG) is a framework that enhances language models by retrieving relevant information from external knowledge sources and incorporating it into the generation process to improve factual accuracy and reduce hallucinations.

Sources:

* RAG Survey Paper (Page 15)
* Self-RAG Paper (Page 8)

---

## Skills Demonstrated

* Large Language Models (LLMs)
* Retrieval-Augmented Generation (RAG)
* Vector Databases
* Semantic Search
* Embeddings
* Prompt Engineering
* Information Retrieval
* LangChain
* Local AI Deployment
* Research-Oriented NLP Systems

---

## Future Improvements

* Multi-document upload from the UI
* Conversational memory
* Hybrid retrieval (BM25 + Vector Search)
* Reranking models
* Evaluation metrics
* Multi-agent retrieval workflows
* Citation highlighting
* Research paper summarization
