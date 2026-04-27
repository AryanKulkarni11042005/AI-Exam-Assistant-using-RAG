# AI Exam Assistant using Hybrid RAG

## Overview

AI Exam Assistant is a Hybrid RAG (Retrieval-Augmented Generation) project built to generate structured exam-style answers from uploaded PDF notes.

Instead of relying only on vector search, this project combines:

* Vector Search using ChromaDB
* Keyword Search using BM25
* Groq LLM (`llama-3.1-8b-instant`)
* LangChain-based document pipeline
* Streamlit UI for interactive usage

This helps produce better retrieval quality and stronger final answers for theory-heavy academic PDFs.

---

## Features

* Upload one or multiple PDFs
* Hybrid Retrieval (Vector Search + BM25)
* Exam-style answer generation
* Comparison tables where needed
* Source-aware retrieval context
* Streamlit-based UI

---

## Tech Stack

### Backend / RAG

* Python
* LangChain
* ChromaDB
* BM25 (`rank-bm25`)
* Sentence Transformers
* HuggingFace Embeddings
* Groq API
* Llama 3.1 8B Instant

### Frontend

* Streamlit

---

## Project Architecture

```text
PDF Upload
↓
Loader (PyMuPDFLoader)
↓
Chunking (RecursiveCharacterTextSplitter)
↓
Embeddings (all-MiniLM-L6-v2)
↓
ChromaDB Vector Store
+
BM25 Keyword Search
↓
Hybrid Search
↓
Groq LLM (Llama 3.1 8B Instant)
↓
Final Structured Answer
```

---

## Folder Structure

```text
AI-Exam-Assistant/
│
├── app.py
├── loader.py
├── chunking.py
├── vectorstore.py
├── retrieval.py
├── bm25_retriever.py
├── hybrid_search.py
├── llm.py
├── rag_pipeline.py
│
├── data/
│   └── sample.pdf
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup Instructions

### 1. Clone Repository

```bash
git clone <your-repo-link>
cd AI-Exam-Assistant
```

---

### 2. Create Virtual Environment

### Mac / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Add Groq API Key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

You can get a free API key from Groq Console.

---

### 5. Run the Application

```bash
python -m streamlit run app.py
```

---

## Example Questions

Try asking:

* Explain Hard Computing vs Soft Computing for 10 marks
* Difference between AI and Computational Intelligence
* Explain Learning and Adaptation in Soft Computing
* Compare LIME and SHAP
* What is Explainable AI?

---

## Why Hybrid RAG?

Pure vector search sometimes misses exact academic keywords like:

* GDPR
  n- LIME
* SHAP
* ANN
* Neurocomputing
* Hard Computing

BM25 improves exact keyword matching, while vector search improves semantic understanding.

Combining both gives significantly better retrieval quality.

---

## Future Improvements

* Agentic RAG using LangGraph
* Metadata filtering
* Re-ranking
* Conversational memory
* SQL + Vector Hybrid RAG
* Deployment on Streamlit Cloud

---
