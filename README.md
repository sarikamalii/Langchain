# 📚 GroqDocChat: Multimodal RAG App for PDF Q&A

This project demonstrates a Retrieval-Augmented Generation (RAG) chatbot powered by the **Groq API** and **LangChain**, capable of answering questions from large PDF documents using **HuggingFace embeddings** and **Chroma vector store**.

💼 Business Problem
Organizations often store critical knowledge in lengthy, unstructured PDF documents—legal contracts, financial reports, compliance manuals, and more. Manually searching through these documents to find specific information is time-consuming and error-prone.
This project solves that problem by using a Retrieval-Augmented Generation (RAG) approach. It enables users to ask natural language questions and get accurate, contextual answers directly from PDFs, along with source page references—making document intelligence fast, scalable, and reliable.

📂 **Technologies**: Python, Streamlit, LangChain, HuggingFace, ChromaDB, Groq API

---

## 🚀 Features

- Upload and index large PDFs into vector stores
- Ask natural language questions based on PDF contents
- Uses **LLaMA-3 (70B)** via Groq for high-quality, low-latency generation
- Displays **source pages** to ensure transparency
- Saves and displays chat history using Streamlit

---

## 🧠 Use Cases

- Legal/contract analysis
- Financial report summarization
- Research paper Q&A
- Business intelligence from internal documents

---

## 🛠️ Tech Stack

| Component | Description |
|----------|-------------|
| **LLM** | LLaMA-3-70B via `ChatGroq` |
| **Embeddings** | `HuggingFaceEmbeddings()` |
| **Vector Store** | `Chroma` DB |
| **Frontend** | `Streamlit` Chat Interface |
| **Document Loader** | `UnstructuredFileLoader` for PDF parsing |

---

