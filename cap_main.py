import os
import json
import streamlit as st
from pathlib import Path

from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# --- Load Groq API key from config.json ---
working_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(working_dir, "config.json")
with open(config_path) as f:
    config = json.load(f)

groq_key = config["GROQ_API_KEY"]

# --- PDF path and vector DB folder ---
file_path = "doc/your_file.pdf"  # ⬅️ Replace with your actual PDF path
persist_directory = "vector_db_dir"

# --- Streamlit UI setup ---
st.set_page_config(page_title="📚 Groq-Powered RAG Chatbot", layout="centered")
st.title("📚 Multimodal RAG App (Groq-powered)")

# --- Load or create vector store ---
if Path(persist_directory).exists():
    st.success("✅ Vector DB found. Loading from disk...")
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=HuggingFaceEmbeddings())
else:
    with st.spinner("Processing PDF and creating vector store..."):
        st.write("🔹 Loading PDF...")
        loader = UnstructuredFileLoader(file_path)
        documents = loader.load()

        st.write("🔹 Splitting into chunks...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        docs = splitter.split_documents(documents)

        st.write("🔹 Creating embeddings and vector DB...")
        embeddings = HuggingFaceEmbeddings()
        vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_directory)
        vectordb.persist()

    st.success("✅ Vector store created and saved!")

# --- Setup Groq LLM ---
llm = ChatGroq(
    model_name="llama3-70b-8192",  # ✅ updated from mixtral
    temperature=0,
    groq_api_key=groq_key
)

# --- Setup Retriever and QA chain ---
retriever = vectordb.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# --- Initialize chat history ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Display previous chat history ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---
user_input = st.chat_input("Ask your question:")

if user_input:
    # Save user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            result = qa_chain(user_input)
            answer = result["result"]
            st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

        # Show sources
        if result.get("source_documents"):
            with st.expander("📄 Source Pages"):
                for doc in result["source_documents"]:
                    page = doc.metadata.get("page", "N/A")
                    source = doc.metadata.get("source", "Unknown")
                    st.markdown(f"📄 **Page:** {page} — `{source}`")
