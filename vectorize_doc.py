import os
os.environ["PATH"] += os.pathsep + r"D:\term3\CAPP\poppler-24.08.0\Library\bin"

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Loading the embedding model
embeddings = HuggingFaceEmbeddings()

# Load PDFs using UnstructuredFileLoader
loader = DirectoryLoader(
    path="data",
    glob="./*.pdf",
    loader_cls=UnstructuredFileLoader
)
documents = loader.load()

# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200
)
text_chunks = text_splitter.split_documents(documents)

# Create a Chroma vector store
vectordb = Chroma.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    persist_directory="vector_db_dir"
)

vectordb.persist()  # <-- this saves the vectors
print("✅ Documents Vectorized and Saved")
