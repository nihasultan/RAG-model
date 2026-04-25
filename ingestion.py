import os
from functools import lru_cache
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_FOLDER = "data"

def clean_text(text):
    return text.replace("\n", " ").strip()

def process_uploaded_file(uploaded_file):
    os.makedirs(DATA_FOLDER, exist_ok=True)
    file_path = os.path.join(DATA_FOLDER, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

def load_and_process_pdfs():
    documents = []

    for file in os.listdir(DATA_FOLDER):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_FOLDER, file))
            docs = loader.load()

            for d in docs:
                d.metadata["source"] = file

            documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)

    processed_chunks = []
    for chunk in chunks:
        processed_chunks.append({
            "text": clean_text(chunk.page_content),
            "source": chunk.metadata["source"],
            "page": chunk.metadata.get("page", 1) + 1
        })

    return processed_chunks