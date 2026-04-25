📄 Ask My Docs – Production RAG Application

An end-to-end Retrieval-Augmented Generation (RAG) system that allows users to query documents using natural language.

 Features

- Hybrid Retrieval (BM25 + FAISS vector search)
- Cross-Encoder Reranking (MS MARCO MiniLM)
- Local LLM inference using TinyLlama via Ollama
- Source-grounded answers with citations
- Streamlit UI with file upload support
- Retrieval evaluation pipeline

Architecture

Query  
→ Hybrid Search (BM25 + Vector)  
→ Reranking  
→ Context Construction  
→ Local LLM  
→ Answer + Sources  

Tech Stack

Python, Streamlit, FAISS, Sentence Transformers, BM25, Ollama

Run Locally

pip install -r requirements.txt
ollama pull tinyllama
streamlit run app.py
