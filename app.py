import streamlit as st
from ingestion import process_uploaded_file
from pipeline import initialize, ask

st.title("📄 Ask Your PDF")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    process_uploaded_file(uploaded_file)
    initialize()

    query = st.text_input("Ask a question about your document:")

    if query:
        with st.spinner("Thinking..."):
            answer, sources = ask(query)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        for s in sources:
            st.write(f"{s['source']} - Page {s['page']}")