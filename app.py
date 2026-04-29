import streamlit as st
from document_loader import load_documents, chunk_documents
from vector_store import create_vector_store, load_vector_store
from rag_chain import build_rag_chain
import os

st.set_page_config(
    page_title="DocChat — RAG Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("DocChat")
st.caption("Ask questions about your documents using AI")

@st.cache_resource
def initialize_rag():
    if os.path.exists("vectorstore/faiss_index"):
        vectorstore = load_vector_store()
    else:
        docs = load_documents("data/")
        chunks = chunk_documents(docs)
        vectorstore = create_vector_store(chunks)
    chain, retriever = build_rag_chain(vectorstore)
    return chain, retriever

with st.spinner("Loading knowledge base..."):
    chain, retriever = initialize_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            answer = chain.invoke(prompt)
            sources = retriever.invoke(prompt)

        st.markdown(answer)

        with st.expander("View source documents"):
            for i, doc in enumerate(sources):
                st.markdown(f"**Source {i+1}:** {doc.metadata.get('source', 'Unknown')}, page {doc.metadata.get('page', '?')}")
                st.markdown(f"> {doc.page_content[:300]}...")
                st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})