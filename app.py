import streamlit as st
import tempfile
import os
from document_loader import load_uploaded_pdf, chunk_documents
from vector_store import create_vector_store
from rag_chain import build_rag_chain

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocChat — RAG Chatbot",
    page_icon="📄",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0F1117; }
    .stChatMessage { background: #1E2130; border-radius: 12px; }
    section[data-testid="stSidebar"] {
        background: #1A1D2E;
        border-right: 1px solid #2D3250;
    }
    .title-text {
        font-size: 32px;
        font-weight: 700;
        color: #7B8CDE;
        margin: 0;
    }
    .subtitle-text {
        font-size: 14px;
        color: #555;
        margin: 4px 0 20px;
    }
    .info-card {
        background: #1E2130;
        border: 1px solid #2D3250;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        font-size: 13px;
        color: #AAA;
        line-height: 1.6;
    }
    .step-badge {
        display: inline-block;
        background: #2D3250;
        color: #7B8CDE;
        font-size: 11px;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 10px;
        margin-bottom: 6px;
    }
    hr { border-color: #2D3250; }
    .footer {
        text-align: center;
        color: #444;
        font-size: 12px;
        margin-top: 40px;
        padding-top: 16px;
        border-top: 1px solid #2D3250;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
    <p class='title-text'>📄 DocChat</p>
    <p class='subtitle-text'>
        Upload any PDF and ask questions about it in plain English · 
        Built by <a href='https://www.linkedin.com/in/ansar-kamal-268045330/'
        target='_blank' style='color:#7B8CDE;text-decoration:none;'>Ansar Kamal</a>
    </p>
    <hr>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Setup")
    st.markdown("---")

    # Step 1 — API Key
    st.markdown('<span class="step-badge">STEP 1</span>', unsafe_allow_html=True)
    st.markdown("**Get your free Groq API key**")
    st.markdown(
        "👉 [Get free key at console.groq.com](https://console.groq.com)",
        unsafe_allow_html=True
    )
    groq_api_key = st.text_input(
        "Paste your Groq API key here",
        type="password",
        placeholder="gsk_..."
    )

    st.markdown("---")

    # Step 2 — Upload PDF
    st.markdown('<span class="step-badge">STEP 2</span>', unsafe_allow_html=True)
    st.markdown("**Upload your PDF**")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Max recommended size: 10MB"
    )

    st.markdown("---")

    # Step 3 — Process
    st.markdown('<span class="step-badge">STEP 3</span>', unsafe_allow_html=True)
    st.markdown("**Process the document**")
    process_btn = st.button("🚀 Process Document", use_container_width=True)

    st.markdown("---")
    st.caption("Model: Llama 3.3 70B via Groq\nEmbeddings: MiniLM-L6 (local)\nVector DB: FAISS")

# ── Main area ─────────────────────────────────────────────────────────────────

# Show instructions if not set up yet
if not groq_api_key or not uploaded_file:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class='info-card'>
            <span class='step-badge'>STEP 1</span><br>
            <strong style='color:#CCC;'>Get a free Groq API key</strong><br><br>
            Go to <a href='https://console.groq.com' target='_blank'
            style='color:#7B8CDE;'>console.groq.com</a>, sign up for free,
            and copy your API key. Groq gives you fast LLaMA 3 access
            at no cost.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='info-card'>
            <span class='step-badge'>STEP 2</span><br>
            <strong style='color:#CCC;'>Upload any PDF</strong><br><br>
            Research papers, business reports, textbooks, contracts —
            anything. The app reads it, breaks it into chunks, and
            builds a searchable knowledge base locally on your machine.
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='info-card'>
            <span class='step-badge'>STEP 3</span><br>
            <strong style='color:#CCC;'>Ask questions in plain English</strong><br><br>
            Type any question about your document. The AI finds the
            most relevant sections and generates a precise answer —
            with source page references so you can verify everything.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-card' style='margin-top:8px;'>
        <strong style='color:#CCC;'>🔒 Privacy first</strong> —
        Your PDF is processed locally on this machine.
        Only your question and the relevant text chunks are
        sent to Groq's API. Your document is never uploaded anywhere.
    </div>
    """, unsafe_allow_html=True)

# ── Process document ──────────────────────────────────────────────────────────
if process_btn:
    if not groq_api_key:
        st.error("⚠️ Please enter your Groq API key in the sidebar first.")
    elif not uploaded_file:
        st.error("⚠️ Please upload a PDF file first.")
    else:
        with st.spinner("Reading and processing your document... this takes about 30 seconds on first run."):
            try:
                # Save uploaded file to temp location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                # Load, chunk, embed
                docs   = load_uploaded_pdf(tmp_path)
                chunks = chunk_documents(docs)
                vectorstore = create_vector_store(chunks)
                chain, retriever = build_rag_chain(vectorstore, groq_api_key)

                # Store in session
                st.session_state.chain     = chain
                st.session_state.retriever = retriever
                st.session_state.messages  = []
                st.session_state.doc_name  = uploaded_file.name
                st.session_state.ready     = True

                # Cleanup temp file
                os.unlink(tmp_path)

                st.success(f"✅ '{uploaded_file.name}' processed successfully! Ask your first question below.")

            except Exception as e:
                st.error(f"Error processing document: {str(e)}")

# ── Chat interface ────────────────────────────────────────────────────────────
if st.session_state.get("ready"):
    st.markdown(f"**📄 Active document:** `{st.session_state.doc_name}`")
    st.markdown("---")

    # Show chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask anything about your document..."):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                answer  = st.session_state.chain.invoke(prompt)
                sources = st.session_state.retriever.invoke(prompt)

            st.markdown(answer)

            # Show sources
            with st.expander("📚 View source sections from document"):
                for i, doc in enumerate(sources):
                    page = doc.metadata.get('page', '?')
                    st.markdown(f"**Source {i+1} — Page {page}**")
                    st.markdown(f"> {doc.page_content[:300]}...")
                    st.divider()

        st.session_state.messages.append({"role": "assistant", "content": answer})

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
    <div class='footer'>
        Built by <a href='https://www.linkedin.com/in/ansar-kamal-268045330/'
        style='color:#7B8CDE;'>Ansar Kamal</a> ·
        <a href='https://github.com/Haru-bit-code/rag-document-chatbot'
        style='color:#7B8CDE;'>GitHub</a> ·
        Powered by LangChain · Groq · FAISS
    </div>
""", unsafe_allow_html=True)
