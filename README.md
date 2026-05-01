# 📄 DocChat — RAG Document Chatbot

> Upload any PDF. Ask questions in plain English. Get answers with source references.

**[🚀 Live Demo][Click here(https://rag-document-chatbot-bwcju5lannbmbapddhvfmb.streamlit.app/)** · Built by [Ansar Kamal](https://www.linkedin.com/in/ansar-kamal-268045330/)

---

## What is this?

Most AI chatbots only know what they were trained on. DocChat is different — it reads *your* document and answers questions about it specifically.

Upload a research paper, a business report, a textbook, or any PDF. Then ask anything:
- *"What is the main conclusion of this paper?"*
- *"What were the revenue figures in Q3?"*
- *"Summarize the methodology section"*

It finds the most relevant parts of your document and generates a precise answer — with source page numbers so you can verify everything.

---

## How it works

```
Your PDF → Split into chunks → Convert to vectors → Store in FAISS
                                                            ↓
Your Question → Convert to vector → Find similar chunks → Send to LLM → Answer
```

1. **Document Loading** — PyPDF reads your PDF page by page
2. **Chunking** — Text is split into 1000-character overlapping chunks so context isn't lost at boundaries
3. **Embedding** — Each chunk is converted into a 384-dimension vector using MiniLM-L6 (runs locally, no API needed)
4. **Vector Search** — When you ask a question, FAISS finds the 4 most relevant chunks in milliseconds
5. **LLM Generation** — Those chunks + your question go to Llama 3.3 70B via Groq, which generates a grounded answer

---

## Tech Stack

| Component | Tool | Why I chose it |
|-----------|------|----------------|
| LLM | Groq (Llama 3.3 70B) | Free tier, extremely fast inference |
| Embeddings | sentence-transformers/MiniLM-L6 | Runs fully locally, no API cost |
| Vector DB | FAISS | No server needed, works in-memory |
| Framework | LangChain | Industry standard for RAG pipelines |
| UI | Streamlit | Fast to build, easy to deploy |

---

## Run it yourself (free)

**1. Clone the repo**
```bash
git clone https://github.com/Haru-bit-code/rag-document-chatbot
cd rag-document-chatbot
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Get a free Groq API key**

Go to [console.groq.com](https://console.groq.com), sign up (free), and copy your API key. No credit card needed.

**4. Run the app**
```bash
streamlit run app.py
```

**5. Use it**
- Paste your Groq API key in the sidebar
- Upload any PDF
- Click "Process Document"
- Start asking questions

---

## What I learned building this

The hardest part wasn't the code — it was understanding **why chunking strategy matters**. 

My first version used chunks of 500 characters with no overlap. The answers were bad because important sentences were being cut in half between chunks. Increasing to 1000 characters with 200-character overlap fixed this — context stayed intact across chunk boundaries.

I also learned that FAISS similarity search returns the closest vectors by cosine distance, not keyword match. This means it finds *semantically similar* content even if the exact words don't match — which is why RAG works so much better than simple search.

---

## Project structure

```
rag-document-chatbot/
├── app.py                  # Main Streamlit UI
├── document_loader.py      # PDF loading and chunking
├── vector_store.py         # FAISS embedding and retrieval
├── rag_chain.py            # LangChain RAG pipeline
├── requirements.txt        # Dependencies
└── assets/                 # Screenshots and demo
```

---

## About me

I'm Ansar Kamal, a Data Science & AI student at Boston Institute of Analytics, Kerala. I build real projects to learn — not just tutorials.

- 🔗 [LinkedIn](https://www.linkedin.com/in/ansar-kamal-268045330/)
- 💻 [GitHub](https://github.com/Haru-bit-code)
- 🔮 [My other project: Customer Churn Predictor](https://github.com/Haru-bit-code/customer-churn-prediction)
