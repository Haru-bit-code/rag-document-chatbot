from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

PROMPT_TEMPLATE = """You are a helpful assistant that answers questions 
based strictly on the provided document context.

If the answer is in the context, answer clearly and concisely.
If the answer is NOT in the context, say:
"I couldn't find information about that in this document."

Always be direct. Never make up information.

Context:
{context}

Question: {question}

Answer:"""

def build_rag_chain(vectorstore, groq_api_key: str):
    """Build the RAG chain using the provided Groq API key."""
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.2
    )

    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever
