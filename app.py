import streamlit as st
import tempfile
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# =========================
# SET YOUR OPENAI API KEY
# =========================
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="SISTec Info Bot",
    page_icon="🎓",
    layout="centered"
)

# =========================
# TITLE
# =========================
st.title("🎓 SISTec Info Bot")
st.write("Ask questions from uploaded college documents.")

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader(
    "Upload College PDF",
    type="pdf"
)

# =========================
# PROCESS PDF
# =========================
if uploaded_file:

    # Save temporary PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_pdf_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(temp_pdf_path)
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.split_documents(documents)

    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Store vectors in FAISS
    vector_store = FAISS.from_documents(docs, embeddings)

    st.success("✅ PDF Uploaded & Processed Successfully!")

    # =========================
    # QUESTION INPUT
    # =========================
    question = st.text_input(
        "Ask a Question"
    )

    # =========================
    # ANSWER QUESTION
    # =========================
    if question:

        # Similarity Search
        results = vector_store.similarity_search_with_score(
            question,
            k=3
        )

        relevant_docs = [doc for doc, score in results]

        # Out-of-scope rejection
        if results[0][1] > 1.0:

            st.error(
                "❌ Answer not found in uploaded document."
            )

        else:

            # Load LLM
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0
            )

            # QA Chain
            chain = load_qa_chain(
                llm,
                chain_type="stuff"
            )

            # Generate answer
            response = chain.run(
                input_documents=relevant_docs,
                question=question
            )

            # =========================
            # DISPLAY ANSWER
            # =========================
            st.subheader("📌 Answer")
            st.write(response)

            # =========================
            # DISPLAY SOURCE
            # =========================
            st.subheader("📄 Source Chunk")
            st.write(relevant_docs[0].page_content)

    # Remove temporary file
    os.remove(temp_pdf_path)
