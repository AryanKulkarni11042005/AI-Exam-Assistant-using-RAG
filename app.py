import streamlit as st
import tempfile

from loader import load_pdfs
from chunking import chunk_docs
from vectorstore import create_vector_store
from bm25_retriever import create_bm25
from hybrid_search import hybrid_search
from llm import create_llm
from rag_pipeline import generate_answer


st.set_page_config(
    page_title="AI Exam Assistant",
    page_icon="📘",
    layout="wide"
)

st.title("📘 AI Exam Assistant")
st.markdown(
    "Upload one or more PDFs and generate structured exam-style answers using Hybrid RAG."
)


uploaded_files = st.file_uploader(
    "Upload your PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload one or more study PDFs"
)


@st.cache_resource
def initialize_pipeline(file_paths):
    docs = load_pdfs(file_paths)

    chunks = chunk_docs(docs)

    vectorstore = create_vector_store()

    try:
        if vectorstore._collection.count() == 0:
            vectorstore.add_documents(chunks)
    except Exception:
        pass

    bm25, text_chunks = create_bm25(chunks)

    llm = create_llm()

    return vectorstore, bm25, text_chunks, llm


query = st.text_area(
    "Enter your question",
    placeholder="Example: Explain Hard Computing vs Soft Computing for 10 marks",
    height=120
)


if uploaded_files:
    temp_pdf_paths = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_pdf_paths.append(tmp_file.name)

    vectorstore, bm25, text_chunks, llm = initialize_pipeline(
        temp_pdf_paths
    )

    if st.button("Generate Answer", use_container_width=True):

        if not query.strip():
            st.warning("Please enter a question first.")

        else:
            with st.spinner(
                "Retrieving relevant content and generating answer..."
            ):

                retrieved_docs = hybrid_search(
                    vectorstore,
                    bm25,
                    query,
                    text_chunks,
                    k=4
                )

                final_answer = generate_answer(
                    llm,
                    retrieved_docs,
                    query
                )

                if hasattr(final_answer, "content"):
                    final_answer = final_answer.content

            st.subheader("Generated Answer")
            st.markdown(final_answer)

            with st.expander("Retrieved Context"):
                for i, doc in enumerate(retrieved_docs, 1):
                    st.markdown(f"### Chunk {i}")
                    st.write(doc)
                    st.divider()

else:
    st.info("Please upload one or more PDFs first.")