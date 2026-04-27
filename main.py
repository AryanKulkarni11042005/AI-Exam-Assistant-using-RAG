from loader import load_pdf
from chunking import chunk_docs
from vectorstore import create_vector_store
from retrieval import retrieve_docs
from bm25_retriever import create_bm25, retrieve_bm25
from hybrid_search import hybrid_search
from llm import create_llm
from rag_pipeline import generate_answer



file_path = "data/Module 1.pdf"
docs = load_pdf(file_path)
chunks = chunk_docs(docs)
vectorstore = create_vector_store()
# vectorstore.add_documents(chunks)

query = "Explain Hard Computing vs Soft Computing"
vector_store_results = retrieve_docs(vectorstore, query, k=3)


bm25, text_chunks = create_bm25(chunks)

bm25_results = retrieve_bm25(
    bm25,
    query,
    text_chunks,
    k=3
)



llm = create_llm()
final_results = hybrid_search(
    vectorstore,
    bm25,
    query,
    text_chunks,
    k=3
)

final_answer = generate_answer(
    llm,
    final_results,
    query
)
print("\nFINAL ANSWER:\n")
print(final_answer)