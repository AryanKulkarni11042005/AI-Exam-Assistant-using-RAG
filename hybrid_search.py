


def hybrid_search(
        vector_store,
        bm25,
        query,
        text_chunks,
        k=3
):
    vector_results = vector_store.similarity_search(query, k=k)
    vector_texts = [doc.page_content for doc in vector_results]
    tokenized_query = query.split()
    score = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(score)), key=lambda i: score[i], reverse=True)[:k]
    bm25_results = [text_chunks[i] for i in top_indices]
    combined_results = list(
        set(vector_texts + bm25_results)
    )
    return combined_results