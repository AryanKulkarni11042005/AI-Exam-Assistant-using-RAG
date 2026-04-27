from rank_bm25 import BM25Okapi

def create_bm25(chunks):

    text_chunks = [doc.page_content for doc in chunks]
    tokenized_docs = [doc.split() for doc in text_chunks]

    bm25 = BM25Okapi(tokenized_docs)
    return bm25,text_chunks

def retrieve_bm25(bm25,query,text_chunks,k=3):

    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    results = [text_chunks[i] for i in top_indices]
    return results