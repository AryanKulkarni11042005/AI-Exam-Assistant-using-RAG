

def generate_answer(llm, retrieved_docs, query):
    context = "\n\n".join(retrieved_docs)

    prompt = f""" 
You are an exam preparation assistant.

Answer in proper 10-mark format:

1. Introduction
2. Main Points
3. Comparison Table (if needed)
4. Example
5. Conclusion

Use ONLY the context below.

Context:
{context}

Question:
{query}
"""
    response = llm.invoke(prompt)
    return response