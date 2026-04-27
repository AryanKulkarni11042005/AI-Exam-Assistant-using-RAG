from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def create_vector_store():
    # Embedding function object
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create persistent Chroma DB
    vector_store = Chroma(
        collection_name="exam_assistant",
        embedding_function=embedding_function,
        persist_directory="./chroma_langchain_db",
    )

    return vector_store