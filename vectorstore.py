from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import uuid


def create_vector_store():
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        collection_name=f"exam_{uuid.uuid4()}",
        embedding_function=embedding_function
    )

    return vector_store