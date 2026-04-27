from langchain_community.document_loaders import PyMuPDFLoader



def load_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    return docs

from langchain_community.document_loaders import PyMuPDFLoader


def load_pdfs(file_paths):
    all_docs = []

    for file_path in file_paths:
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        all_docs.extend(docs)

    return all_docs
