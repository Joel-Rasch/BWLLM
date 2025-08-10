from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Globale Variablen
embeddings = None
faiss_index = None
all_docs = None
known_companies = None

def load_variables(index_path="faiss_index", model_name="sentence-transformers/all-MiniLM-L6-v2"):
    global embeddings, faiss_index, all_docs, known_companies

    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(model_name=model_name)

    if faiss_index is None:
        faiss_index = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    if all_docs is None:
        all_docs = list(faiss_index.docstore._dict.values())

    if known_companies is None:
        known_companies = set(doc.metadata.get("company", "").lower() for doc in all_docs if doc.metadata.get("company"))
