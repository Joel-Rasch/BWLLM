from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

faiss_index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

results = faiss_index.similarity_search("What is a Ducati 900GTS 1978", k=3)