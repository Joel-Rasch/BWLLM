from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

data = []

texts = {f"doc_{i}": text for i, text in enumerate(data.data[:20])}

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""],  # Wichtig für gute Schnitte
)

# Chunks erzeugen + Metadaten behalten
documents = []

for doc_id, text in texts.items():
    chunks = text_splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        documents.append(Document(
            page_content=chunk,
            metadata={"source": doc_id, "chunk": i}
        ))

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# FAISS-Index erstellen (automatisch mit Embeddings)
faiss_index = FAISS.from_documents(documents, embedding=embeddings)

# Lokal speichern
faiss_index.save_local("faiss_index")