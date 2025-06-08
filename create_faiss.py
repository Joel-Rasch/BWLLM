from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from sklearn.datasets import fetch_20newsgroups # nur für testdaten

# Beispiel-Daten: Ersetze das mit deinem tatsächlichen Textinput
data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes')).data

# Parameter
index_path = "faiss_index"
model_name = "sentence-transformers/all-MiniLM-L6-v2"
chunk_size = 500
chunk_overlap = 100

# Dokument-ID-Zuordnung
texts = {f"doc_{i}": text for i, text in enumerate(data)}

# Text splitter konfigurieren
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", ".", " ", ""]
)

# Texte in Chunks aufteilen mit Metadaten
documents = []
for doc_id, text in texts.items():
    chunks = text_splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        documents.append(Document(
            page_content=chunk,
            metadata={"source": doc_id, "chunk": i}
        ))

# Embeddings vorbereiten
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# FAISS-Index erstellen
faiss_index = FAISS.from_documents(documents, embedding=embeddings)

# Lokal speichern
faiss_index.save_local(index_path)

print(f"✅ FAISS-Index erfolgreich unter '{index_path}' gespeichert.")