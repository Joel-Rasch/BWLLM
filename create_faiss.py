import os
import re
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

import re
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings

def build_faiss_index(
    input_dir: Path = Path("data/processed"),
    chunk_size: int = 200,
    chunk_overlap: int = 0,
    index_path: str = "faiss_index",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> None:
    # Text Splitter konfigurieren
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    all_documents = []

    for md_file in input_dir.glob("*.md"):
        filename = md_file.stem  # z. B. "bmw_2023_processed"
        parts = filename.split("_")

        if len(parts) < 2:
            continue  # Datei entspricht nicht dem erwarteten Muster

        company = parts[0]
        year = parts[1]

        with md_file.open("r", encoding="utf-8") as f:
            text = f.read()

        # Preprocessing
        text = text.replace('\u202f', ' ')
        text = re.sub(r'(?<=\d)\.(?=\d)', '', text)

        chunks = text_splitter.split_text(text)

        for i, chunk in enumerate(chunks):
            all_documents.append(Document(
                page_content=chunk,
                metadata={"company": company, "year": year, "chunk": i}
            ))

    # Embeddings erzeugen
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # FAISS-Index erstellen
    faiss_index = FAISS.from_documents(all_documents, embedding=embeddings)

    # Index lokal speichern
    faiss_index.save_local(index_path)

    print(f"✅ FAISS-Index erfolgreich erstellt und gespeichert unter: {index_path}")