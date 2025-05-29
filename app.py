import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from config import config
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS  # Import FAISS

class SimpleRAGSystem:
    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self.qa_chain = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len
        )

    @st.cache_resource
    def initialize(_self):
        """Initialize the RAG system (cached for performance)"""
        if not config.GOOGLE_API_KEY: # Ensure your config.py loads GOOGLE_API_KEY
            st.error("Google API Key fehlt! Bitte in .env Datei eintragen und config.py anpassen.")
            return False

        try:
            # Initialize embeddings
            _self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", # Standard Gemini embedding model
                google_api_key=config.GOOGLE_API_KEY
            )

            # Check if vector store exists (now for FAISS)
            if os.path.exists(os.path.join(config.VECTOR_DB_PATH, "index.faiss")) and \
               os.path.exists(os.path.join(config.VECTOR_DB_PATH, "docstore.json")):
                st.info("Lade existierende Vektordatenbank (Faiss)...")
                _self.vector_store = FAISS.load_local(config.VECTOR_DB_PATH, _self.embeddings)
            else:
                st.info("Erstelle neue Vektordatenbank (Faiss) aus PDF-Korpus...")
                _self._create_vector_store()

            # Initialize QA chain
            _self._create_qa_chain()

            return True

        except Exception as e:
            st.error(f"Fehler bei der Initialisierung: {str(e)}")
            return False

    def _create_vector_store(self):
        """Create vector store from PDF corpus using FAISS"""
        documents = []

        # Check if PDF folder exists
        if not os.path.exists(config.PDF_FOLDER):
            os.makedirs(config.PDF_FOLDER)
            st.warning(f"PDF-Ordner '{config.PDF_FOLDER}' wurde erstellt. Bitte PDFs hinzufügen und App neu starten.")
            return

        # Load all PDFs
        pdf_files = [f for f in os.listdir(config.PDF_FOLDER) if f.endswith('.pdf')]

        if not pdf_files:
            st.warning(f"Keine PDF-Dateien in '{config.PDF_FOLDER}' gefunden!")
            return

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, pdf_file in enumerate(pdf_files):
            status_text.text(f"Verarbeite: {pdf_file}")

            try:
                pdf_path = os.path.join(config.PDF_FOLDER, pdf_file)
                loader = PyPDFLoader(pdf_path)
                pages = loader.load()

                # Split documents
                chunks = self.text_splitter.split_documents(pages)

                # Add metadata
                for chunk in chunks:
                    chunk.metadata['source'] = pdf_file

                documents.extend(chunks)

            except Exception as e:
                st.warning(f"Fehler beim Laden von {pdf_file}: {str(e)}")

            progress_bar.progress((i + 1) / len(pdf_files))

        if documents:
            # Create vector store using FAISS
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            # Save the FAISS index locally
            os.makedirs(config.VECTOR_DB_PATH, exist_ok=True)
            self.vector_store.save_local(config.VECTOR_DB_PATH)
            st.success(f"Vektordatenbank (Faiss) erstellt mit {len(documents)} Dokumentenabschnitten!")
        else:
            st.error("Keine Dokumente verarbeitet!")

    def _create_qa_chain(self):
        """Create QA chain"""
        if not self.vector_store:
            return

        # Custom prompt template
        template = """Nutze den folgenden Kontext, um die Frage zu beantworten.
        Wenn du die Antwort nicht aus dem Kontext ableiten kannst, sage ehrlich dass du es nicht weißt.
        Gib konkrete, hilfreiche Antworten basierend auf den bereitgestellten Dokumenten.

        Kontext: {context}

        Frage: {question}

        Antwort:"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro", # Or another Gemini model like "gemini-1.5-pro-latest"
            google_api_key=config.GOOGLE_API_KEY,
            temperature=0.3,   # Lower temperature for more focused answers
            convert_system_message_to_human=True # Often helpful for chat models in chains
        )

        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

    def query(self, question: str):
        """Query the RAG system"""
        if not self.qa_chain:
            return None, []

        try:
            result = self.qa_chain({"query": question})
            answer = result['result']
            sources = [doc.metadata.get('source', 'Unbekannt') for doc in result['source_documents']]
            return answer, list(set(sources))   # Remove duplicates

        except Exception as e:
            st.error(f"Fehler bei der Abfrage: {str(e)}")
            return None, []