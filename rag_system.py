### Code for the Rag to return a rag anser, based on a question

# Import necessary classes from langchain
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from typing import Optional
import os
# importing necessary functions from dotenv library
from dotenv import load_dotenv, dotenv_values 

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def get_company(query, index_path="faiss_index", model_name="sentence-transformers/all-MiniLM-L6-v2"):
    # Embedding-Modell und Index laden
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    faiss_index = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    all_docs = faiss_index.docstore._dict.values()
    
    # Alle bekannten Firmennamen aus Metadaten extrahieren
    known_companies = set(doc.metadata.get("company", "").lower() for doc in all_docs)

    # Original-Query unverändert speichern
    original_query = query
    cleaned_query = query
    query_lower = query.lower()
    matched_company = None

    # Firmennamen suchen und entfernen
    for company in known_companies:
        if re.search(rf"\b{re.escape(company)}\b", query_lower):
            matched_company = company
            cleaned_query = re.sub(rf"\b{re.escape(company)}\b", "", cleaned_query, flags=re.IGNORECASE)
            break  # nur die erste gefundene Firma berücksichtigen

    # Überflüssige Leerzeichen entfernen
    cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()

    return original_query, cleaned_query, matched_company

def query_faiss_index(query, company, index_path="faiss_index", model_name="sentence-transformers/all-MiniLM-L6-v2", k=7):

    # 1. Embedding-Modell laden
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # 2. FAISS-Index mit Metadaten laden
    faiss_index = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    # 3. Alle Metadaten durchsuchen, um bekannte Firmen zu sammeln
    all_docs = faiss_index.docstore._dict.values()

    # 5. Nur Dokumente dieser Firma auswählen
    filtered_docs = [doc for doc in all_docs if doc.metadata.get("company", "").lower() == company]

    if not filtered_docs:
        print("Keine Chunks für die Firma gefunden")
        return None

    # 6. Temporären FAISS-Index mit diesen Dokumenten erstellen
    filtered_index = FAISS.from_documents(filtered_docs, embedding=embeddings)

    # 7. Similarity Search
    results = filtered_index.similarity_search(query, k=k)

    # 8. Ergebnisse zurückgeben
    return results

def rag(question: Optional[str] = '', chat_history: Optional[str] = '') -> str:
    load_dotenv() 
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=1,max_tokens=None,google_api_key=os.getenv("GOOGLE_API_KEY"))

    query_org, query_cleaned, company = get_company(question)

    similarities = query_faiss_index(query_cleaned, company)
    context = [result.page_content for result in similarities]
    
    prompt = f"""Du bist ein erfahrener Chat-Assistent, der Informationen aus dem zwischen <context> und </context> stehenden KONTEXT extrahiert.
            Beim Beantworten der Frage, die zwischen <question> und </question> steht,
            sei präzise und erfinde nichts dazu.
            Wenn du die Information nicht hast, sage das einfach.
            Beantworte die Frage nur, wenn du sie eindeutig aus dem bereitgestellten KONTEXT entnehmen kannst.

            Erwähne den verwendeten KONTEXT in deiner Antwort nicht.

            <context>          
            {context}
            </context>
            <question>  
            {question}
            </question>
            Antwort: """

    response = llm.invoke([HumanMessage(content=prompt)])
    return response

