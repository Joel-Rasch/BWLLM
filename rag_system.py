import re
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain.vectorstores import FAISS

from typing import Optional
from dotenv import load_dotenv
#from extract_entities import extract_key_entities, delete_stopwords
import variable_loader as loader

def process_query(query):

    query_lower = query.lower()
    matched_companies = []

    # Firmen in Query suchen
    for company in loader.known_companies:
        if re.search(rf"\b{re.escape(company)}\b", query_lower):
            matched_companies.append(company)

    if not matched_companies:
        return {
            "error": "Bitte den Firmennamen in der query angeben"
        }

    # Bereinigte Query über extract_key_entities
    cleaned_query = query
    # cleaned_query = delete_stopwords(query)

    # Alle bekannten Firmennamen aus cleaned_query entfernen (case-insensitive, als ganze Wörter)
    for company in loader.known_companies:
        cleaned_query = re.sub(rf"\b{re.escape(company)}\b", "", cleaned_query, flags=re.IGNORECASE)

    # Überflüssige Leerzeichen entfernen
    cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()

    return matched_companies, cleaned_query

def query_faiss_index(query, companies, k=5):

    # Sicherstellen, dass companies eine Liste ist
    if not isinstance(companies, list):
        raise ValueError("companies muss eine Liste von Firmennamen sein")

    # Alle Firmennamen kleinschreiben für Vergleich
    companies = [c.lower() for c in companies]

    all_results = []

    # Pro Firma: Dokumente filtern → FAISS-Index → Similarity Search
    for company in companies:
        company_docs = [
            doc for doc in loader.all_docs
            if doc.metadata.get("company", "").lower() == company
        ]

        if not company_docs:
            print(f"Keine Chunks für die Firma '{company}' gefunden")
            continue

        # Temporären FAISS-Index für diese Firma erstellen
        temp_index = FAISS.from_documents(company_docs, embedding=loader.embeddings)

        # Similarity Search für diese Firma
        results = temp_index.similarity_search(query, k=k)
        all_results.extend(results)

    return all_results if all_results else None

def rag(question: Optional[str] = '', chat_history: Optional[str] = ''):

    load_dotenv()

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=1,
        max_tokens=None,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    companies, query_cleaned = process_query(question)
    similarities = query_faiss_index(query_cleaned, companies)

    # Kontext mit company-Metadaten formatieren
    context = []
    for result in similarities:
        company = result.metadata.get("company", "Unbekannt")
        content = result.page_content
        context.append(f"Firma: {company}\nInhalt: {content}")

    context_str = "\n\n---\n\n".join(context)

    prompt = f"""Du bist ein erfahrener Chat-Assistent, der Informationen aus dem zwischen <context> und </context> stehenden KONTEXT extrahiert.
                Beim Beantworten der Frage, die zwischen <question> und </question> steht,
                sei präzise und erfinde nichts dazu.
                Wenn du die Information nicht hast, sage das einfach.
                Beantworte die Frage nur, wenn du sie eindeutig aus dem bereitgestellten KONTEXT entnehmen kannst.

                Erwähne den verwendeten KONTEXT in deiner Antwort nicht.

                <context>          
                {context_str}
                </context>
                <question>  
                {question}
                </question>
                Antwort:"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response