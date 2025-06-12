### Code for the Rag to return a rag answer, based on a question

# Import necessary classes from langchain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from typing import Optional
import os
import re
# importing necessary functions from dotenv library
from dotenv import load_dotenv, dotenv_values 

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from pathlib import Path

def extract_keywords(query):
    """
    Extrahiert wichtige Schlüsselwörter aus der Frage für besseres Retrieval
    """
    # Stopwörter entfernen
    stopwords = {
        'wie', 'was', 'wo', 'wann', 'warum', 'wer', 'welche', 'welcher', 'welches',
        'ist', 'sind', 'war', 'waren', 'hat', 'haben', 'hatte', 'hatten',
        'der', 'die', 'das', 'den', 'dem', 'des', 'ein', 'eine', 'einen', 'einem',
        'und', 'oder', 'aber', 'doch', 'noch', 'auch', 'nur', 'schon',
        'in', 'auf', 'an', 'bei', 'mit', 'nach', 'vor', 'über', 'unter',
        'sich', 'sie', 'er', 'es', 'ich', 'du', 'wir', 'ihr'
    }
    
    # Text normalisieren
    words = re.findall(r'\b\w+\b', query.lower())
    
    # Wichtige Wörter filtern
    keywords = [word for word in words if word not in stopwords and len(word) > 2]
    
    # Zahlen und Jahre sind oft wichtig
    numbers = re.findall(r'\b\d{4}\b|\b\d+[.,]?\d*\b', query)
    keywords.extend(numbers)
    
    return keywords

def hybrid_search(query, index_path="faiss_index", model_name="sentence-transformers/all-MiniLM-L6-v2", k=10):
    """
    Hybrid-Suche: Kombiniert Keyword-basierte und semantische Suche
    """
    if not Path(index_path).exists():
        print(f"❌ FAISS-Index nicht gefunden: {index_path}")
        return []

    try:
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        faiss_index = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        
        # 1. Semantische Suche mit der ursprünglichen Frage
        semantic_results = faiss_index.similarity_search(query, k=k)
        
        # 2. Keyword-basierte Suche
        keywords = extract_keywords(query)
        keyword_query = " ".join(keywords)
        
        keyword_results = []
        if keywords:
            keyword_results = faiss_index.similarity_search(keyword_query, k=k//2)
        
        # 3. Ergebnisse kombinieren und Duplikate entfernen
        all_results = []
        seen_content = set()
        
        # Priorisiere semantische Ergebnisse
        for result in semantic_results:
            content_hash = hash(result.page_content[:100])  # Erste 100 Zeichen als Hash
            if content_hash not in seen_content:
                all_results.append(result)
                seen_content.add(content_hash)
        
        # Füge Keyword-Ergebnisse hinzu
        for result in keyword_results:
            content_hash = hash(result.page_content[:100])
            if content_hash not in seen_content and len(all_results) < k:
                all_results.append(result)
                seen_content.add(content_hash)
        
        # 4. Formatiere Ergebnisse
        context_chunks = []
        for result in all_results[:k]:
            chunk_type = result.metadata.get('chunk_type', 'unknown')
            source = result.metadata.get('source', 'unknown')
            
            formatted_chunk = f"[{chunk_type.upper()} aus {source}]\n{result.page_content}"
            context_chunks.append(formatted_chunk)
        
        print(f"🔍 Gefunden: {len(semantic_results)} semantische + {len(keyword_results)} keyword Ergebnisse")
        print(f"📊 Keywords: {', '.join(keywords)}")
        
        return context_chunks
    
    except Exception as e:
        print(f"❌ Fehler beim Laden des FAISS-Index: {e}")
        return []

def query_faiss_index(query, index_path="faiss_index", model_name="sentence-transformers/all-MiniLM-L6-v2", k=5):
    """
    Verbesserte FAISS-Suche mit mehreren Strategien
    """
    
    # Prüfe ob Index existiert
    if not Path(index_path).exists():
        print(f"❌ FAISS-Index nicht gefunden: {index_path}")
        print("Führe zuerst setup.py aus!")
        return []

    try:
        # Embeddings laden
        embeddings = HuggingFaceEmbeddings(model_name=model_name)

        # FAISS-Index laden
        faiss_index = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

        # Mehrere Suchstrategien
        all_results = []
        seen_content = set()
        
        # 1. Originale Frage
        results1 = faiss_index.similarity_search(query, k=k)
        
        # 2. Schlüsselwörter extrahieren und suchen
        keywords = extract_keywords(query)
        if keywords:
            keyword_query = " ".join(keywords)
            results2 = faiss_index.similarity_search(keyword_query, k=k//2)
        else:
            results2 = []
        
        # 3. Wichtige Begriffe einzeln suchen (besonders für Zahlen/Jahre)
        important_terms = [term for term in keywords if term.isdigit() or len(term) > 4]
        results3 = []
        for term in important_terms[:2]:  # Max 2 wichtige Begriffe
            term_results = faiss_index.similarity_search(term, k=2)
            results3.extend(term_results)
        
        # Alle Ergebnisse kombinieren
        for result_set in [results1, results2, results3]:
            for result in result_set:
                content_hash = hash(result.page_content[:100])
                if content_hash not in seen_content:
                    all_results.append(result)
                    seen_content.add(content_hash)
                    if len(all_results) >= k:
                        break
            if len(all_results) >= k:
                break

        # Text und Metadaten extrahieren
        context_chunks = []
        for result in all_results[:k]:
            chunk_type = result.metadata.get('chunk_type', 'unknown')
            source = result.metadata.get('source', 'unknown')
            
            # Formatiere Chunk mit Metadaten
            formatted_chunk = f"[{chunk_type.upper()} aus {source}]\n{result.page_content}"
            context_chunks.append(formatted_chunk)
        
        print(f"🔍 Suchstrategien: Original({len(results1)}) + Keywords({len(results2)}) + Terms({len(results3)})")
        print(f"📊 Extrahierte Keywords: {', '.join(keywords[:5])}")
        
        return context_chunks
    
    except Exception as e:
        print(f"❌ Fehler beim Laden des FAISS-Index: {e}")
        return []

def rag(question: Optional[str] = '', chat_history: Optional[str] = '', k: int = 8) -> str:
    """
    RAG-Funktion die eine Frage beantwortet basierend auf dem FAISS-Index
    """
    load_dotenv() 
    
    # Prüfe API Key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_google_api_key_here":
        return "❌ Google API Key nicht konfiguriert! Bitte .env Datei prüfen."
    
    # Hole relevanten Kontext mit verbesserter Suche
    context_chunks = query_faiss_index(question, k=k)
    
    if not context_chunks:
        return "❌ Keine relevanten Informationen im Dokumentenindex gefunden. Bitte führe setup.py aus oder versuche andere Suchbegriffe."
    
    # Kombiniere Kontext
    context = "\n\n".join(context_chunks)
    
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp", 
            temperature=0.1,
            max_tokens=None,
            google_api_key=api_key
        )

        prompt = f"""Du bist ein Experte für Finanzanalyse und Geschäftsberichte. 
Beantworte die Frage basierend auf dem bereitgestellten KONTEXT aus deutschen Geschäftsberichten.

WICHTIGE REGELN:
- Antworte nur basierend auf den bereitgestellten Informationen
- Wenn du die Information nicht im Kontext findest, sage das ehrlich
- Gib konkrete Zahlen und Fakten aus den Dokumenten wieder
- Erwähne nicht, dass du einen "Kontext" verwendest
- Antworte auf Deutsch
- Sei präzise und sachlich
- Bei Zahlenangaben nenne die Quelle (z.B. "laut Geschäftsbericht 2023")

KONTEXT:
{context}

FRAGE: {question}

Antwort: """

        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
        
    except Exception as e:
        return f"❌ Fehler bei der LLM-Anfrage: {e}"

