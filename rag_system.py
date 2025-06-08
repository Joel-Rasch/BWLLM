### Code for the Rag to return a rag anser, based on a question

# Import necessary classes from langchain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from typing import Optional
import os
# importing necessary functions from dotenv library
from dotenv import load_dotenv, dotenv_values 

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def query_faiss_index(query, index_path="faiss_index", model_name="sentence-transformers/all-MiniLM-L6-v2", k=3):
    """
    Lädt einen FAISS-Index und gibt die Top-k relevantesten Text-Chunks als Liste von Strings zurück.

    Args:
        query (str): Die Suchanfrage in natürlicher Sprache.
        index_path (str): Pfad zum gespeicherten FAISS-Index.
        model_name (str): HuggingFace-Modellname für das Embedding.
        k (int): Anzahl der zurückzugebenden ähnlichen Ergebnisse.

    Returns:
        list of str: Liste der gefundenen Text-Chunks.
    """

    # Embeddings laden
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # FAISS-Index laden
    faiss_index = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    # Ähnliche Dokumente suchen
    results = faiss_index.similarity_search(query, k=k)

    # Nur den Text extrahieren
    return [result.page_content for result in results]

def rag(question: Optional[str] = '', chat_history: Optional[str] = '', context: Optional[str] = '') -> str:
    load_dotenv() 
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=1,max_tokens=None,google_api_key=os.getenv("GOOGLE_API_KEY"))

    prompt = f"""You are an expert chat assistance that extracs information from the CONTEXT provided
           between <context> and </context> tags.
           When ansering the question contained between <question> and </question> tags
           be concise and do not hallucinate. 
           If you don't have the information just say so.
           Only anwer the question if you can extract it from the CONTEXT provideed.
           
           Do not mention the CONTEXT used in your answer.
    
           <context>          
           {context}
           </context>
           <question>  
           {question}
           </question>
           Answer: """

    response = llm.invoke([HumanMessage(content=prompt)])
    return response

