import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness
)
from dotenv import load_dotenv
import rag_system
import variable_loader as loader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from ragas.run_config import RunConfig

load_dotenv()
loader.load_variables()

def get_rag_answer_with_context(question):
    companies, query_cleaned = rag_system.process_query(question)

    # Context aus dem FAISS laden und RAG-Antwort generieren
    context_docs = rag_system.query_faiss_index(query_cleaned, companies, k=5)
    answer_response = rag_system.rag(question=question)
    answer = answer_response.content
    context_list = [doc.page_content for doc in context_docs]
    
    return answer, context_list

def create_evaluation_dataset(test_questions):
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    # RAG-Antworten für alle Testfragen generieren
    for i, test_item in enumerate(test_questions):
        question = test_item["question"]
        ground_truth = test_item["ground_truth"]
        
        print(f"Verarbeite Frage {i+1}/{len(test_questions)}: {question[:60]}...")
        
        answer, context_list = get_rag_answer_with_context(question)
        
        questions.append(question)
        answers.append(answer)
        contexts.append(context_list)
        ground_truths.append(ground_truth)
    
    # Dataset für RAGAS aus RAG-Antworten und Testfragen erstellen
    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "retrieved_contexts": contexts,
        "ground_truth": ground_truths
    })

def run_ragas_evaluation():
    test_questions = [ 
    {
        "question": "Wie hoch war das Konzernergebnis vor Steuern von BMW im Jahr 2022?",
        "ground_truth": "Das Konzernergebnis vor Steuern von BMW betrug im Jahr 2022 23.509 Mio. €."
    },
    {
        "question": "Wie viele Gesamtauslieferungen hatte BMW über alle Marken im Jahr 2021?",
        "ground_truth": "Die Gesamtauslieferungen aller Marken von BMW betrugen 2021 2.521.514 Einheiten"
    },
    {
        "question": "Wie hoch war der Umsatz von BMW im Jahr 2021?",
        "ground_truth": "Der Umsatz von BMW betrug im Jahr 2021 118.909 Mio. €."
    },  
    {
        "question": "Wie hoch war der Konzernumsatz von BMW im Jahr 2021?",
        "ground_truth": "Der Konzernumsatz von BMW lag im Jahr 2021 bei 118.909 Mio. €."
    },
    {
        "question": "Wie hoch war der gesamte Energieverbrauch von BMW im Jahr 2022?",
        "ground_truth": "Der gesamte Energieverbrauch von BMW betrug im Jahr 2022 6.295.990 MWh."
    },
    {
        "question": "Was sind die vier Elemente der Strategie von BMW?",
        "ground_truth": "Die Elemente der Strategie von BMW sind: Positionierung, Ausrichtung, strategische Stoßrichtung und Zusammenarbeit."
    },
    {
        "question": "Wie viel Gewinn vor Steuern machte VW im Jahr 2023?",
        "ground_truth": "Der Gewinn vor Steuern von VW betrug im Jahr 2023 23.194 Millionen Euro."
    },
    {
        "question": "Wie hoch war das Ergebnis vor Steuern bei VW im Jahr 2023?",
        "ground_truth": "Das Ergebnis vor Steuern von VW betrug im Jahr 2023 23.194 Millionen Euro"
    },
    {
        "question": "Wie hoch waren die gesamten Personalkosten von VW im Jahr 2023?",
        "ground_truth": "Die gesamten Personalkosten von VW betrugen im Jahr 2023 49.755 Millionen Euro."
    },
    {
        "question": "Auf welchen Bereichen Lag für VW im Jahr 2023 ein besonderer Fokus im Rahmen des TOP 10 Programms?",
        "ground_truth": "Im Rahmen des TOP 10 Programms lag der Fokus von VW im Jahr 2023 auf den Bereichen finanzielle Robustheit und Planung, Produkte, die Regionen China und Nordamerika, Software, Technologien, Batterie und Laden, Mobilitätslösungen, Nachhaltigkeit und Kapitalmärkte."
    },
    {
        "question": "Wie hoch waren die Umsatzerlöse des VW Konzerns im Jahr 2022?",
        "ground_truth": "Die Umsatzerlöse des VW Konzerns betrugen im Jahr 2022 279.050 Mio. €."
    },
    {
        "question": "Wie hoch ist das Grundgehalt für Vorstandsmitglieder von VW?",
        "ground_truth": "Das Grundgehalt für Vorstandsmitglieder von VW liegt bei 1.500.000 Euro pro Jahr."
    },
    {
        "question": "Welches Ziel hat sich Daimler mit 'Ambition 2039' gesetzt?",
        "ground_truth": "Mit der 'Ambition 2039' hat sich Daimler bis 2039 das Ziel gesetzt, eine neutrale CO2-Bilanz für die Neufahrzeugflotte zu erreichen."
    },
    {
        "question": "Wie hoch waren die liquiden Mittel im Jahr 2023 bei Daimler?",
        "ground_truth": "Die liquiden Mittel von Daimler betrugen im Jahr 2023 22.830 Millionen Euro."
    },
    {
        "question": "Wie hoch war das Ergebnis vor Steuern bei Daimler im Jahr 2023?",
        "ground_truth": "Das Ergebnis vor Steuern von Daimler betrug im Jahr 2023 20.084 Millionen Euro."
    },
    {
        "question": "Wie hoch war die Dividende pro Aktie bei Daimler im Jahr 2022?",
        "ground_truth": "Die Dividende von Daimler betrug im Jahr 2022 5,20 Euro pro Aktie."
    },
    {
        "question": "Wie hoch war der Umsatz von Daimler im Jahr 2023?",
        "ground_truth": "Der Umsatz von Daimler betrug im Jahr 2023 153.218 Millionen Euro."
    },
    {
        "question": "Wie hoch waren die Pensionsverpflichtungen von Daimler zum 31. Dezember 2023?",
        "ground_truth": "Die Pensionsverpflichtungen von Daimler betrugen am 31. Dezember 2023 insgesamt 760 Millionen €"
    },
    {
        "question": "Wie viele Mitarbeiter hatte Continental zum 31.12.2023?",
        "ground_truth": "Continental hatte zum 31.12.2023 202.763 Mitarbeiter."
    },
    {
        "question": "Welche Technologien umfasst der Unternehmensbereich Automotive bei Continental?",
        "ground_truth": "Der Unternehmensbereich Automotive bei Continental umfasst Technologien für Sicherheits-, Brems-, Fahrwerk- sowie Bewegungs- und Bewegungskontrollsysteme."
    },
    {
        "question": "Wie hoch war das Ergebnis vor Ertragsteuern von Continental im Jahr 2023?",
        "ground_truth": "Das Ergebnis vor Ertragsteuern von Continental betrug im Jahr 2023 1.618 Millionen Euro."
    },
    {
        "question": "Wie hoch war der Umsatz von Continental im Jahr 2023?",
        "ground_truth": "Der Umsatz von Continental betrug im Jahr 2023 41.420,5 Millionen Euro."
    },
    {
        "question": "Wie hoch waren die gesamten CO2-Emissionen von Continental im Jahr 2023?",
        "ground_truth": "Die gesamten CO2-Emissionen von Continental betrugen im Jahr 2023 0,89 Mio t CO2."
    },
    {
        "question": "Wie hoch war die Dividende pro Aktie bei Continental im Jahr 2023?",
        "ground_truth": "Die Dividende pro Aktie bei Continental betrug im Jahr 2023 2,20 Euro."
    }]
    
    dataset = create_evaluation_dataset(test_questions)
    
    # LLM für die Evaluierung mit RAGAS konfigurieren
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # Embeddings für die Evaluierung mit RAGAS
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # RAGAS-Metriken
    metrics = [
        faithfulness,
        context_recall,
        context_precision,
        answer_correctness
    ]
    
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=1, timeout=240)  
    )
    
    print("\n---")
    print("RAGAS EVALUIERUNGSERGEBNISSE")
    print("---")
    
    df = result.to_pandas()
    # Durchschnittswerte der Metriken über alle Testfragen berechnen
    print(df.loc[:, [m.name for m in metrics]].mean())

    # Ergebnisse in CSV speichern
    output_file = "ragas_evaluation_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nErgebnisse gespeichert in: {output_file}")
    
    return result, df

if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("Fehler: GOOGLE_API_KEY nicht gefunden. Bitte in .env-Datei eintragen.")
    elif not os.path.exists("faiss_index"):
        print("Fehler: FAISS-Index-Ordner 'faiss_index' nicht gefunden.")
        print("Bitte zuerst den Index erstellen.")
    else:
        result, df = run_ragas_evaluation()