"""
Beispiel für die Nutzung des RAG-Systems mit tabellenreichen Dokumenten
"""
from rag_chunking import RAGEmbeddingSystem

def main():
    # Initialisiere das RAG System
    rag_system = RAGEmbeddingSystem()
    
    # Lade bereits erstellten Index (falls vorhanden)
    try:
        rag_system._load_index()
        print("Index erfolgreich geladen!")
    except FileNotFoundError:
        print("Kein Index gefunden. Führe erst 'python rag_chunking.py' aus.")
        return
    
    # Interaktive Suche
    print("\n" + "="*60)
    print("RAG-System für Continental Geschäftsbericht 2023")
    print("="*60)
    print("Beispiele für Suchanfragen:")
    print("- 'Umsatz 2023'")
    print("- 'EBIT Margin'") 
    print("- 'Anzahl Mitarbeiter'")
    print("- 'Automotive Kennzahlen'")
    print("- 'Dividende pro Aktie'")
    print("\nGib 'quit' ein um zu beenden.\n")
    
    while True:
        query = input("Deine Suche: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Auf Wiedersehen!")
            break
            
        if not query:
            continue
            
        print(f"\nSuche nach: '{query}'")
        print("-" * 50)
        
        # Suche mit niedrigerer Schwelle für mehr Ergebnisse
        results = rag_system.search(query, top_k=5, score_threshold=0.1)
        
        if not results:
            print("Keine relevanten Ergebnisse gefunden.")
            continue
            
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Relevanz: {result['score']:.3f} | Typ: {result['type']}")
            
            # Zeige mehr Kontext für bessere Ergebnisse
            content_preview = result['content'][:300] + "..." if len(result['content']) > 300 else result['content']
            print(f"   Inhalt: {content_preview}")
            
            # Zusätzliche Metadaten
            metadata = result['metadata']
            print(f"   Quelle: {metadata.get('source_file', 'Unbekannt')}")
            
            if metadata.get('has_table', False):
                print("   -> Enthält Tabellendaten")
                if metadata.get('is_table_part', False):
                    print("   -> Teil einer größeren Tabelle")
        
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()