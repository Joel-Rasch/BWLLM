import streamlit as st
import rag_system
from pathlib import Path

def check_setup():
    """Prüft ob das System richtig eingerichtet ist"""
    issues = []
    
    # Prüfe FAISS-Index
    if not Path("faiss_index").exists():
        issues.append("FAISS-Index fehlt")
    
    # Prüfe .env
    if not Path(".env").exists():
        issues.append(".env Datei fehlt")
    
    # Prüfe Markdown-Dateien
    if not Path("Extrahierter_Text_Markdown").exists():
        issues.append("Extrahierter_Text_Markdown Ordner fehlt")
    
    return issues

def get_rag_response(query):
    """Holt RAG-Antwort"""
    try:
        answer = rag_system.rag(query)
        return answer
    except Exception as e:
        return f"❌ Fehler: {e}"

def get_vector_search_results(query, k=10):
    """Holt nur die Vektorsuche-Ergebnisse für Debugging"""
    try:
        results = rag_system.query_faiss_index(query, k=k)
        return results
    except Exception as e:
        return [f"❌ Fehler: {e}"]

def chat_page():
    """Hauptseite - Chat Interface"""
    st.title("📊 Geschäftsbericht RAG-System")
    st.markdown("**Stelle Fragen zu den Geschäftsberichten!**")
    
    # Setup-Prüfung
    setup_issues = check_setup()
    if setup_issues:
        st.error("❌ System nicht richtig eingerichtet!")
        st.markdown("**Fehlende Komponenten:**")
        for issue in setup_issues:
            st.markdown(f"- {issue}")
        st.markdown("**Lösung:** Führe `python setup.py` aus, um das System einzurichten.")
        st.stop()
    
    # Sidebar mit Infos
    with st.sidebar:
        st.header("ℹ️ System-Info")
        
        # Prüfe verfügbare Dokumente
        markdown_dir = Path("Extrahierter_Text_Markdown")
        if markdown_dir.exists():
            md_files = list(markdown_dir.glob("*.md"))
            st.markdown(f"**📄 Verfügbare Dokumente:** {len(md_files)}")
            
            if md_files:
                st.markdown("**Dokumente:**")
                for md_file in md_files[:5]:  # Zeige max 5
                    st.markdown(f"- {md_file.stem}")
                if len(md_files) > 5:
                    st.markdown(f"... und {len(md_files) - 5} weitere")
        
        st.markdown("---")
        st.markdown("**💡 Beispiel-Fragen:**")
        st.markdown("- Wie hoch war der Umsatz in 2023?")
        st.markdown("- Welche Risiken werden genannt?")
        st.markdown("- Was sind die strategischen Ziele?")
        st.markdown("- Wie entwickelte sich das EBITDA?")

    # Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat-Verlauf anzeigen
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if prompt := st.chat_input("Frage zu den Geschäftsberichten..."):
        # User Message hinzufügen
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        # RAG Response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Durchsuche Dokumente..."):
                response = get_rag_response(prompt)
            st.markdown(response)

        # Assistant Response hinzufügen
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Clear Chat Button
    if st.session_state.messages:
        if st.button("🗑️ Chat löschen"):
            st.session_state.messages = []
            st.rerun()

def debug_page():
    """Debug-Seite - Vektorsuche testen"""
    st.title("🔍 Debug: Vektorsuche")
    st.markdown("**Teste die Vektorsuche direkt - zeigt die Top 10 Chunks ohne LLM-Verarbeitung**")
    
    # Setup-Prüfung
    setup_issues = check_setup()
    if setup_issues:
        st.error("❌ System nicht richtig eingerichtet!")
        st.markdown("**Fehlende Komponenten:**")
        for issue in setup_issues:
            st.markdown(f"- {issue}")
        st.markdown("**Lösung:** Führe `python setup.py` aus, um das System einzurichten.")
        st.stop()
    
    # Eingabefeld für Suchanfrage
    search_query = st.text_input(
        "🔍 Suchanfrage eingeben:",
        placeholder="z.B. Umsatz 2023, Risiken, Strategie, EBITDA...",
        help="Gib hier deine Suchanfrage ein, um die ähnlichsten Chunks zu finden"
    )
    
    # Anzahl der Ergebnisse
    col1, col2 = st.columns([3, 1])
    with col2:
        k_results = st.selectbox("Anzahl Ergebnisse:", [5, 10, 15, 20], index=1)
    
    if st.button("🚀 Suche starten", type="primary"):
        if search_query.strip():
            with st.spinner("🔍 Durchsuche Vektorindex..."):
                results = get_vector_search_results(search_query, k=k_results)
            
            if results and not results[0].startswith("❌"):
                st.success(f"✅ {len(results)} Ergebnisse gefunden!")
                
                # Statistiken
                text_chunks = [r for r in results if "[TEXT" in r]
                table_chunks = [r for r in results if "[TABLE" in r]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📄 Text-Chunks", len(text_chunks))
                with col2:
                    st.metric("📊 Tabellen-Chunks", len(table_chunks))
                with col3:
                    st.metric("🔢 Gesamt", len(results))
                
                st.markdown("---")
                
                # Ergebnisse anzeigen
                for i, result in enumerate(results, 1):
                    # Chunk-Typ und Quelle extrahieren
                    lines = result.split('\n', 1)
                    if len(lines) > 1:
                        header = lines[0]
                        content = lines[1]
                        
                        # Icon basierend auf Chunk-Typ
                        if "[TEXT" in header:
                            icon = "📄"
                            chunk_type = "Text"
                        elif "[TABLE" in header:
                            icon = "📊"
                            chunk_type = "Tabelle"
                        else:
                            icon = "📝"
                            chunk_type = "Unbekannt"
                        
                        with st.expander(f"{icon} **Ergebnis {i}** - {chunk_type} {header.replace('[', '').replace(']', '')}"):
                            st.markdown(f"**Quelle:** {header}")
                            st.markdown("**Inhalt:**")
                            st.markdown(content)
                            
                            # Zeichen-/Wort-Statistik
                            words = len(content.split())
                            chars = len(content)
                            st.caption(f"📊 {words} Wörter, {chars} Zeichen")
                    else:
                        with st.expander(f"📝 **Ergebnis {i}**"):
                            st.markdown(result)
            else:
                if results:
                    st.error(results[0])
                else:
                    st.warning("Keine Ergebnisse gefunden!")
        else:
            st.warning("Bitte gib eine Suchanfrage ein!")
    
    # Beispiel-Queries
    st.markdown("---")
    st.markdown("**💡 Beispiel-Suchanfragen zum Testen:**")
    
    examples = [
        "Umsatz 2023",
        "Risiken und Herausforderungen", 
        "Strategische Ziele",
        "EBITDA Entwicklung",
        "Nachhaltigkeit",
        "Digitalisierung",
        "Mitarbeiter",
        "Dividende"
    ]
    
    cols = st.columns(4)
    for i, example in enumerate(examples):
        with cols[i % 4]:
            if st.button(f"🔍 {example}", key=f"example_{i}"):
                st.session_state['example_query'] = example
                st.rerun()
    
    # Automatisches Setzen der Beispiel-Query
    if 'example_query' in st.session_state:
        st.session_state['search_input'] = st.session_state['example_query']
        del st.session_state['example_query']

def main():
    st.set_page_config(
        page_title="Geschäftsbericht RAG-System",
        page_icon="📊", 
        layout="wide"
    )
    
    # Sidebar Navigation
    with st.sidebar:
        st.title("📊 Navigation")
        page = st.radio(
            "Seite auswählen:",
            ["💬 Chat", "🔍 Debug"],
            help="Wähle zwischen Chat-Interface und Debug-Modus"
        )
        
        st.markdown("---")
        
        # System-Info in Sidebar
        if Path("faiss_index").exists():
            st.success("✅ FAISS-Index geladen")
        else:
            st.error("❌ FAISS-Index fehlt")
            
        if Path(".env").exists():
            st.success("✅ .env konfiguriert")
        else:
            st.error("❌ .env fehlt")
            
        markdown_dir = Path("Extrahierter_Text_Markdown")
        if markdown_dir.exists():
            md_count = len(list(markdown_dir.glob("*.md")))
            st.success(f"✅ {md_count} Dokumente")
        else:
            st.error("❌ Keine Dokumente")
    
    # Seiten-Routing
    if page == "💬 Chat":
        chat_page()
    elif page == "🔍 Debug":
        debug_page()

if __name__ == "__main__":
    main()