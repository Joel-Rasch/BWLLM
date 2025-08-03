import streamlit as st
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Import our RAG system
from src.rag.retriever import RAGSystem
from src.vector.embedder import get_index_stats
from src.config import INDEX_PATH, METADATA_PATH, EMBEDDING_MODEL, GEMINI_MODEL

# Page configuration
st.set_page_config(
    page_title="RAG System für Geschäftsberichte",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .process-step {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .success-step {
        background-color: #d4edda;
        border-left-color: #28a745;
    }
    .warning-step {
        background-color: #fff3cd;
        border-left-color: #ffc107;
    }
    .error-step {
        background-color: #f8d7da;
        border-left-color: #dc3545;
    }
    .source-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
    }
    .metric-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'process_steps' not in st.session_state:
        st.session_state.process_steps = []
    if 'last_query_stats' not in st.session_state:
        st.session_state.last_query_stats = None

def add_process_step(step: str, status: str = "info", details: str = ""):
    """Add a process step to the tracking"""
    step_data = {
        "step": step,
        "status": status,
        "details": details,
        "timestamp": time.time()
    }
    st.session_state.process_steps.append(step_data)

def display_process_steps():
    """Display the process steps with status indicators"""
    if st.session_state.process_steps:
        st.subheader("🔄 Prozess-Verlauf")
        
        for i, step in enumerate(st.session_state.process_steps):
            status_class = f"{step['status']}-step" if step['status'] != "info" else ""
            status_emoji = {
                "success": "✅",
                "warning": "⚠️", 
                "error": "❌",
                "info": "ℹ️"
            }.get(step['status'], "ℹ️")
            
            st.markdown(f"""
            <div class="process-step {status_class}">
                <strong>{status_emoji} Schritt {i+1}: {step['step']}</strong>
                {f"<br><small>{step['details']}</small>" if step['details'] else ""}
            </div>
            """, unsafe_allow_html=True)

def check_system_status():
    """Check if the RAG system files exist and are ready"""
    index_path = Path(INDEX_PATH)
    metadata_path = Path(METADATA_PATH)
    
    if not index_path.exists():
        return False, "FAISS Index nicht gefunden. Bitte führen Sie zuerst 'python main.py' aus."
    
    if not metadata_path.exists():
        return False, "Metadata-Datei nicht gefunden. Bitte führen Sie zuerst 'python main.py' aus."
    
    return True, "System bereit"

@st.cache_resource
def initialize_rag_system(api_key: str, company_filter: str = None, year_filter: str = None):
    """Initialize the RAG system (cached)"""
    try:
        add_process_step("RAG System wird initialisiert...", "info")
        
        rag_system = RAGSystem(
            index_path=INDEX_PATH,
            metadata_path=METADATA_PATH,
            model_name=EMBEDDING_MODEL,
            gemini_model=GEMINI_MODEL,
            k=5,
            api_key=api_key,
            company_filter=company_filter if company_filter != "Alle" else None,
            year_filter=year_filter if year_filter != "Alle" else None
        )
        
        add_process_step("RAG System erfolgreich initialisiert", "success")
        return rag_system, None
        
    except Exception as e:
        add_process_step("Fehler beim Initialisieren des RAG Systems", "error", str(e))
        return None, str(e)

def get_system_statistics():
    """Get statistics about the system"""
    try:
        stats = get_index_stats(INDEX_PATH, METADATA_PATH)
        return stats
    except Exception as e:
        st.error(f"Fehler beim Laden der Statistiken: {e}")
        return None

def display_system_overview():
    """Display system overview and statistics"""
    st.subheader("📊 System-Übersicht")
    
    stats = get_system_statistics()
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Gesamt Dokumente", stats['total_entities'])
        
        with col2:
            st.metric("Unternehmen", len(stats['company_distribution']))
        
        with col3:
            st.metric("Jahre", len(stats['year_distribution']))
        
        with col4:
            st.metric("Dimension", stats['dimension'])
        
        # Company distribution chart
        if stats['company_distribution']:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Verteilung nach Unternehmen")
                company_df = pd.DataFrame(
                    list(stats['company_distribution'].items()),
                    columns=['Unternehmen', 'Anzahl Chunks']
                )
                fig_company = px.pie(
                    company_df, 
                    values='Anzahl Chunks', 
                    names='Unternehmen',
                    title="Dokumente pro Unternehmen"
                )
                st.plotly_chart(fig_company, use_container_width=True)
            
            with col2:
                st.subheader("Verteilung nach Jahren")
                year_df = pd.DataFrame(
                    list(stats['year_distribution'].items()),
                    columns=['Jahr', 'Anzahl Chunks']
                )
                fig_year = px.bar(
                    year_df,
                    x='Jahr',
                    y='Anzahl Chunks',
                    title="Dokumente pro Jahr"
                )
                st.plotly_chart(fig_year, use_container_width=True)

def display_query_results(result: Dict[str, Any], use_expanders: bool = True):
    """Display query results with source information"""
    st.subheader("💡 Antwort")
    
    # Show applied intelligent filters if any
    if result.get('applied_filters', {}).get('intelligent_filtering_used', False):
        applied_filters = result['applied_filters']
        filter_info = []
        if applied_filters.get('companies'):
            filter_info.append(f"Unternehmen: {', '.join(applied_filters['companies'])}")
        if applied_filters.get('years'):
            filter_info.append(f"Jahre: {', '.join(applied_filters['years'])}")
        
        if filter_info:
            st.info(f"🎯 **Intelligente Filter angewendet:** {' | '.join(filter_info)}")
    
    # Main answer
    st.markdown(f"""
    <div style="background-color: #e3f2fd; padding: 1.5rem; border-radius: 0.5rem; margin: 1rem 0;">
        {result['answer']}
    </div>
    """, unsafe_allow_html=True)
    
    # Sources section
    if result['sources']:
        st.subheader(f"📚 Quellen ({result['num_sources']} gefunden)")
        
        # Create source statistics
        source_stats = {}
        for source in result['sources']:
            company = source.get('company', 'Unbekannt')
            if company not in source_stats:
                source_stats[company] = {'count': 0, 'avg_score': 0, 'years': set()}
            source_stats[company]['count'] += 1
            source_stats[company]['avg_score'] += source.get('score', 0)
            source_stats[company]['years'].add(source.get('year', 'Unbekannt'))
        
        # Calculate averages
        for company in source_stats:
            source_stats[company]['avg_score'] /= source_stats[company]['count']
            source_stats[company]['years'] = list(source_stats[company]['years'])
        
        # Display source overview
        st.subheader("📈 Quellen-Übersicht")
        overview_df = pd.DataFrame([
            {
                'Unternehmen': company,
                'Anzahl Quellen': data['count'],
                'Durchschnittlicher Score': round(data['avg_score'], 3),
                'Jahre': ', '.join(map(str, data['years']))
            }
            for company, data in source_stats.items()
        ])
        st.dataframe(overview_df, use_container_width=True)
        
        # Detailed sources
        st.subheader("🔍 Detaillierte Quellen")
        for i, source in enumerate(result['sources'], 1):
            # Get source data with safe defaults
            company = source.get('company', 'Unbekannt')
            year = source.get('year', 'Unbekannt')
            score = source.get('score', 0.0)
            chunk_type = source.get('chunk_type', 'text')
            source_file = source.get('source_file', 'Unbekannt')
            content_preview = source.get('content_preview', source.get('content', 'Kein Inhalt verfügbar'))
            
            if use_expanders:
                with st.expander(f"Quelle {i}: {company} ({year}) - Score: {score:.3f}"):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.write(f"**Unternehmen:** {company}")
                        st.write(f"**Jahr:** {year}")
                        st.write(f"**Relevanz-Score:** {score:.3f}")
                        st.write(f"**Typ:** {chunk_type}")
                        st.write(f"**Datei:** {source_file}")
                    
                    with col2:
                        st.write("**Inhalt:**")
                        st.text_area(
                            "Quellen-Inhalt",
                            value=content_preview,
                            height=150,
                            key=f"source_{hash(str(content_preview))}_{i}",
                            disabled=True
                        )
            else:
                # Display without expanders for nested contexts
                st.markdown(f"**Quelle {i}: {company} ({year}) - Score: {score:.3f}**")
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.write(f"**Unternehmen:** {company}")
                    st.write(f"**Jahr:** {year}")
                    st.write(f"**Relevanz-Score:** {score:.3f}")
                    st.write(f"**Typ:** {chunk_type}")
                    st.write(f"**Datei:** {source_file}")
                
                with col2:
                    st.write("**Inhalt:**")
                    st.text_area(
                        "Quellen-Inhalt",
                        value=content_preview,
                        height=150,
                        key=f"source_history_{hash(str(content_preview))}_{i}_{len(st.session_state.chat_history)}",
                        disabled=True
                    )
                st.divider()

def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">📊 RAG System für Deutsche Geschäftsberichte</h1>', unsafe_allow_html=True)
    
    # Check system status
    system_ready, status_message = check_system_status()
    
    if not system_ready:
        st.error(f"❌ {status_message}")
        st.info("💡 Führen Sie zuerst das Pipeline-Setup aus: `python main.py`")
        st.stop()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Einstellungen")
        
        # API Key input
        api_key = st.text_input(
            "🔑 Google Gemini API Key",
            type="password",
            help="Geben Sie Ihren Google Gemini API Key ein"
        )
        
        if not api_key:
            st.warning("⚠️ Bitte geben Sie einen API Key ein")
        
        st.divider()
        
        # Filters
        st.subheader("🔍 Filter")
        
        # Get available companies and years
        stats = get_system_statistics()
        companies = ["Alle"] + list(stats['company_distribution'].keys()) if stats else ["Alle"]
        years = ["Alle"] + list(stats['year_distribution'].keys()) if stats else ["Alle"]
        
        company_filter = st.selectbox("Unternehmen", companies)
        year_filter = st.selectbox("Jahr", years)
        
        st.divider()
        
        # Advanced settings
        with st.expander("🔧 Erweiterte Einstellungen"):
            k_value = st.slider("Anzahl Quellen", 1, 10, 5)
            gemini_model = st.selectbox(
                "Gemini Modell",
                ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
            )
            show_process = st.checkbox("Prozess-Schritte anzeigen", value=True)
        
        st.divider()
        
        # Clear history button
        if st.button("🗑️ Verlauf löschen"):
            st.session_state.chat_history = []
            st.session_state.process_steps = []
            st.rerun()
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Statistiken", "📖 Hilfe"])
    
    with tab1:
        # Chat interface
        st.subheader("💬 Stellen Sie Ihre Frage")
        
        # Query input
        query = st.text_input(
            "Ihre Frage:",
            placeholder="z.B. Wie hoch war der Umsatz von Continental im Jahr 2023?",
            key="query_input"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            search_button = st.button("🔍 Suchen", type="primary")
        
        # Process query
        if search_button and query and api_key:
            # Clear previous process steps
            st.session_state.process_steps = []
            
            # Initialize RAG system
            add_process_step("System wird vorbereitet...", "info")
            
            with st.spinner("RAG System wird initialisiert..."):
                rag_system, error = initialize_rag_system(api_key, company_filter, year_filter)
            
            if error:
                st.error(f"❌ Fehler beim Initialisieren: {error}")
                st.stop()
            
            # Process query
            add_process_step("Frage wird verarbeitet...", "info")
            add_process_step("Relevante Dokumente werden gesucht...", "info")
            
            with st.spinner("Antwort wird generiert..."):
                try:
                    # Query the system
                    result = rag_system.query(query)
                    
                    add_process_step(f"Suche abgeschlossen - {result['num_sources']} Quellen gefunden", "success")
                    add_process_step("Antwort mit Gemini generiert", "success")
                    
                    # Store in chat history
                    st.session_state.chat_history.append({
                        'query': query,
                        'result': result,
                        'timestamp': time.time(),
                        'filters': {
                            'company': company_filter,
                            'year': year_filter
                        }
                    })
                    
                    # Display results
                    display_query_results(result)
                    
                except Exception as e:
                    add_process_step("Fehler bei der Verarbeitung", "error", str(e))
                    st.error(f"❌ Fehler: {e}")
        
        elif search_button and not api_key:
            st.warning("⚠️ Bitte geben Sie einen API Key ein")
        
        # Show process steps if enabled
        if show_process and st.session_state.process_steps:
            with st.expander("🔄 Prozess-Details", expanded=False):
                display_process_steps()
        
        # Chat history
        if st.session_state.chat_history:
            st.divider()
            st.subheader("📝 Verlauf")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(f"Frage {len(st.session_state.chat_history)-i}: {chat['query'][:100]}..."):
                    st.write(f"**Filter:** {chat['filters']['company']} | {chat['filters']['year']}")
                    st.write(f"**Zeit:** {time.strftime('%H:%M:%S', time.localtime(chat['timestamp']))}")
                    display_query_results(chat['result'], use_expanders=False)
    
    with tab2:
        # Statistics tab
        display_system_overview()
    
    with tab3:
        # Help tab
        st.subheader("📖 Bedienungsanleitung")
        
        st.markdown("""
        ### 🚀 Erste Schritte
        
        1. **API Key eingeben**: Geben Sie Ihren Google Gemini API Key in der Seitenleiste ein
        2. **Filter setzen**: Wählen Sie optional ein Unternehmen oder Jahr aus
        3. **Frage stellen**: Geben Sie Ihre Frage in das Textfeld ein
        4. **Ergebnis ansehen**: Die Antwort wird mit Quellen angezeigt
        
        ### 💡 Beispiel-Fragen
        
        - "Wie hoch war der Umsatz von Continental 2023?" *(intelligente Filter: Continental + 2023)*
        - "Was waren die wichtigsten Geschäftszahlen von VW im Jahr 2022?" *(intelligente Filter: Volkswagen + 2022)*
        - "Zeige mir die Finanzkennzahlen von BMW für 2023" *(intelligente Filter: BMW + 2023)*
        - "Welche Ausgaben hatte Mercedes-Benz 2021?" *(intelligente Filter: Mercedes + 2021)*
        - "Zeige mir wichtige Kennzahlen aus den Geschäftsberichten" *(keine Filter)*
        
        ### 🔍 Filter verwenden
        
        **Manuelle Filter:**
        - **Unternehmen**: Beschränkt die Suche auf ein bestimmtes Unternehmen
        - **Jahr**: Beschränkt die Suche auf ein bestimmtes Jahr
        - **Anzahl Quellen**: Legt fest, wie viele Quellen zurückgegeben werden
        
        **🎯 Intelligente Filter (NEU!):**
        - Das System erkennt automatisch Unternehmensnamen und Jahre in Ihrer Frage
        - Filtert automatisch nach den erkannten Entitäten für präzisere Ergebnisse
        - Funktioniert auch mit Abkürzungen (z.B. "VW" für "Volkswagen")
        - Zeigt angewendete Filter transparent in den Ergebnissen an
        
        ### 🔄 Prozess-Transparenz
        
        Das System zeigt Ihnen transparent, was im Hintergrund passiert:
        
        1. **System-Initialisierung**: RAG System wird mit Ihren Einstellungen geladen
        2. **Dokumenten-Suche**: Relevante Dokumente werden im FAISS Index gesucht
        3. **Kontext-Aufbereitung**: Gefundene Dokumente werden formatiert
        4. **Antwort-Generierung**: Gemini LLM generiert die Antwort basierend auf dem Kontext
        5. **Ergebnis-Präsentation**: Antwort und Quellen werden angezeigt
        
        ### ⚙️ Technische Details
        
        - **Vector Database**: FAISS für schnelle Similaritätssuche
        - **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
        - **LLM**: Google Gemini für Antwort-Generierung
        - **Datenquelle**: Deutsche Geschäftsberichte (PDF)
        """)

if __name__ == "__main__":
    main()