"""
Main Streamlit app for the RAG system
"""
import streamlit as st
import sys
import os

# Add modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules import Config, DocumentProcessor, EmbeddingManager, QueryEnhancer, HybridRetriever, RAGSystem
from langchain_google_genai import ChatGoogleGenerativeAI

# Page configuration
st.set_page_config(
    page_title="RAG System für deutsche Geschäftsberichte",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'system_ready' not in st.session_state:
    st.session_state.system_ready = False

def initialize_system():
    """Initialize the RAG system"""
    
    try:
        # Load configuration
        config = Config()
        
        if not config.GOOGLE_API_KEY:
            st.error("❌ GOOGLE_API_KEY nicht gefunden. Bitte in .env Datei setzen.")
            return False
        
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            google_api_key=config.GOOGLE_API_KEY
        )
        
        # Initialize modules
        document_processor = DocumentProcessor(llm, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        embedding_manager = EmbeddingManager(config.EMBEDDING_MODEL)
        query_enhancer = QueryEnhancer(llm)
        
        # Load or create vector store
        if not embedding_manager.load_vectorstore(config.VECTOR_STORE_PATH):
            st.warning("Vector Store nicht gefunden. Erstelle neue...")
            
            if not os.path.exists(config.MARKDOWN_FOLDER):
                st.error(f"❌ Markdown Ordner nicht gefunden: {config.MARKDOWN_FOLDER}")
                return False
            
            with st.spinner("Verarbeite Dokumente..."):
                documents, stats = document_processor.process_all_documents(config.MARKDOWN_FOLDER)
                
                if not documents:
                    st.error("❌ Keine Dokumente gefunden")
                    return False
                
                embedding_manager.create_vectorstore(documents, config.VECTOR_STORE_PATH)
                st.success(f"✅ Vector Store erstellt mit {len(documents)} Dokumenten")
        else:
            st.success("✅ Vector Store geladen")
        
        # Initialize retriever and RAG system
        retriever = HybridRetriever(embedding_manager, query_enhancer)
        rag_system = RAGSystem(retriever, llm)
        
        st.session_state.rag_system = rag_system
        st.session_state.embedding_manager = embedding_manager
        st.session_state.system_ready = True
        
        return True
        
    except Exception as e:
        st.error(f"❌ Fehler beim Initialisieren: {str(e)}")
        return False

def main():
    """Main application"""
    
    st.title("🤖 RAG System für deutsche Geschäftsberichte")
    st.markdown("Stellen Sie Fragen zu den deutschen Automobilunternehmen und erhalten Sie Antworten basierend auf deren Geschäftsberichten.")   
    # Sidebar for system status
    with st.sidebar:
        st.header("System Status")
        
        if not st.session_state.system_ready:
            if st.button("🚀 System initialisieren"):
                initialize_system()
        else:
            st.success("✅ System bereit")
            
            # Show vector store stats
            if 'embedding_manager' in st.session_state:
                stats = st.session_state.embedding_manager.get_vectorstore_stats()
                st.info(f"📊 {stats.get('total_documents', 0):,} Dokumente geladen")
        
        st.markdown("---")
        st.markdown("**Verfügbare Unternehmen:**")
        st.markdown("• BMW • Continental • Daimler • Daimler Truck • KnorrBremse • Porsche • TRATON • VW")
    
    # Main content area
    if not st.session_state.system_ready:
        st.info("👆 Bitte System in der Sidebar initialisieren")
        return
    
    # Query input
    st.header("💬 Frage stellen")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_query = st.text_input(
            "Ihre Frage:",
            placeholder="z.B. Wie hat sich der Umsatz von BMW entwickelt?",
            help="Stellen Sie Fragen zu Finanzkennzahlen, Geschäftsentwicklung, etc."
        )
    
    with col2:
        ask_button = st.button("🔍 Frage stellen", type="primary")
    
    # Process query
    if ask_button and user_query:
        with st.spinner("🔍 Suche nach relevanten Informationen..."):
            result = st.session_state.rag_system.answer_question(user_query)
        
        # Display results
        st.header("💡 Antwort")
        st.write(result['answer'])
        
        # Show sources
        if result['sources']:
            st.subheader("📚 Quellen")
            st.write(", ".join(result['sources']))        
        # Show query variants used
        # Show query variants used
        with st.expander("🔄 Verwendete Suchanfragen"):
            st.write("Das System hat diese Anfragen verwendet:")
            for i, variant in enumerate(result['query_variants'], 1):
                st.write(f"{i}. {variant}")
        # Store result in session state for the chunks page
        st.session_state.last_result = result
        st.session_state.last_query = user_query
    
    # Recent queries (if any)
    if 'last_query' in st.session_state:
        st.markdown("---")
        st.subheader("🕒 Letzte Anfrage")
        st.write(f"**Frage:** {st.session_state.last_query}")

if __name__ == "__main__":
    main()
