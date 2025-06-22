"""
Setup script for the modular RAG system
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules import Config, DocumentProcessor, EmbeddingManager, QueryEnhancer, HybridRetriever, RAGSystem
from langchain_google_genai import ChatGoogleGenerativeAI

def setup_rag_system():
    """Setup the complete RAG system"""
    
    print("🚀 Setting up RAG system...")
    
    # Load configuration
    config = Config()
    
    if not config.GOOGLE_API_KEY:
        print("❌ Error: GOOGLE_API_KEY not found in environment variables")
        print("Please create a .env file with: GOOGLE_API_KEY=your_key_here")
        return None
    
    print("✅ Configuration loaded")
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        google_api_key=config.GOOGLE_API_KEY
    )
    print("✅ LLM initialized")
    
    # Initialize modules
    document_processor = DocumentProcessor(llm, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    embedding_manager = EmbeddingManager(config.EMBEDDING_MODEL)
    query_enhancer = QueryEnhancer(llm)
    
    print("✅ Modules initialized")
    
    # Load or create vector store
    if not embedding_manager.load_vectorstore(config.VECTOR_STORE_PATH):
        print("📁 Vector store not found, creating new one...")
        
        if not os.path.exists(config.MARKDOWN_FOLDER):
            print(f"❌ Error: Markdown folder not found: {config.MARKDOWN_FOLDER}")
            return None
        
        print("📄 Processing documents...")
        documents, stats = document_processor.process_all_documents(config.MARKDOWN_FOLDER)
        
        if not documents:
            print("❌ Error: No documents found")
            return None
        
        print(f"📊 Processed {stats['files_processed']} files, {stats['total_chunks']} chunks, {stats['total_tables']} tables")
        
        print("🔗 Creating vector store...")
        embedding_manager.create_vectorstore(documents, config.VECTOR_STORE_PATH)
        
    else:
        print("✅ Vector store loaded")
    
    # Initialize retriever and RAG system
    retriever = HybridRetriever(embedding_manager, query_enhancer)
    rag_system = RAGSystem(retriever, llm)
    
    print("✅ RAG system ready!")
    
    return rag_system

def test_rag_system(rag_system):
    """Test the RAG system with a simple query"""
    
    print("\n🧪 Testing RAG system...")
    
    test_query = "Wie hat sich der Umsatz von BMW entwickelt?"
    print(f"Test query: {test_query}")
    
    result = rag_system.answer_question(test_query)
    
    print(f"\n💡 Answer: {result['answer'][:200]}...")
    print(f"📚 Sources: {', '.join(result['sources'])}")
    print(f"🔄 Query variants used: {len(result['query_variants'])}")
    print(f"📄 Retrieved chunks: {len(result['retrieved_chunks'])}")
    
    return result

if __name__ == "__main__":
    # Setup system
    rag_system = setup_rag_system()
    
    if rag_system:
        # Test system
        test_result = test_rag_system(rag_system)
        
        print("\n🎉 Setup complete! You can now:")
        print("1. Run the Streamlit app: streamlit run streamlit_app.py")
        print("2. Use the system in Python scripts")
        print("3. Access the modules individually")
    else:
        print("\n❌ Setup failed. Please check the errors above.")
