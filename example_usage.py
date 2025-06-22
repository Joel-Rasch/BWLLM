"""
Example usage of the modular RAG system
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules import Config, DocumentProcessor, EmbeddingManager, QueryEnhancer, HybridRetriever, RAGSystem
from langchain_google_genai import ChatGoogleGenerativeAI

def main():
    """Example of using the modular RAG system"""
    
    # Initialize system (same as setup_rag.py but simpler)
    config = Config()
    
    if not config.GOOGLE_API_KEY:
        print("Please set GOOGLE_API_KEY in your .env file")
        return
    
    # Initialize components
    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        google_api_key=config.GOOGLE_API_KEY
    )
    
    embedding_manager = EmbeddingManager(config.EMBEDDING_MODEL)
    query_enhancer = QueryEnhancer(llm)
    
    # Load vector store (assuming it exists)
    if not embedding_manager.load_vectorstore(config.VECTOR_STORE_PATH):
        print("Vector store not found. Please run setup_rag.py first.")
        return
    
    # Create RAG system
    retriever = HybridRetriever(embedding_manager, query_enhancer)
    rag_system = RAGSystem(retriever, llm)
    
    # Example questions
    questions = [
        "Wie hat sich der Umsatz von BMW entwickelt?",
        "Was sind die wichtigsten Kennzahlen von Volkswagen?",
        "Welche Unternehmen berichten über Elektromobilität?"
    ]
    
    print("🤖 RAG System Example")
    print("=" * 50)
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        print("-" * 30)
        
        # Get answer
        result = rag_system.answer_question(question)
        
        print(f"💡 Answer: {result['answer']}")
        print(f"📚 Sources: {', '.join(result['sources'])}")
        
        # Show query variants
        print(f"🔄 Query variants used:")
        for i, variant in enumerate(result['query_variants'], 1):
            print(f"  {i}. {variant}")
        
        # Show retrieved chunks summary
        print(f"📄 Retrieved {len(result['retrieved_chunks'])} chunks")
        
        print("\n" + "=" * 50)

if __name__ == "__main__":
    main()
