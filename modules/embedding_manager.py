"""
Embedding module for vector store management
"""
import os
from typing import List
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embeddings = SentenceTransformerEmbeddings(model_name=model_name)
        self.vectorstore = None
    
    def create_vectorstore(self, documents: List[Document], save_path: str) -> FAISS:
        """Create and save vector store from documents"""
        if not documents:
            raise ValueError("No documents provided")
        
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        
        self.vectorstore.save_local(save_path)
        return self.vectorstore
    
    def load_vectorstore(self, load_path: str) -> bool:
        """Load existing vector store"""
        try:
            if (os.path.exists(load_path) and 
                os.path.exists(os.path.join(load_path, "index.faiss"))):
                
                self.vectorstore = FAISS.load_local(
                    load_path, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                return True
        except Exception as e:
            print(f"Error loading vector store: {e}")
        
        return False
    
    def get_vectorstore_stats(self) -> dict:
        """Get statistics about the vector store"""
        if not self.vectorstore:
            return {"status": "No vector store loaded"}
        
        try:
            total_vectors = self.vectorstore.index.ntotal
            vector_dimension = self.vectorstore.index.d
            
            return {
                "status": "loaded",
                "total_documents": total_vectors,
                "vector_dimension": vector_dimension,
                "embedding_model": "all-MiniLM-L6-v2"
            }
        except:
            return {"status": "Error getting stats"}
