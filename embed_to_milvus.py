#!/usr/bin/env python3
"""
Milvus Local Embedding Script - University Demo

Embeds processed documents into local vector database with metadata
for improved retrieval in RAG systems.

Usage: python embed_to_milvus.py
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
import tiktoken


class DocumentEmbedder:
    def __init__(self, collection_name: str = "documents"):
        self.collection_name = collection_name
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.max_chunk_size = 512  # tokens
        self.db_file = "./milvus_local.db"  # Local database file
        
        # Connect to local Milvus
        self._connect_milvus()
        self._setup_collection()
    
    def _connect_milvus(self):
        """Connect to local Milvus instance"""
        try:
            # Use local file-based storage
            self.client = MilvusClient(uri=self.db_file)
            print("✓ Connected to local Milvus database")
            print(f"✓ Database file: {self.db_file}")
        except Exception as e:
            print(f"✗ Failed to connect to Milvus: {e}")
            raise
    
    def _setup_collection(self):
        """Create or connect to Milvus collection"""
        # Check if collection exists and drop it (for clean demo)
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
            print(f"✓ Dropped existing collection '{self.collection_name}'")
        
        # Create collection with schema
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=384,  # MiniLM embedding dimension
            metric_type="COSINE",
            auto_id=True
        )
        print(f"✓ Created collection '{self.collection_name}' with auto-indexing")
    
    def extract_metadata(self, filename: str) -> Dict[str, Any]:
        """Extract company, year, and document type from filename"""
        # Examples: "Continental_2023.pdf", "BMW_Geschaeftsbericht_2022.pdf"
        
        # Remove extension
        base_name = Path(filename).stem
        
        # Extract year (4 digits)
        year_match = re.search(r'\b(20\d{2})\b', base_name)
        year = int(year_match.group(1)) if year_match else 2023
        
        # Extract company name (first part before underscore or number)
        company_match = re.match(r'^([A-Za-z]+)', base_name)
        company = company_match.group(1) if company_match else "Unknown"
        
        # Determine document type
        doc_type = "enriched_tables" if "enriched" in filename.lower() else "basic_text"
        
        return {
            "company": company,
            "year": year,
            "document_type": doc_type,
            "source_file": filename
        }
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks that fit within token limit"""
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Check if adding this paragraph would exceed token limit
            test_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            tokens = self.tokenizer.encode(test_chunk)
            
            if len(tokens) <= self.max_chunk_size:
                current_chunk = test_chunk
            else:
                # Save current chunk if not empty
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Handle long paragraphs by splitting them
                if len(self.tokenizer.encode(paragraph)) > self.max_chunk_size:
                    # Split by sentences
                    sentences = re.split(r'[.!?]+', paragraph)
                    current_chunk = ""
                    
                    for sentence in sentences:
                        if not sentence.strip():
                            continue
                        
                        test_chunk = current_chunk + ". " + sentence if current_chunk else sentence
                        tokens = self.tokenizer.encode(test_chunk)
                        
                        if len(tokens) <= self.max_chunk_size:
                            current_chunk = test_chunk
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence.strip()
                else:
                    current_chunk = paragraph
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def embed_document(self, file_path: Path):
        """Embed a single document into Milvus"""
        print(f"Processing {file_path.name}...")
        
        # Read document
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"  ✗ Failed to read file: {e}")
            return
        
        # Extract metadata
        metadata = self.extract_metadata(file_path.name)
        
        # Chunk the document
        chunks = self.chunk_text(content)
        
        if not chunks:
            print(f"  ✗ No content to embed")
            return
        
        print(f"  Creating {len(chunks)} chunks...")
        
        # Generate embeddings
        embeddings = self.model.encode(chunks, convert_to_numpy=True)
        
        # Prepare data for insertion
        data = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            data.append({
                "vector": embedding.tolist(),
                "text": chunk[:1500],  # Keep within reasonable limits
                "company": metadata["company"],
                "year": metadata["year"],
                "document_type": metadata["document_type"],
                "chunk_index": i,
                "source_file": metadata["source_file"]
            })
        
        # Insert into Milvus
        try:
            self.client.insert(
                collection_name=self.collection_name,
                data=data
            )
            print(f"  ✓ Embedded {len(chunks)} chunks")
        except Exception as e:
            print(f"  ✗ Failed to insert: {e}")
    
    def embed_directory(self, directory: Path):
        """Embed all markdown files in directory"""
        md_files = list(directory.glob("*.md"))
        
        if not md_files:
            print(f"No markdown files found in {directory}")
            return
        
        print(f"Found {len(md_files)} markdown files to embed:")
        
        for file_path in md_files:
            self.embed_document(file_path)
        
        # Get total count
        try:
            stats = self.client.get_collection_stats(self.collection_name)
            total_entities = stats.get('row_count', 0)
        except:
            total_entities = "unknown"
        
        print(f"\n✓ Embedding complete!")
        print(f"Total chunks embedded: {total_entities}")
        print(f"Collection '{self.collection_name}' is ready for search")
    
    def test_search(self, query: str = "financial performance"):
        """Test search functionality"""
        print(f"\nTesting search with query: '{query}'")
        
        # Generate query embedding
        query_embedding = self.model.encode([query]).tolist()
        
        # Search
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                limit=3,
                output_fields=["text", "company", "year", "document_type", "source_file"]
            )
            
            print("Top 3 results:")
            for i, result in enumerate(results[0], 1):
                entity = result['entity']
                print(f"{i}. Company: {entity['company']}, Year: {entity['year']}")
                print(f"   Type: {entity['document_type']}, Score: {result['distance']:.3f}")
                print(f"   Text: {entity['text'][:100]}...")
                print()
                
        except Exception as e:
            print(f"Search failed: {e}")
            print("This is normal if no data has been embedded yet.")


def main():
    """Main function"""
    output_dir = Path("Extrahierter_Text_Markdown")
    
    if not output_dir.exists():
        print(f"Error: Output directory '{output_dir}' not found")
        print("Run main.py first to generate processed documents")
        return
    
    try:
        # Create embedder (uses local file database - no server needed!)
        embedder = DocumentEmbedder()
        
        # Embed all documents
        embedder.embed_directory(output_dir)
        
        # Test search
        embedder.test_search("Umsatz und Gewinn")
        embedder.test_search("tables and financial data")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have installed the requirements:")
        print("pip install -r requirements.txt")


if __name__ == "__main__":
    main()