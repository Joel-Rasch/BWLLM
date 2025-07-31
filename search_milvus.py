#!/usr/bin/env python3
"""
Milvus Lite Search Script - University Demo

Simple script to search the Milvus Lite vector database with metadata filtering.

Usage: python search_milvus.py
"""

from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
from typing import List, Dict, Any, Optional


class DocumentSearcher:
    def __init__(self, collection_name: str = "documents"):
        self.collection_name = collection_name
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.db_file = "./milvus_local.db"
        
        # Connect to local Milvus
        self.client = MilvusClient(uri=self.db_file)
        
        # Check if collection exists
        if not self.client.has_collection(collection_name):
            print(f"✗ Collection '{collection_name}' not found")
            print("Run embed_to_milvus.py first to create and populate the database")
            raise FileNotFoundError("Collection not found")
        
        stats = self.client.get_collection_stats(collection_name)
        
        print(f"✓ Connected to Milvus Lite collection '{collection_name}'")
        print(f"Total documents: {stats.get('row_count', 0)}")
    
    def search(self, 
               query: str, 
               company: Optional[str] = None,
               year: Optional[int] = None,
               doc_type: Optional[str] = None,
               limit: int = 5) -> List[Dict[str, Any]]:
        """Search documents with optional metadata filters"""
        
        # Generate query embedding
        query_embedding = self.model.encode([query]).tolist()
        
        # Build filter expression
        filters = []
        if company:
            filters.append(f'company == "{company}"')
        if year:
            filters.append(f'year == {year}')
        if doc_type:
            filters.append(f'document_type == "{doc_type}"')
        
        filter_expr = " and ".join(filters) if filters else None
        
        # Perform search with Milvus Lite
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=limit,
            filter=filter_expr,
            output_fields=["text", "company", "year", "document_type", "source_file", "chunk_index"]
        )
        
        # Format results
        formatted_results = []
        for result in results[0]:
            formatted_results.append({
                "text": result['entity']['text'],
                "company": result['entity']['company'],
                "year": result['entity']['year'],
                "document_type": result['entity']['document_type'],
                "source_file": result['entity']['source_file'],
                "chunk_index": result['entity']['chunk_index'],
                "score": result['distance']
            })
        
        return formatted_results
    
    def get_available_metadata(self) -> Dict[str, Any]:
        """Get available companies, years, and document types"""
        # Query for unique values using Milvus Lite
        try:
            results = self.client.query(
                collection_name=self.collection_name,
                filter="",
                output_fields=["company", "year", "document_type"],
                limit=1000
            )
            
            companies = set()
            years = set()
            doc_types = set()
            
            for result in results:
                companies.add(result.get('company'))
                years.add(result.get('year'))
                doc_types.add(result.get('document_type'))
            
            return {
                "companies": sorted(list(companies)),
                "years": sorted(list(years)),
                "document_types": sorted(list(doc_types))
            }
        except Exception as e:
            print(f"Warning: Could not retrieve metadata: {e}")
            return {
                "companies": [],
                "years": [],
                "document_types": []
            }
    
    def print_results(self, results: List[Dict[str, Any]], query: str):
        """Pretty print search results"""
        print(f"\nSearch Results for: '{query}'")
        print("=" * 60)
        
        if not results:
            print("No results found.")
            return
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['company']} ({result['year']})")
            print(f"   Type: {result['document_type']}")
            print(f"   Score: {result['score']:.3f}")
            print(f"   Source: {result['source_file']} (chunk {result['chunk_index']})")
            print(f"   Text: {result['text'][:200]}...")
            print()


def demo_searches():
    """Demonstrate various search capabilities"""
    searcher = DocumentSearcher()
    
    # Show available metadata
    metadata = searcher.get_available_metadata()
    print(f"\nAvailable data:")
    print(f"Companies: {metadata['companies']}")
    print(f"Years: {metadata['years']}")
    print(f"Document types: {metadata['document_types']}")
    
    # Example searches
    searches = [
        {
            "query": "Umsatz und Gewinn",
            "description": "General financial search"
        },
        {
            "query": "Tabellen mit Finanzkennzahlen",
            "company": "Continental",
            "description": "Continental-specific table search"
        },
        {
            "query": "revenue growth",
            "year": 2023,
            "description": "2023 revenue information"
        },
        {
            "query": "financial performance metrics",
            "doc_type": "enriched_tables",
            "description": "Search only in enriched table content"
        }
    ]
    
    for search in searches:
        print(f"\n{'='*60}")
        print(f"DEMO: {search['description']}")
        print(f"{'='*60}")
        
        results = searcher.search(
            query=search['query'],
            company=search.get('company'),
            year=search.get('year'),
            doc_type=search.get('doc_type'),
            limit=3
        )
        
        searcher.print_results(results, search['query'])


def interactive_search():
    """Interactive search interface"""
    searcher = DocumentSearcher()
    metadata = searcher.get_available_metadata()
    
    print(f"\n{'='*60}")
    print("INTERACTIVE SEARCH")
    print(f"{'='*60}")
    print(f"Available companies: {', '.join(metadata['companies'])}")
    print(f"Available years: {', '.join(map(str, metadata['years']))}")
    print(f"Available types: {', '.join(metadata['document_types'])}")
    print("\nType 'quit' to exit")
    
    while True:
        print(f"\n{'-'*40}")
        query = input("Enter search query: ").strip()
        
        if query.lower() == 'quit':
            break
        
        if not query:
            continue
        
        # Optional filters
        company = input("Filter by company (optional): ").strip() or None
        year_input = input("Filter by year (optional): ").strip()
        year = int(year_input) if year_input.isdigit() else None
        doc_type = input("Filter by document type (optional): ").strip() or None
        
        # Search
        results = searcher.search(query, company, year, doc_type)
        searcher.print_results(results, query)


def main():
    """Main function"""
    try:
        print("Milvus Document Search Demo")
        print("1. Run demo searches")
        print("2. Interactive search")
        
        choice = input("\nChoose option (1 or 2): ").strip()
        
        if choice == "1":
            demo_searches()
        elif choice == "2":
            interactive_search()
        else:
            print("Invalid choice")
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure:")
        print("1. Milvus is running")
        print("2. Documents are embedded (run embed_to_milvus.py first)")


if __name__ == "__main__":
    main()