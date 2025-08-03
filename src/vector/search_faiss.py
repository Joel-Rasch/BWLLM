import logging
import argparse
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from src.vector.embedder import search_faiss_index, get_index_stats, setup_embedding_model
from src.config import INDEX_PATH, METADATA_PATH, EMBEDDING_MODEL


def search_documents(
    query: str,
    index_path: str = INDEX_PATH,
    metadata_path: str = METADATA_PATH,
    model_name: str = EMBEDDING_MODEL,
    k: int = 5,
    company_filter: Optional[str] = None,
    year_filter: Optional[str] = None
) -> List[dict]:
    """
    Search documents using FAISS index with company and year filters
    """
    logger = logging.getLogger(__name__)
    
    # Setup embedding model
    model = setup_embedding_model(model_name)
    
    # Create query embedding
    query_embedding = model.encode([query])
    
    # Search index
    results = search_faiss_index(
        query_embedding[0],
        index_path=index_path,
        metadata_path=metadata_path,
        k=k,
        company_filter=company_filter,
        year_filter=year_filter
    )
    
    return results


def print_search_results(results: List[dict], query: str):
    """
    Print search results in a formatted way
    """
    print(f"\n🔍 Search Query: '{query}'")
    print(f"📊 Found {len(results)} results:\n")
    
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  📈 Score: {result['score']:.4f}")
        print(f"  🏢 Company: {result['company']}")
        print(f"  📅 Year: {result['year']}")
        print(f"  📄 Type: {result['chunk_type']}")
        print(f"  📊 Has Table: {'Yes' if result['has_table'] else 'No'}")
        
        content = result['content']
        if len(content) > 200:
            content = content
        print(f"  📝 Content: {content}")
        print("-" * 50)


def main():
    """
    Main function for document search CLI
    """
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Search documents using FAISS index")
    
    parser.add_argument(
        "query",
        help="Search query text"
    )
    
    parser.add_argument(
        "--index-path",
        default="faiss_index.bin",
        help="Path to FAISS index file (default: faiss_index.bin)"
    )
    
    parser.add_argument(
        "--metadata-path",
        default="chunks_metadata.pkl",
        help="Path to metadata file (default: chunks_metadata.pkl)"
    )
    
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name (default: all-MiniLM-L6-v2)"
    )
    
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)"
    )
    
    parser.add_argument(
        "--company",
        help="Filter by company name"
    )
    
    parser.add_argument(
        "--year",
        help="Filter by year"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show index statistics"
    )
    
    args = parser.parse_args()
    
    # Show stats if requested
    if args.stats:
        stats = get_index_stats(args.index_path, args.metadata_path)
        if stats:
            print(f"📊 Index Statistics:")
            print(f"  Total entities: {stats['total_entities']}")
            print(f"  Dimension: {stats['dimension']}")
            print(f"  Index type: {stats['index_type']}")
            print(f"  Companies: {stats['company_distribution']}")
            print(f"  Years: {stats['year_distribution']}")
            print(f"  Types: {stats['type_distribution']}")
        else:
            print("❌ Could not load index statistics")
        return
    
    # Perform search
    try:
        results = search_documents(
            query=args.query,
            index_path=args.index_path,
            metadata_path=args.metadata_path,
            model_name=args.model_name,
            k=args.k,
            company_filter=args.company,
            year_filter=args.year
        )
        
        if results:
            print_search_results(results, args.query)
        else:
            print(f"❌ No results found for query: '{args.query}'")
            
            # Show available filters
            stats = get_index_stats(args.index_path, args.metadata_path)
            if stats:
                print(f"\n💡 Available companies: {list(stats['company_distribution'].keys())}")
                print(f"💡 Available years: {list(stats['year_distribution'].keys())}")
        
    except Exception as e:
        print(f"❌ Search failed: {e}")


if __name__ == "__main__":
    main()