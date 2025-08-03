import logging
import numpy as np
import pickle
import faiss
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from sentence_transformers import SentenceTransformer


def setup_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Initialize the embedding model
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading embedding model: {model_name}")
    
    try:
        model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise


def create_embeddings(chunks: List[Dict[str, Any]], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> List[Dict[str, Any]]:
    """
    Create embeddings for chunks
    """
    logger = logging.getLogger(__name__)
    
    if not chunks:
        logger.warning("No chunks provided for embedding")
        return []
    
    model = setup_embedding_model(model_name)
    
    # Extract text content for embedding
    texts = []
    for chunk in chunks:
        content = chunk.get('content', '')
        # For table chunks, also include AI description if available
        if chunk.get('ai_description'):
            content = f"{content}\n\nBeschreibung: {chunk['ai_description']}"
        texts.append(content)
    
    logger.info(f"Creating embeddings for {len(texts)} chunks")
    
    try:
        # Create embeddings in batches
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
        
        # Add embeddings to chunks
        enriched_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_with_embedding = chunk.copy()
            chunk_with_embedding['embedding'] = embedding.tolist()
            chunk_with_embedding['embedding_model'] = model_name
            enriched_chunks.append(chunk_with_embedding)
        
        logger.info(f"Successfully created embeddings for {len(enriched_chunks)} chunks")
        return enriched_chunks
        
    except Exception as e:
        logger.error(f"Failed to create embeddings: {e}")
        raise


def create_faiss_index(dimension: int = 384, index_type: str = "IndexFlatIP"):
    """
    Create FAISS index for storing embeddings
    """
    logger = logging.getLogger(__name__)
    
    try:
        if index_type == "IndexFlatIP":
            # Inner Product (cosine similarity for normalized vectors)
            index = faiss.IndexFlatIP(dimension)
        elif index_type == "IndexFlatL2":
            # L2 distance
            index = faiss.IndexFlatL2(dimension)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        logger.info(f"Created FAISS index: {index_type} with dimension {dimension}")
        return index
        
    except Exception as e:
        logger.error(f"Failed to create FAISS index: {e}")
        raise


def save_faiss_index_and_metadata(index, chunks_metadata: List[Dict[str, Any]], 
                                  index_path: str = "faiss_index.bin", 
                                  metadata_path: str = "chunks_metadata.pkl"):
    """
    Save FAISS index and associated metadata to files
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Save FAISS index
        faiss.write_index(index, index_path)
        logger.info(f"Saved FAISS index to {index_path}")
        
        # Save metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump(chunks_metadata, f)
        logger.info(f"Saved metadata to {metadata_path}")
        
    except Exception as e:
        logger.error(f"Failed to save FAISS index and metadata: {e}")
        raise


def load_faiss_index_and_metadata(index_path: str = "faiss_index.bin", 
                                  metadata_path: str = "chunks_metadata.pkl") -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """
    Load FAISS index and associated metadata from files
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Load FAISS index
        index = faiss.read_index(index_path)
        logger.info(f"Loaded FAISS index from {index_path}")
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            chunks_metadata = pickle.load(f)
        logger.info(f"Loaded metadata from {metadata_path}")
        
        return index, chunks_metadata
        
    except Exception as e:
        logger.error(f"Failed to load FAISS index and metadata: {e}")
        raise


def insert_chunks_to_faiss(chunks: List[Dict[str, Any]], 
                          index_path: str = "faiss_index.bin",
                          metadata_path: str = "chunks_metadata.pkl",
                          index_type: str = "IndexFlatIP"):
    """
    Insert chunks with embeddings into FAISS index
    """
    logger = logging.getLogger(__name__)
    
    if not chunks:
        logger.warning("No chunks to insert")
        return None, []
    
    # Check if chunks have embeddings
    if not chunks[0].get('embedding'):
        raise ValueError("Chunks must have embeddings before inserting to FAISS")
    
    # Prepare embeddings and metadata
    embeddings = []
    chunks_metadata = []
    
    for i, chunk in enumerate(chunks):
        # Extract embedding
        embedding = np.array(chunk['embedding'], dtype=np.float32)
        embeddings.append(embedding)
        
        # Prepare metadata
        metadata = chunk.get('metadata', {})
        chunk_metadata = {
            'id': i,
            'content': chunk.get('content', ''),
            'company': metadata.get('company', 'unknown'),
            'year': metadata.get('year', 'unknown'),
            'source_file': metadata.get('source_file', 'unknown'),
            'chunk_type': chunk.get('type', 'text'),
            'has_table': metadata.get('has_table', False),
            'char_count': metadata.get('char_count', 0),
            'ai_description': chunk.get('ai_description', ''),
            'enriched': chunk.get('enriched', False)
        }
        chunks_metadata.append(chunk_metadata)
    
    # Convert to numpy array
    embeddings_array = np.vstack(embeddings)
    
    # Normalize embeddings for cosine similarity (if using IndexFlatIP)
    if index_type == "IndexFlatIP":
        faiss.normalize_L2(embeddings_array)
    
    # Create or load existing index
    dimension = embeddings_array.shape[1]
    
    if Path(index_path).exists():
        logger.info("Loading existing FAISS index")
        try:
            index, existing_metadata = load_faiss_index_and_metadata(index_path, metadata_path)
            # Update IDs to continue from existing count
            start_id = len(existing_metadata)
            for i, metadata in enumerate(chunks_metadata):
                metadata['id'] = start_id + i
            chunks_metadata = existing_metadata + chunks_metadata
        except Exception as e:
            logger.warning(f"Failed to load existing index: {e}. Creating new index.")
            index = create_faiss_index(dimension, index_type)
    else:
        logger.info("Creating new FAISS index")
        index = create_faiss_index(dimension, index_type)
    
    try:
        # Add embeddings to index
        index.add(embeddings_array)
        logger.info(f"Successfully added {len(embeddings)} embeddings to FAISS index")
        
        # Save index and metadata
        save_faiss_index_and_metadata(index, chunks_metadata, index_path, metadata_path)
        
        return index, chunks_metadata
        
    except Exception as e:
        logger.error(f"Failed to insert chunks into FAISS: {e}")
        raise


def embed_and_store_chunks(chunks: List[Dict[str, Any]], 
                          index_path: str = "faiss_index.bin",
                          metadata_path: str = "chunks_metadata.pkl",
                          model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                          index_type: str = "IndexFlatIP"):
    """
    Complete pipeline: create embeddings and store in FAISS
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Starting embedding and storage pipeline")
    
    # Create embeddings
    chunks_with_embeddings = create_embeddings(chunks, model_name)
    
    # Store in FAISS
    index, metadata = insert_chunks_to_faiss(
        chunks_with_embeddings, 
        index_path, 
        metadata_path, 
        index_type
    )
    
    logger.info("Embedding and storage pipeline completed successfully")
    return index, metadata


def get_index_stats(index_path: str = "faiss_index.bin", 
                   metadata_path: str = "chunks_metadata.pkl"):
    """
    Get statistics about the FAISS index
    """
    logger = logging.getLogger(__name__)
    
    if not Path(index_path).exists():
        logger.warning(f"FAISS index {index_path} does not exist")
        return None
    
    try:
        index, metadata = load_faiss_index_and_metadata(index_path, metadata_path)
        
        # Count by company and year
        company_counts = {}
        year_counts = {}
        type_counts = {}
        
        for item in metadata:
            company = item.get('company', 'unknown')
            year = item.get('year', 'unknown')
            chunk_type = item.get('chunk_type', 'text')
            
            company_counts[company] = company_counts.get(company, 0) + 1
            year_counts[year] = year_counts.get(year, 0) + 1
            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
        
        stats = {
            'total_entities': index.ntotal,
            'dimension': index.d,
            'index_type': type(index).__name__,
            'company_distribution': company_counts,
            'year_distribution': year_counts,
            'type_distribution': type_counts
        }
        
        logger.info(f"FAISS index contains {stats['total_entities']} entities")
        logger.info(f"Companies: {list(company_counts.keys())}")
        logger.info(f"Years: {list(year_counts.keys())}")
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get index stats: {e}")
        return None


def search_faiss_index(query_embedding: np.ndarray, 
                      index_path: str = "faiss_index.bin",
                      metadata_path: str = "chunks_metadata.pkl",
                      k: int = 5,
                      company_filter: Optional[str] = None,
                      year_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Search FAISS index with optional company and year filters
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Load index and metadata
        index, metadata = load_faiss_index_and_metadata(index_path, metadata_path)
        
        # Normalize query embedding for cosine similarity
        query_embedding = query_embedding.copy().astype(np.float32)
        if hasattr(index, 'metric_type') or 'IP' in str(type(index)):
            faiss.normalize_L2(query_embedding.reshape(1, -1))
        
        # Search all documents first
        scores, indices = index.search(query_embedding.reshape(1, -1), min(k * 3, index.ntotal))
        
        # Filter results by company and year
        filtered_results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # Invalid index
                continue
                
            chunk_metadata = metadata[idx]
            
            # Apply filters
            if company_filter and chunk_metadata.get('company', '').lower() != company_filter.lower():
                continue
            if year_filter and chunk_metadata.get('year', '') != year_filter:
                continue
            
            result = {
                'score': float(score),
                'metadata': chunk_metadata,
                'content': chunk_metadata.get('content', ''),
                'company': chunk_metadata.get('company', 'unknown'),
                'year': chunk_metadata.get('year', 'unknown'),
                'chunk_type': chunk_metadata.get('chunk_type', 'text'),
                'has_table': chunk_metadata.get('has_table', False)
            }
            filtered_results.append(result)
            
            if len(filtered_results) >= k:
                break
        
        logger.info(f"Found {len(filtered_results)} results for query")
        return filtered_results
        
    except Exception as e:
        logger.error(f"Failed to search FAISS index: {e}")
        return []


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Example chunks
    example_chunks = [
        {
            'content': 'Dies ist ein Beispieltext über Umsätze.',
            'type': 'text',
            'metadata': {
                'source_file': 'test.md',
                'company': 'TestFirma',
                'year': '2023',
                'has_table': False,
                'char_count': 45
            }
        },
        {
            'content': 'TABELLE:\n| Jahr | Umsatz |\n|------|--------|\n| 2023 | 100M   |',
            'type': 'table',
            'metadata': {
                'source_file': 'test.md',
                'company': 'TestFirma',
                'year': '2023',
                'has_table': True,
                'char_count': 60
            }
        }
    ]
    
    try:
        # Test embedding and storage
        index, metadata = embed_and_store_chunks(example_chunks)
        print(f"Successfully processed {len(example_chunks)} chunks")
        
        # Show index stats
        stats = get_index_stats()
        if stats:
            print(f"FAISS index now contains {stats['total_entities']} entities")
            print(f"Companies: {stats['company_distribution']}")
            print(f"Years: {stats['year_distribution']}")
            
        # Test search
        model = setup_embedding_model()
        query_embedding = model.encode(["Umsatz 2023"])
        results = search_faiss_index(
            query_embedding[0], 
            company_filter="TestFirma",
            year_filter="2023"
        )
        
        print(f"\nSearch results: {len(results)} found")
        for result in results:
            print(f"Score: {result['score']:.3f}, Company: {result['company']}, Year: {result['year']}")
            
    except Exception as e:
        print(f"Error: {e}")