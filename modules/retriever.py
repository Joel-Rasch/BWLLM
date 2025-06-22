"""
Retrieval module for hybrid document retrieval
"""
from typing import List, Dict, Any
from langchain.schema import Document

class HybridRetriever:
    def __init__(self, embedding_manager, query_enhancer):
        self.embedding_manager = embedding_manager
        self.query_enhancer = query_enhancer
    
    def retrieve(self, query: str, top_k: int = 5, max_query_variants: int = 4) -> Dict[str, Any]:
        """Perform hybrid retrieval using multiple query variants"""
        
        if not self.embedding_manager.vectorstore:
            return {
                'documents': [],
                'query_variants': [query],
                'retrieval_details': {'error': 'No vector store available'}
            }
        
        # Get enhanced queries
        query_variants = self.query_enhancer.enhance_query(query, max_query_variants)
        
        all_results = []
        seen_content = set()
        retrieval_details = {
            'original_query': query,
            'total_variants': len(query_variants),
            'results_per_variant': {}
        }
        
        # Search with each query variant
        for i, variant_query in enumerate(query_variants):
            try:
                results = self.embedding_manager.vectorstore.similarity_search_with_score(
                    variant_query, k=top_k
                )
                
                variant_results = []
                for doc, score in results:
                    # Create unique identifier to avoid duplicates
                    content_hash = hash(doc.page_content[:100])
                    
                    if content_hash not in seen_content:
                        # Add retrieval metadata
                        doc.metadata['retrieval_score'] = float(score)
                        doc.metadata['query_variant'] = variant_query
                        doc.metadata['variant_index'] = i
                        
                        all_results.append(doc)
                        variant_results.append({
                            'score': float(score),
                            'company': doc.metadata.get('company', 'Unknown'),
                            'source': doc.metadata.get('source', 'Unknown')
                        })
                        seen_content.add(content_hash)
                
                retrieval_details['results_per_variant'][f'variant_{i+1}'] = {
                    'query': variant_query,
                    'results_count': len(variant_results),
                    'results': variant_results
                }
                
            except Exception as e:
                retrieval_details['results_per_variant'][f'variant_{i+1}'] = {
                    'query': variant_query,
                    'error': str(e)
                }
        
        # Sort by relevance score (lower is better for similarity search)
        all_results.sort(key=lambda x: x.metadata.get('retrieval_score', float('inf')))
        
        # Limit to requested number
        final_results = all_results[:top_k * 2]  # Return more for better context
        
        retrieval_details['total_unique_results'] = len(final_results)
        
        return {
            'documents': final_results,
            'query_variants': query_variants,
            'retrieval_details': retrieval_details
        }
