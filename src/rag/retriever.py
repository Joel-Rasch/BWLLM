import logging
import os
import re
import argparse
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
import faiss
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
# Removed unused LangChain imports for cleaner code

from src.vector.embedder import load_faiss_index_and_metadata, setup_embedding_model


class QueryAnalyzer:
    """
    Analyzes user queries to extract company names and years for intelligent pre-filtering
    """
    
    def __init__(self, available_companies: List[str] = None, available_years: List[str] = None):
        self.logger = logging.getLogger(__name__)
        self.available_companies = [c.lower() for c in available_companies] if available_companies else []
        self.available_years = available_years if available_years else []
        
        # Common German company name patterns
        self.company_patterns = {
            'continental': ['continental', 'conti'],
            'volkswagen': ['volkswagen', 'vw', 'volkswagen group', 'volkswagen ag'],
            'bmw': ['bmw', 'bayerische motoren werke'],
            'mercedes': ['mercedes', 'mercedes-benz', 'daimler'],
            'adidas': ['adidas', 'adidas ag'],
            'siemens': ['siemens', 'siemens ag'],
            'bayer': ['bayer', 'bayer ag'],
            'basf': ['basf', 'basf se'],
            'deutsche bank': ['deutsche bank', 'db'],
            'sap': ['sap', 'sap se']
        }
        
        # Year pattern (2000-2099)
        self.year_pattern = re.compile(r'\b(20\d{2})\b')
        
    def extract_companies_and_years(self, query: str) -> Tuple[List[str], List[str]]:
        """
        Extract company names and years mentioned in the query
        
        Args:
            query: User query text
            
        Returns:
            Tuple of (found_companies, found_years)
        """
        query_lower = query.lower()
        found_companies = []
        found_years = []
        
        # Extract years
        year_matches = self.year_pattern.findall(query)
        for year in year_matches:
            if year in self.available_years:
                found_years.append(year)
        
        # Extract companies
        for canonical_name, patterns in self.company_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    # Check if this company is in our available companies
                    for available_company in self.available_companies:
                        if canonical_name in available_company.lower() or available_company.lower() in canonical_name:
                            found_companies.append(available_company.title())
                            break
                    break
        
        # Remove duplicates while preserving order
        found_companies = list(dict.fromkeys(found_companies))
        found_years = list(dict.fromkeys(found_years))
        
        self.logger.info(f"Query analysis - Found companies: {found_companies}, years: {found_years}")
        return found_companies, found_years
        
    def should_apply_filter(self, found_companies: List[str], found_years: List[str]) -> bool:
        """
        Determine if filtering should be applied based on found entities
        """
        # Apply filter if we found specific companies or years
        return len(found_companies) > 0 or len(found_years) > 0


class FAISSRetriever:
    """
    Simple FAISS retriever without LangChain BaseRetriever to avoid Pydantic conflicts
    """
    
    def __init__(
        self,
        index_path: str = "faiss_index.bin",
        metadata_path: str = "chunks_metadata.pkl",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        k: int = 5,
        company_filter: Optional[str] = None,
        year_filter: Optional[str] = None
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.model_name = model_name
        self.k = k
        self.company_filter = company_filter
        self.year_filter = year_filter
        
        # Load FAISS index and metadata
        self.index, self.metadata = load_faiss_index_and_metadata(
            self.index_path, self.metadata_path
        )
        
        # Setup embedding model
        self.embedding_model = setup_embedding_model(self.model_name)
        
        self.logger = logging.getLogger(__name__)
    
    def get_relevant_documents(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for the given query
        """
        try:
            # Create query embedding
            query_embedding = self.embedding_model.encode([query])
            query_embedding = query_embedding[0].astype(np.float32)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding.reshape(1, -1))
            
            # Search index
            scores, indices = self.index.search(
                query_embedding.reshape(1, -1), 
                min(self.k * 3, self.index.ntotal)
            )
            
            # Filter and convert to simple dictionaries
            documents = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # Invalid index
                    continue
                    
                if idx >= len(self.metadata):  # Index out of bounds check
                    self.logger.warning(f"Index {idx} out of bounds for metadata of length {len(self.metadata)}")
                    continue
                    
                chunk_metadata = self.metadata[idx]
                
                # Apply filters
                if self.company_filter and chunk_metadata.get('company', '').lower() != self.company_filter.lower():
                    continue
                if self.year_filter and chunk_metadata.get('year', '') != self.year_filter:
                    continue
                
                # Create simple document dictionary
                doc = {
                    'content': chunk_metadata.get('content', ''),
                    'score': float(score),
                    'company': chunk_metadata.get('company', 'unknown'),
                    'year': chunk_metadata.get('year', 'unknown'),
                    'source_file': chunk_metadata.get('source_file', 'unknown'),
                    'chunk_type': chunk_metadata.get('chunk_type', 'text'),
                    'has_table': chunk_metadata.get('has_table', False),
                    'char_count': chunk_metadata.get('char_count', 0),
                    'ai_description': chunk_metadata.get('ai_description', ''),
                    'enriched': chunk_metadata.get('enriched', False)
                }
                documents.append(doc)
                
                if len(documents) >= self.k:
                    break
            
            self.logger.info(f"Retrieved {len(documents)} documents for query: '{query}'")
            return documents
            
        except Exception as e:
            self.logger.error(f"Error retrieving documents: {e}")
            return []
    


class RAGSystem:
    """
    Complete RAG system using FAISS retriever and Gemini LLM
    """
    
    def __init__(
        self,
        index_path: str = "faiss_index.bin",
        metadata_path: str = "chunks_metadata.pkl",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        gemini_model: str = "gemini-1.5-flash",
        k: int = 5,
        api_key: Optional[str] = None,
        company_filter: Optional[str] = None,
        year_filter: Optional[str] = None
    ):
        self.logger = logging.getLogger(__name__)
        
        # Setup API key
        if not api_key:
            load_dotenv()
            api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            raise ValueError("Gemini API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter")
        
        # Initialize retriever
        self.retriever = FAISSRetriever(
            index_path=index_path,
            metadata_path=metadata_path,
            model_name=model_name,
            k=k,
            company_filter=company_filter,
            year_filter=year_filter
        )
        
        # Get available companies and years for query analysis
        try:
            from src.vector.embedder import get_index_stats
            stats = get_index_stats(index_path, metadata_path)
            if stats:
                available_companies = list(stats['company_distribution'].keys())
                available_years = list(stats['year_distribution'].keys())
            else:
                available_companies = []
                available_years = []
        except Exception as e:
            self.logger.warning(f"Could not load index stats for query analysis: {e}")
            available_companies = []
            available_years = []
        
        # Initialize query analyzer
        self.query_analyzer = QueryAnalyzer(
            available_companies=available_companies,
            available_years=available_years
        )
        
        # Store original filters
        self.original_company_filter = company_filter
        self.original_year_filter = year_filter
        
        # Initialize Gemini LLM using google-generativeai directly
        try:
            genai.configure(api_key=api_key)
            self.llm = genai.GenerativeModel(gemini_model)
            self.generation_config = genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=2048,
                top_p=0.8,
                top_k=10
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini model: {e}")
            raise
        
        # Setup prompt template
        self.prompt_template = """
Du bist ein hilfreicher Assistent, der Fragen zu deutschen Geschäftsberichten beantwortet.

Kontext aus den Dokumenten:
{context}

Benutzer-Frage: {question}

Anweisungen:
1. Beantworte die Frage basierend auf dem bereitgestellten Kontext
2. Wenn der Kontext Tabellen enthält, erkläre die wichtigsten Kennzahlen
3. Nenne spezifische Unternehmen und Jahre, wenn relevant
4. Wenn der Kontext nicht ausreicht, sage das ehrlich
5. Verwende klare, professionelle deutsche Sprache
6. Strukturiere deine Antwort mit Absätzen für bessere Lesbarkeit

Antwort:
"""
        
        self.logger.info("RAG system initialized successfully")
    
    def _format_docs(self, docs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents for the prompt
        """
        if not docs:
            return "Keine relevanten Dokumente gefunden."
        
        formatted_docs = []
        for i, doc in enumerate(docs, 1):
            # Format document info
            doc_info = f"Dokument {i}:"
            doc_info += f"\n- Unternehmen: {doc.get('company', 'Unbekannt')}"
            doc_info += f"\n- Jahr: {doc.get('year', 'Unbekannt')}"
            doc_info += f"\n- Typ: {doc.get('chunk_type', 'text')}"
            doc_info += f"\n- Relevanz-Score: {doc.get('score', 0):.3f}"
            
            if doc.get('has_table'):
                doc_info += "\n- Enthält Tabelle: Ja"
            
            if doc.get('ai_description'):
                doc_info += f"\n- AI-Beschreibung: {doc['ai_description']}"
            
            doc_info += f"\n- Inhalt: {doc.get('content', '')}"
            
            formatted_docs.append(doc_info)
        
        return "\n\n" + "\n\n".join(formatted_docs)
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Process a question and return answer with sources with intelligent pre-filtering
        """
        self.logger.info(f"Processing query: '{question}'")
        
        try:
            # Analyze query for company and year mentions
            found_companies, found_years = self.query_analyzer.extract_companies_and_years(question)
            
            # Apply intelligent pre-filtering if entities were found
            if self.query_analyzer.should_apply_filter(found_companies, found_years):
                # Temporarily update retriever filters
                original_company_filter = self.retriever.company_filter
                original_year_filter = self.retriever.year_filter
                
                # Set filters based on query analysis
                if found_companies and not self.original_company_filter:
                    # Use the first found company if no original filter is set
                    self.retriever.company_filter = found_companies[0]
                    self.logger.info(f"Applied intelligent company filter: {found_companies[0]}")
                
                if found_years and not self.original_year_filter:
                    # Use the first found year if no original filter is set
                    self.retriever.year_filter = found_years[0]
                    self.logger.info(f"Applied intelligent year filter: {found_years[0]}")
                
                # Get retrieved documents with intelligent filtering
                retrieved_docs = self.retriever.get_relevant_documents(question)
                
                # Restore original filters
                self.retriever.company_filter = original_company_filter
                self.retriever.year_filter = original_year_filter
                
            else:
                # No intelligent filtering needed, use regular retrieval
                retrieved_docs = self.retriever.get_relevant_documents(question)
            
            # Format context
            context = self._format_docs(retrieved_docs)
            
            # Create prompt
            prompt = self.prompt_template.format(context=context, question=question)
            
            # Generate answer using Gemini
            try:
                response = self.llm.generate_content(prompt, generation_config=self.generation_config)
                
                # Check for blocked content or empty response
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'finish_reason') and candidate.finish_reason.name in ['SAFETY', 'RECITATION']:
                        answer = "Die Antwort wurde aus Sicherheitsgründen blockiert. Bitte formulieren Sie Ihre Frage anders."
                    elif hasattr(candidate, 'content') and candidate.content.parts:
                        answer = candidate.content.parts[0].text
                    else:
                        answer = "Es konnte keine Antwort generiert werden."
                elif response.text:
                    answer = response.text
                else:
                    answer = "Es konnte keine Antwort generiert werden."
                    
            except Exception as api_error:
                self.logger.error(f"Gemini API error: {api_error}")
                answer = f"Fehler beim Generieren der Antwort: {str(api_error)}"
            
            # Format sources
            sources = []
            for doc in retrieved_docs:
                source = {
                    'company': doc.get('company', 'unknown'),
                    'year': doc.get('year', 'unknown'),
                    'source_file': doc.get('source_file', 'unknown'),
                    'chunk_type': doc.get('chunk_type', 'text'),
                    'score': doc.get('score', 0),
                    'content_preview': doc.get('content', '')[:200] + "..." if len(doc.get('content', '')) > 200 else doc.get('content', ''),
                    'metadata': doc  # Include full document for compatibility
                }
                sources.append(source)
            
            result = {
                'question': question,
                'answer': answer,
                'sources': sources,
                'num_sources': len(sources),
                'applied_filters': {
                    'companies': found_companies if 'found_companies' in locals() else [],
                    'years': found_years if 'found_years' in locals() else [],
                    'intelligent_filtering_used': 'found_companies' in locals() and self.query_analyzer.should_apply_filter(found_companies, found_years)
                }
            }
            
            self.logger.info(f"Generated answer with {len(sources)} sources")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return {
                'question': question,
                'answer': f"Entschuldigung, bei der Bearbeitung Ihrer Frage ist ein Fehler aufgetreten: {str(e)}",
                'sources': [],
                'num_sources': 0
            }
    
    def update_filters(self, company_filter: Optional[str] = None, year_filter: Optional[str] = None):
        """
        Update company and year filters for the retriever
        """
        self.retriever.company_filter = company_filter
        self.retriever.year_filter = year_filter
        self.logger.info(f"Updated filters - Company: {company_filter}, Year: {year_filter}")


def print_rag_response(result: Dict[str, Any]):
    """
    Print RAG response in a formatted way
    """
    print(f"\n🤖 RAG Antwort für: '{result['question']}'")
    print("=" * 80)
    
    print(f"\n📝 Antwort:")
    print(result['answer'])
    
    print(f"\n📚 Quellen ({result['num_sources']} gefunden):")
    for i, source in enumerate(result['sources'], 1):
        print(f"\n  {i}. {source['company']} ({source['year']}) - Score: {source['score']:.3f}")
        print(f"     📄 Datei: {source['source_file']}")
        print(f"     📊 Typ: {source['chunk_type']}")
        print(f"     📝 Vorschau: {source['content_preview']}")
    
    print("\n" + "=" * 80)


def main():
    """
    Main CLI interface for the RAG system
    """
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="RAG System mit FAISS und Gemini")
    
    parser.add_argument(
        "question",
        nargs="?",
        help="Frage an das System (interaktiv wenn nicht angegeben)"
    )
    
    parser.add_argument(
        "--index-path",
        default="faiss_index.bin",
        help="Pfad zur FAISS Index Datei (default: faiss_index.bin)"
    )
    
    parser.add_argument(
        "--metadata-path",
        default="chunks_metadata.pkl",
        help="Pfad zur Metadata Datei (default: chunks_metadata.pkl)"
    )
    
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding Modell Name (default: all-MiniLM-L6-v2)"
    )
    
    parser.add_argument(
        "--gemini-model",
        default="gemini-1.5-flash",
        help="Gemini Modell Name (default: gemini-1.5-flash)"
    )
    
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Anzahl der abzurufenden Dokumente (default: 5)"
    )
    
    parser.add_argument(
        "--company",
        help="Filter nach Unternehmen"
    )
    
    parser.add_argument(
        "--year",
        help="Filter nach Jahr"
    )
    
    parser.add_argument(
        "--api-key",
        help="Google Gemini API Key (oder GOOGLE_API_KEY env var setzen)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize RAG system
        rag_system = RAGSystem(
            index_path=args.index_path,
            metadata_path=args.metadata_path,
            model_name=args.model_name,
            gemini_model=args.gemini_model,
            k=args.k,
            api_key=args.api_key,
            company_filter=args.company,
            year_filter=args.year
        )
        
        if args.question:
            # Single question mode
            result = rag_system.query(args.question)
            print_rag_response(result)
        else:
            # Interactive mode
            print("🚀 RAG System bereit! (Tippe 'exit' zum Beenden)")
            if args.company:
                print(f"📊 Filter: Unternehmen = {args.company}")
            if args.year:
                print(f"📅 Filter: Jahr = {args.year}")
            print()
            
            while True:
                try:
                    question = input("❓ Ihre Frage: ").strip()
                    
                    if question.lower() in ['exit', 'quit', 'bye']:
                        print("👋 Auf Wiedersehen!")
                        break
                    
                    if not question:
                        continue
                    
                    # Check for filter commands
                    if question.startswith("/company "):
                        company = question[9:].strip()
                        rag_system.update_filters(company_filter=company)
                        print(f"✅ Unternehmensfilter gesetzt: {company}")
                        continue
                    
                    if question.startswith("/year "):
                        year = question[6:].strip()
                        rag_system.update_filters(year_filter=year)
                        print(f"✅ Jahresfilter gesetzt: {year}")
                        continue
                    
                    if question == "/clear":
                        rag_system.update_filters()
                        print("✅ Filter gelöscht")
                        continue
                    
                    # Process question
                    result = rag_system.query(question)
                    print_rag_response(result)
                    
                except KeyboardInterrupt:
                    print("\n👋 Auf Wiedersehen!")
                    break
                except Exception as e:
                    print(f"❌ Fehler: {e}")
        
    except Exception as e:
        print(f"❌ Initialisierung fehlgeschlagen: {e}")


if __name__ == "__main__":
    main()