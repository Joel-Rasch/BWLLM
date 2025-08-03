import sys
import os
import logging
from pathlib import Path
from typing import List, Optional
import argparse
from dotenv import load_dotenv

from src.config import *
from src.utils import setup_logging, find_files_by_extension
from src.processing.table_extractor import extract_content_from_pdf
from src.processing.chunk_processor import process_pdf_content, save_chunks_for_inspection
from src.processing.chunk_enricher import enrich_table_chunks_only
from src.vector.embedder import embed_and_store_chunks, get_index_stats


def find_pdf_files(directory: str = "Geschaeftsberichte") -> List[Path]:
    """Find all PDF files in the specified directory"""
    return find_files_by_extension(directory, ".pdf")


def process_single_pdf(pdf_path: Path, enrich_tables: bool = True, api_key: Optional[str] = None) -> List[dict]:
    """
    Process a single PDF through the complete pipeline.
    
    Args:
        pdf_path: Path to PDF file
        enrich_tables: Whether to enrich tables with AI
        api_key: API key for enrichment
        
    Returns:
        List of processed chunks
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        Exception: If processing fails
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing PDF: {pdf_path}")
    
    try:
        # Step 1: Extract content from PDF
        logger.info("Step 1: Extracting content from PDF")
        content_list = extract_content_from_pdf(str(pdf_path))
        
        if not content_list:
            logger.warning(f"No content extracted from {pdf_path}")
            return []
        
        # Step 2: Process into chunks
        logger.info("Step 2: Processing content into chunks")
        chunks = process_pdf_content(content_list, pdf_path.name)
        
        if not chunks:
            logger.warning(f"No chunks created from {pdf_path}")
            return []
        
        # Step 3: Enrich chunks with AI descriptions (optional)
        if enrich_tables and api_key:
            logger.info("Step 3: Enriching table chunks with AI descriptions")
            try:
                chunks = enrich_table_chunks_only(chunks, api_key)
            except Exception as e:
                logger.error(f"Enrichment failed: {e}, continuing without enrichment")
        elif enrich_tables and not api_key:
            logger.warning("Step 3: Skipping enrichment - no API key provided")
        else:
            logger.info("Step 3: Skipping enrichment - disabled")
        
        logger.info(f"Successfully processed {pdf_path.name}: {len(chunks)} chunks")
        return chunks
        
    except Exception as e:
        logger.error(f"Failed to process {pdf_path}: {e}")
        raise


def main(
    input_dir: str = str(DATA_DIR),
    index_path: str = INDEX_PATH,
    metadata_path: str = METADATA_PATH,
    enrich_tables: bool = True,
    api_key: Optional[str] = None,
    model_name: str = EMBEDDING_MODEL,
    index_type: str = FAISS_INDEX_TYPE,
    log_level: str = LOG_LEVEL
):
    """
    Main pipeline function that orchestrates the complete RAG processing
    
    Args:
        input_dir: Directory containing PDF files
        index_path: Path to FAISS index file
        metadata_path: Path to metadata pickle file
        enrich_tables: Whether to enrich table chunks with AI descriptions
        api_key: Google Gemini API key for enrichment
        model_name: Sentence transformer model name
        index_type: FAISS index type (IndexFlatIP or IndexFlatL2)
        log_level: Logging level
    """
    setup_logging(log_level, LOG_FILE)
    logger = logging.getLogger(__name__)
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Get API key from environment if not provided
    if not api_key and enrich_tables:
        api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        if api_key:
            logger.info("API key loaded from environment variables")
        else:
            logger.warning("No API key found. Table enrichment will be skipped.")
            enrich_tables = False
    
    logger.info("=" * 60)
    logger.info("Starting RAG Pipeline")
    logger.info("=" * 60)
    
    try:
        # Find PDF files
        logger.info(f"Searching for PDF files in: {input_dir}")
        pdf_files = find_pdf_files(input_dir)
        logger.info(f"Found {len(pdf_files)} PDF files: {[f.name for f in pdf_files]}")
        
        # Process all PDFs
        all_chunks = []
        for pdf_file in pdf_files:
            try:
                chunks = process_single_pdf(pdf_file, enrich_tables, api_key)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to process {pdf_file}: {e}")
                continue
        
        if not all_chunks:
            logger.error("No chunks were successfully processed")
            return False
        
        logger.info(f"Total chunks processed: {len(all_chunks)}")
        
        # Save chunks for inspection (transparency)
        logger.info("Saving chunks inspection file for transparency...")
        try:
            inspection_file = save_chunks_for_inspection(all_chunks, "processed_chunks_inspection.json")
            if inspection_file:
                logger.info(f"✅ Chunks inspection file saved: {inspection_file}")
            else:
                logger.warning("⚠️ Failed to save chunks inspection file")
        except Exception as e:
            logger.warning(f"⚠️ Could not save inspection file: {e}")
        
        # Step 4: Create embeddings and store in FAISS
        logger.info("Step 4: Creating embeddings and storing in FAISS index")
        try:
            index, metadata = embed_and_store_chunks(
                all_chunks, 
                index_path=index_path,
                metadata_path=metadata_path,
                model_name=model_name,
                index_type=index_type
            )
            logger.info(f"Successfully stored {len(all_chunks)} chunks in FAISS index '{index_path}'")
        except Exception as e:
            logger.error(f"Failed to embed and store chunks: {e}")
            return False
        
        # Show final statistics
        stats = get_index_stats(index_path, metadata_path)
        if stats:
            logger.info(f"Final index stats: {stats['total_entities']} total entities")
            logger.info(f"Companies: {list(stats['company_distribution'].keys())}")
            logger.info(f"Years: {list(stats['year_distribution'].keys())}")
        
        logger.info("=" * 60)
        logger.info("RAG Pipeline completed successfully!")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return False


def cli():
    """Command line interface for the pipeline"""
    parser = argparse.ArgumentParser(description="RAG Pipeline for PDF document processing")
    
    parser.add_argument(
        "--input-dir", 
        default=str(DATA_DIR),
        help=f"Directory containing PDF files (default: {DATA_DIR})"
    )
    
    parser.add_argument(
        "--index-path",
        default=INDEX_PATH, 
        help=f"FAISS index file path (default: {INDEX_PATH})"
    )
    
    parser.add_argument(
        "--metadata-path",
        default=METADATA_PATH,
        help=f"Metadata pickle file path (default: {METADATA_PATH})"
    )
    
    parser.add_argument(
        "--no-enrichment",
        action="store_true",
        help="Disable AI enrichment of table chunks"
    )
    
    parser.add_argument(
        "--api-key",
        help="Google Gemini API key for enrichment (or set GOOGLE_API_KEY env var)"
    )
    
    parser.add_argument(
        "--model-name",
        default=EMBEDDING_MODEL,
        help=f"Embedding model name (default: {EMBEDDING_MODEL})"
    )
    
    parser.add_argument(
        "--index-type",
        default=FAISS_INDEX_TYPE,
        choices=["IndexFlatIP", "IndexFlatL2"],
        help=f"FAISS index type (default: {FAISS_INDEX_TYPE})"
    )
    
    parser.add_argument(
        "--log-level",
        default=LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help=f"Logging level (default: {LOG_LEVEL})"
    )
    
    args = parser.parse_args()
    
    success = main(
        input_dir=args.input_dir,
        index_path=args.index_path,
        metadata_path=args.metadata_path,
        enrich_tables=not args.no_enrichment,
        api_key=args.api_key,
        model_name=args.model_name,
        index_type=args.index_type,
        log_level=args.log_level
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    cli()

