import os
import time
import logging
from typing import List, Dict, Any
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from dotenv import load_dotenv


def setup_gemini_api(api_key: str = None):
    """
    Setup Gemini API with key from parameter or environment
    """
    if not api_key:
        load_dotenv()
        api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        raise ValueError("Gemini API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter")
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')


def create_enrichment_prompt(chunk: Dict[str, Any]) -> str:
    """
    Create a prompt for enriching a chunk with AI description
    """
    content = chunk['content']
    metadata = chunk.get('metadata', {})
    chunk_type = chunk.get('type', 'text')
    
    if chunk_type == 'table':
        return f"""
Analysiere diese Tabelle und erstelle eine prägnante, natürlichsprachige Beschreibung:

{content}

Bitte beschreibe:
1. Was diese Tabelle zeigt (Hauptinhalt)
2. Die wichtigsten Kennzahlen oder Trends
3. Besondere Auffälligkeiten oder Erkenntnisse

Antworte auf Deutsch in max. 150 Wörtern.
"""
    else:
        return f"""
Analysiere diesen Textabschnitt und erstelle eine kurze Zusammenfassung:

{content}

Bitte fasse zusammen:
1. Die Hauptthemen des Texts
2. Wichtige Informationen oder Fakten
3. Relevante Kennzahlen falls vorhanden

Antworte auf Deutsch in max. 100 Wörtern.
"""


def get_retry_delay(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
    """
    Calculate exponential backoff delay with jitter
    """
    import random
    delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
    return delay

def is_rate_limit_error(error) -> bool:
    """
    Check if error is a rate limiting error
    """
    error_str = str(error).lower()
    return any(indicator in error_str for indicator in [
        '429', 'rate limit', 'quota', 'too many requests', 'exceeded'
    ])

def is_retryable_error(error) -> bool:
    """
    Determine if an error is worth retrying
    """
    error_str = str(error).lower()
    
    # Rate limiting - definitely retry
    if is_rate_limit_error(error):
        return True
    
    # Server errors - retry
    if any(indicator in error_str for indicator in ['500', '502', '503', '504', 'server error', 'timeout']):
        return True
    
    # Network errors - retry
    if any(indicator in error_str for indicator in ['connection', 'network', 'dns']):
        return True
    
    # Client errors that shouldn't be retried
    if any(indicator in error_str for indicator in ['400', '401', '403', 'unauthorized', 'forbidden', 'invalid']):
        return False
    
    # Default to retry for unknown errors
    return True

def enrich_single_chunk(chunk: Dict[str, Any], model, max_retries: int = 5) -> Dict[str, Any]:
    """
    Enrich a single chunk with AI description using robust retry logic
    """
    logger = logging.getLogger(__name__)
    
    prompt = create_enrichment_prompt(chunk)
    source_file = chunk.get('metadata', {}).get('source_file', 'unknown')
    
    for attempt in range(max_retries + 1):
        try:
            response = model.generate_content(prompt)
            
            # Check if response is valid
            if not response or not hasattr(response, 'text') or not response.text:
                raise ValueError("Empty or invalid response from Gemini API")
            
            description = response.text.strip()
            
            # Add enrichment to chunk
            enriched_chunk = chunk.copy()
            enriched_chunk['ai_description'] = description
            enriched_chunk['enriched'] = True
            
            logger.debug(f"✅ Successfully enriched chunk from {source_file}")
            return enriched_chunk
            
        except Exception as e:
            is_final_attempt = attempt == max_retries
            
            if is_rate_limit_error(e):
                # Handle rate limiting with longer delays
                if not is_final_attempt:
                    delay = get_retry_delay(attempt, base_delay=5.0, max_delay=120.0)
                    logger.warning(f"⏱️ Rate limit hit (attempt {attempt + 1}/{max_retries + 1}) for {source_file}. Waiting {delay:.1f}s before retry...")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"❌ Rate limit exceeded for {source_file} after {max_retries + 1} attempts")
            
            elif is_retryable_error(e):
                # Handle other retryable errors
                if not is_final_attempt:
                    delay = get_retry_delay(attempt, base_delay=2.0, max_delay=30.0)
                    logger.warning(f"⚠️ Retryable error (attempt {attempt + 1}/{max_retries + 1}) for {source_file}: {str(e)[:100]}... Retrying in {delay:.1f}s")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"❌ Retryable error persisted for {source_file} after {max_retries + 1} attempts: {e}")
            
            else:
                # Non-retryable error
                logger.error(f"❌ Non-retryable error for {source_file}: {e}")
                break
    
    # Return original chunk with error information
    enriched_chunk = chunk.copy()
    
    if is_rate_limit_error(e):
        enriched_chunk['ai_description'] = "⏱️ AI-Beschreibung nicht verfügbar: API-Ratenlimit erreicht"
    else:
        enriched_chunk['ai_description'] = f"❌ AI-Beschreibung fehlgeschlagen: {str(e)[:100]}"
    
    enriched_chunk['enriched'] = False
    enriched_chunk['enrichment_error'] = str(e)
    
    logger.error(f"❌ Failed to enrich chunk from {source_file} after {max_retries + 1} attempts")
    return enriched_chunk


def enrich_chunks(chunks: List[Dict[str, Any]], api_key: str = None, batch_size: int = 5) -> List[Dict[str, Any]]:
    """
    Enrich multiple chunks with AI descriptions using rate-limiting friendly approach
    """
    logger = logging.getLogger(__name__)
    
    if not chunks:
        logger.info("No chunks to enrich")
        return []
    
    try:
        model = setup_gemini_api(api_key)
    except ValueError as e:
        logger.error(f"Failed to setup Gemini API: {e}")
        # Return chunks without enrichment
        return chunks
    
    enriched_chunks = []
    total_chunks = len(chunks)
    successful_enrichments = 0
    failed_enrichments = 0
    
    logger.info(f"🚀 Starting enrichment of {total_chunks} chunks (batch size: {batch_size})")
    
    for i, chunk in enumerate(chunks):
        try:
            # Add small delay between requests to be gentle on API
            if i > 0:
                time.sleep(0.5)  # 500ms between requests
            
            enriched_chunk = enrich_single_chunk(chunk, model)
            enriched_chunks.append(enriched_chunk)
            
            # Track success/failure
            if enriched_chunk.get('enriched', False):
                successful_enrichments += 1
            else:
                failed_enrichments += 1
            
            # Progress reporting and batch delays
            if (i + 1) % batch_size == 0:
                progress_pct = ((i + 1) / total_chunks) * 100
                logger.info(f"📊 Progress: {i + 1}/{total_chunks} ({progress_pct:.1f}%) - ✅ {successful_enrichments} success, ❌ {failed_enrichments} failed")
                
                # Longer delay between batches to prevent rate limiting
                if i + 1 < total_chunks:  # Don't delay after the last batch
                    batch_delay = 2.0  # 2 seconds between batches
                    logger.debug(f"⏸️ Batch delay: {batch_delay}s")
                    time.sleep(batch_delay)
                
        except Exception as e:
            logger.error(f"❌ Failed to process chunk {i}: {e}")
            # Add original chunk without enrichment
            chunk_copy = chunk.copy()
            chunk_copy['enriched'] = False
            chunk_copy['ai_description'] = f"❌ Verarbeitungsfehler: {str(e)[:50]}"
            enriched_chunks.append(chunk_copy)
            failed_enrichments += 1
    
    # Final summary
    success_rate = (successful_enrichments / total_chunks) * 100 if total_chunks > 0 else 0
    logger.info(f"🎯 Enrichment completed: ✅ {successful_enrichments}/{total_chunks} successful ({success_rate:.1f}%)")
    
    if failed_enrichments > 0:
        logger.warning(f"⚠️ {failed_enrichments} chunks failed enrichment - check logs for details")
    
    return enriched_chunks


def enrich_table_chunks_only(chunks: List[Dict[str, Any]], api_key: str = None) -> List[Dict[str, Any]]:
    """
    Enrich only table chunks, leave text chunks unchanged - with improved error handling
    """
    logger = logging.getLogger(__name__)
    
    # Find table chunks (supporting both 'type' and 'chunk_type' fields)
    table_chunks = [chunk for chunk in chunks if 
                   chunk.get('type') == 'table' or 
                   chunk.get('chunk_type') == 'table']
    
    if not table_chunks:
        logger.info("📊 No table chunks found to enrich")
        return chunks
    
    logger.info(f"📊 Found {len(table_chunks)} table chunks out of {len(chunks)} total chunks")
    
    try:
        model = setup_gemini_api(api_key)
    except ValueError as e:
        logger.error(f"❌ Failed to setup Gemini API: {e}")
        return chunks
    
    # Process table chunks with improved error handling
    enriched_mapping = {}
    successful_enrichments = 0
    failed_enrichments = 0
    
    logger.info(f"🚀 Starting table enrichment with improved retry logic...")
    
    for i, chunk in enumerate(table_chunks):
        try:
            # Add small delay between requests
            if i > 0:
                time.sleep(1.0)  # 1 second between table enrichments
            
            enriched_chunk = enrich_single_chunk(chunk, model)
            chunk_id = id(chunk)  # Use memory id as unique identifier
            enriched_mapping[chunk_id] = enriched_chunk
            
            if enriched_chunk.get('enriched', False):
                successful_enrichments += 1
            else:
                failed_enrichments += 1
            
            # Progress update every 3 chunks
            if (i + 1) % 3 == 0 or i + 1 == len(table_chunks):
                logger.info(f"📊 Table enrichment progress: {i + 1}/{len(table_chunks)} - ✅ {successful_enrichments} success, ❌ {failed_enrichments} failed")
                
        except Exception as e:
            logger.error(f"❌ Failed to process table chunk {i}: {e}")
            failed_enrichments += 1
    
    # Replace table chunks with enriched versions
    result_chunks = []
    for chunk in chunks:
        chunk_id = id(chunk)
        if chunk_id in enriched_mapping:
            result_chunks.append(enriched_mapping[chunk_id])
        else:
            result_chunks.append(chunk)
    
    # Final summary
    success_rate = (successful_enrichments / len(table_chunks)) * 100 if table_chunks else 0
    logger.info(f"🎯 Table enrichment completed: ✅ {successful_enrichments}/{len(table_chunks)} tables successful ({success_rate:.1f}%)")
    
    if failed_enrichments > 0:
        logger.warning(f"⚠️ {failed_enrichments} table chunks failed enrichment due to API issues")
    
    return result_chunks


def save_enriched_chunks(chunks: List[Dict[str, Any]], output_path: str):
    """
    Save enriched chunks to a file for inspection
    """
    import json
    
    logger = logging.getLogger(__name__)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(chunks)} enriched chunks to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save enriched chunks: {e}")


def load_enriched_chunks(input_path: str) -> List[Dict[str, Any]]:
    """
    Load enriched chunks from a file
    """
    import json
    
    logger = logging.getLogger(__name__)
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        logger.info(f"Loaded {len(chunks)} enriched chunks from {input_path}")
        return chunks
    except Exception as e:
        logger.error(f"Failed to load enriched chunks: {e}")
        return []


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Example chunk
    example_chunk = {
        'content': 'TABELLE:\n| Jahr | Umsatz | Gewinn |\n|------|--------|--------|\n| 2022 | 100M   | 10M    |\n| 2023 | 120M   | 15M    |',
        'type': 'table',
        'metadata': {
            'source_file': 'test.md',
            'company': 'TestFirma',
            'year': '2023'
        }
    }
    
    try:
        enriched = enrich_chunks([example_chunk])
        print("Enriched chunk:")
        print(f"AI Description: {enriched[0].get('ai_description', 'Not available')}")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to set GOOGLE_API_KEY environment variable")