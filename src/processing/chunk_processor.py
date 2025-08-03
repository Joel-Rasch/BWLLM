import re
import json
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_company_and_year_from_filename(filename: str) -> Tuple[str, str]:
    """
    Extract company name and year from filename
    Example: Continental_2023.md -> ("Continental", "2023")
    """
    base_name = Path(filename).stem
    
    # Search for year (4-digit number)
    year_match = re.search(r'(20\d{2})', base_name)
    year = year_match.group(1) if year_match else "unknown"
    
    # Extract company name (first part before underscore)
    company_part = re.sub(r'_(20\d{2})|_geschäftsbericht|_annual|_report|_enriched', '', base_name, flags=re.IGNORECASE)
    company = company_part.split('_')[0].strip().title() if company_part else "unknown"
    
    return company, year


def create_text_splitter(chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Create a LangChain text splitter with optimized settings
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "],
        keep_separator=True,
        strip_whitespace=True
    )


def chunk_text_content(text: str, filename: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Split text content into chunks using LangChain splitter
    """
    company, year = extract_company_and_year_from_filename(filename)
    
    # Remove AI description markers
    text = re.sub(r'### KI-Beschreibung\s*\n?', '', text)
    text = re.sub(r'##+ KI-Beschreibung\s*\n?', '', text)
    
    text_splitter = create_text_splitter(chunk_size, chunk_overlap)
    text_chunks = text_splitter.split_text(text)
    
    chunks = []
    for chunk_text in text_chunks:
        chunk_text = chunk_text.strip()
        if not chunk_text:
            continue
            
        chunks.append({
            'content': chunk_text,
            'type': 'text',
            'metadata': {
                'source_file': filename,
                'company': company,
                'year': year,
                'has_table': False,
                'char_count': len(chunk_text)
            }
        })
    
    return chunks


def process_markdown_content(content: str, filename: str) -> List[Dict[str, Any]]:
    """
    Process markdown content into chunks, handling tables separately
    """
    company, year = extract_company_and_year_from_filename(filename)
    
    # Split content into table and text sections
    sections = split_content_by_tables(content)
    
    chunks = []
    for section in sections:
        if section['type'] == 'table':
            # Create table chunk
            chunk = {
                'content': f"TABELLE:\n{section['content']}\n\nUNTERNEHMEN: {company}\nJAHR: {year}",
                'type': 'table',
                'metadata': {
                    'source_file': filename,
                    'company': company,
                    'year': year,
                    'has_table': True,
                    'char_count': len(section['content'])
                }
            }
            chunks.append(chunk)
        else:
            # Process text with chunking
            text_chunks = chunk_text_content(section['content'], filename)
            chunks.extend(text_chunks)
    
    return chunks


def split_content_by_tables(content: str) -> List[Dict[str, str]]:
    """
    Split content into table and text sections based on markers
    """
    sections = []
    
    # Find table markers
    table_pattern = r'--- Tabelle Start ---(.*?)--- Tabelle Ende ---'
    
    last_end = 0
    for match in re.finditer(table_pattern, content, re.DOTALL):
        # Text before table
        if last_end < match.start():
            text_content = content[last_end:match.start()].strip()
            if text_content:
                sections.append({'type': 'text', 'content': text_content})
        
        # Table
        table_content = match.group(1).strip()
        if table_content:
            sections.append({'type': 'table', 'content': table_content})
        
        last_end = match.end()
    
    # Remaining text
    if last_end < len(content):
        remaining_text = content[last_end:].strip()
        if remaining_text:
            sections.append({'type': 'text', 'content': remaining_text})
    
    return sections


def process_pdf_content(content_list: List[Dict], filename: str) -> List[Dict[str, Any]]:
    """
    Process extracted PDF content into chunks
    """
    company, year = extract_company_and_year_from_filename(filename)
    chunks = []
    
    # Group text content together for better chunking
    text_buffer = []
    
    for item in content_list:
        if item['type'] == 'table':
            # Process any accumulated text first
            if text_buffer:
                combined_text = "\n\n".join(text_buffer)
                text_chunks = chunk_text_content(combined_text, filename)
                chunks.extend(text_chunks)
                text_buffer = []
            
            # Add table chunk
            chunk = {
                'content': f"TABELLE:\n{item['content']}\n\nUNTERNEHMEN: {company}\nJAHR: {year}",
                'type': 'table',
                'metadata': {
                    'source_file': filename,
                    'company': company,
                    'year': year,
                    'has_table': True,
                    'page_number': item.get('page_number'),
                    'element_index': item.get('element_index'),
                    'char_count': len(item['content'])
                }
            }
            chunks.append(chunk)
        else:
            # Accumulate text content
            text_buffer.append(item['content'])
    
    # Process any remaining text
    if text_buffer:
        combined_text = "\n\n".join(text_buffer)
        text_chunks = chunk_text_content(combined_text, filename)
        chunks.extend(text_chunks)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Processed {len(chunks)} chunks from {filename}")
    
    return chunks


def chunk_document(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Main function to chunk a document (PDF or markdown)
    """
    file_path = Path(file_path)
    
    if file_path.suffix.lower() == '.pdf':
        from src.processing.table_extractor import extract_content_from_pdf
        content_list = extract_content_from_pdf(str(file_path))
        return process_pdf_content(content_list, file_path.name)
    
    elif file_path.suffix.lower() in ['.md', '.markdown']:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return process_markdown_content(content, file_path.name)
    
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def batch_chunk_documents(file_paths: List[str], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Process multiple documents into chunks
    """
    all_chunks = []
    logger = logging.getLogger(__name__)
    
    for file_path in file_paths:
        try:
            chunks = chunk_document(file_path, chunk_size, chunk_overlap)
            all_chunks.extend(chunks)
            logger.info(f"Successfully processed {file_path}: {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
    
    logger.info(f"Total chunks processed: {len(all_chunks)}")
    return all_chunks


def save_chunks_for_inspection(chunks: List[Dict[str, Any]], output_file: str = "processed_chunks_inspection.json"):
    """
    Save processed chunks to a readable JSON file for transparency and debugging
    
    Args:
        chunks: List of processed chunks with metadata
        output_file: Output file name
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Prepare chunks for readable output
        inspection_data = {
            "metadata": {
                "total_chunks": len(chunks),
                "generation_timestamp": str(Path().resolve()),
                "chunk_types": {},
                "companies": set(),
                "years": set()
            },
            "chunks": []
        }
        
        # Count chunk types and collect unique companies/years
        for chunk in chunks:
            chunk_type = chunk.get('chunk_type', 'unknown')
            inspection_data["metadata"]["chunk_types"][chunk_type] = inspection_data["metadata"]["chunk_types"].get(chunk_type, 0) + 1
            inspection_data["metadata"]["companies"].add(chunk.get('company', 'unknown'))
            inspection_data["metadata"]["years"].add(chunk.get('year', 'unknown'))
            
            # Create readable chunk entry
            chunk_entry = {
                "chunk_id": len(inspection_data["chunks"]) + 1,
                "source_info": {
                    "company": chunk.get('company', 'unknown'),
                    "year": chunk.get('year', 'unknown'),
                    "source_file": chunk.get('source_file', 'unknown'),
                    "chunk_type": chunk.get('chunk_type', 'text')
                },
                "content_info": {
                    "character_count": chunk.get('char_count', len(chunk.get('content', ''))),
                    "has_table": chunk.get('has_table', False),
                    "enriched": chunk.get('enriched', False)
                },
                "content": {
                    "raw_content": chunk.get('content', ''),
                    "ai_description": chunk.get('ai_description', None) if chunk.get('enriched', False) else None
                },
                "preview": {
                    "content_preview": chunk.get('content', '')[:200] + "..." if len(chunk.get('content', '')) > 200 else chunk.get('content', ''),
                    "description_preview": (chunk.get('ai_description', '')[:200] + "..." if len(chunk.get('ai_description', '')) > 200 else chunk.get('ai_description', '')) if chunk.get('ai_description') else None
                }
            }
            
            inspection_data["chunks"].append(chunk_entry)
        
        # Convert sets to lists for JSON serialization
        inspection_data["metadata"]["companies"] = sorted(list(inspection_data["metadata"]["companies"]))
        inspection_data["metadata"]["years"] = sorted(list(inspection_data["metadata"]["years"]))
        
        # Save to file
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(inspection_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Chunks inspection file saved to: {output_path.absolute()}")
        logger.info(f"Inspection summary:")
        logger.info(f"  - Total chunks: {inspection_data['metadata']['total_chunks']}")
        logger.info(f"  - Chunk types: {inspection_data['metadata']['chunk_types']}")
        logger.info(f"  - Companies: {inspection_data['metadata']['companies']}")
        logger.info(f"  - Years: {inspection_data['metadata']['years']}")
        
        return str(output_path.absolute())
        
    except Exception as e:
        logger.error(f"Failed to save chunks inspection file: {e}")
        return None