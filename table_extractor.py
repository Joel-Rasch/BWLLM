import logging
from pathlib import Path
import markdownify
from unstructured.partition.pdf import partition_pdf


def extract_tables_from_pdf(pdf_path: str):
    """
    Extract tables from PDF and return as simple list of dictionaries
    """
    logger = logging.getLogger(__name__)
    
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    logger.info(f"Extracting tables from {pdf_file.name}")
    
    try:
        pdf_elements = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",
            infer_table_structure=True,
            languages=['deu']
        )
    except Exception as e:
        logger.error(f"Failed to parse PDF {pdf_path}: {e}")
        raise

    tables = []
    texts = []
    
    for i, element in enumerate(pdf_elements):
        page_num = getattr(element.metadata, 'page_number', None)
        
        if element.category == "Table" and element.text:
            raw_html = getattr(element.metadata, 'text_as_html', None)
            
            # Convert HTML to markdown if available
            if raw_html:
                try:
                    content = markdownify.markdownify(raw_html).strip()
                except Exception:
                    content = element.text.strip()
            else:
                content = element.text.strip()
            
            if content:
                tables.append({
                    'content': content,
                    'page_number': page_num,
                    'element_index': i,
                    'type': 'table',
                    'raw_html': raw_html
                })
        
        elif element.text and element.text.strip():
            texts.append({
                'content': element.text.strip(),
                'page_number': page_num,
                'element_index': i,
                'type': 'text'
            })
    
    logger.info(f"Extracted {len(tables)} tables and {len(texts)} text elements")
    return {'tables': tables, 'texts': texts}


def extract_content_from_pdf(pdf_path: str):
    """
    Extract all content (tables and text) from PDF in order
    """
    logger = logging.getLogger(__name__)
    
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    logger.info(f"Extracting content from {pdf_file.name}")
    
    try:
        pdf_elements = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",
            infer_table_structure=True,
            languages=['deu']
        )
    except Exception as e:
        logger.error(f"Failed to parse PDF {pdf_path}: {e}")
        raise

    content = []
    
    for i, element in enumerate(pdf_elements):
        page_num = getattr(element.metadata, 'page_number', None)
        
        if not element.text or not element.text.strip():
            continue
            
        if element.category == "Table":
            raw_html = getattr(element.metadata, 'text_as_html', None)
            
            # Convert HTML to markdown if available
            if raw_html:
                try:
                    table_content = markdownify.markdownify(raw_html).strip()
                except Exception:
                    table_content = element.text.strip()
            else:
                table_content = element.text.strip()
            
            content.append({
                'content': table_content,
                'page_number': page_num,
                'element_index': i,
                'type': 'table',
                'raw_html': raw_html
            })
        else:
            content.append({
                'content': element.text.strip(),
                'page_number': page_num,
                'element_index': i,
                'type': 'text'
            })
    
    logger.info(f"Extracted {len(content)} elements total")
    return content
