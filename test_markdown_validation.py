#!/usr/bin/env python3
"""
Test script to demonstrate the new markdown validation feature
"""

import logging
from pathlib import Path
from src.processing.chunk_processor import chunk_document, save_extraction_as_markdown

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_markdown_validation():
    """Test the markdown validation feature"""
    
    # Example usage with a PDF file (if available)
    test_files = []
    
    # Look for any PDF files in the current directory
    for pdf_file in Path(".").glob("*.pdf"):
        test_files.append(str(pdf_file))
    
    # Look for any markdown files in the current directory  
    for md_file in Path(".").glob("*.md"):
        test_files.append(str(md_file))
    
    if not test_files:
        logger.warning("No test files found in current directory")
        return
    
    for test_file in test_files[:2]:  # Test only first 2 files
        logger.info(f"Testing extraction with validation for: {test_file}")
        
        try:
            # Process the document with markdown validation enabled
            chunks = chunk_document(test_file, create_validation_md=True)
            logger.info(f"Successfully processed {test_file}: {len(chunks)} chunks created")
            
        except Exception as e:
            logger.error(f"Failed to process {test_file}: {e}")

def create_sample_content_and_test():
    """Create sample content and test the markdown validation"""
    
    # Sample extracted content (simulating PDF extraction)
    sample_content = [
        {
            'content': 'This is a sample text paragraph from page 1.',
            'type': 'text',
            'page_number': 1
        },
        {
            'content': '| Column 1 | Column 2 | Column 3 |\n|----------|----------|----------|\n| Value A  | Value B  | Value C  |\n| Value X  | Value Y  | Value Z  |',
            'type': 'table',
            'page_number': 2
        },
        {
            'content': 'Another text paragraph that continues the document content with more detailed information about the company operations.',
            'type': 'text',
            'page_number': 2
        }
    ]
    
    # Test the markdown creation function directly
    md_path = save_extraction_as_markdown(sample_content, "Sample_Company_2023.pdf")
    if md_path:
        logger.info(f"Sample validation markdown created at: {md_path}")
        
        # Show the content
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print("\nGenerated Validation Markdown:")
        print("=" * 50)
        print(content[:800] + "..." if len(content) > 800 else content)

if __name__ == "__main__":
    logger.info("Testing markdown validation feature...")
    
    # Test with sample content first
    create_sample_content_and_test()
    
    # Then test with actual files if available
    test_markdown_validation()
    
    logger.info("Testing completed!")