#!/usr/bin/env python3
"""
Document Processing Pipeline - University Demo

Processes PDFs from Geschaeftsberichte folder and creates AI-enhanced table descriptions.

Usage: python main.py
"""

import os
from pathlib import Path
import dotenv
from table_enricher import TableEnricher


def main():
    """Main function"""
    # Setup
    input_dir = Path("Geschaeftsberichte")
    output_dir = Path("Extrahierter_Text_Markdown")
    
    # Check directories
    if not input_dir.exists():
        print(f"Error: Input directory 'Geschaeftsberichte' not found")
        return
    
    output_dir.mkdir(exist_ok=True)
    
    # Get API key
    dotenv.load_dotenv()
    api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("Warning: No Gemini API key found. Table descriptions will be basic.")
        print("Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable for AI features.")
    
    # Find PDF files
    pdf_files = list(input_dir.glob("*.pdf")) + list(input_dir.glob("*.PDF"))
    
    if not pdf_files:
        print("No PDF files found in Geschaeftsberichte folder.")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process...")
    
    # Process files
    enricher = TableEnricher(api_key)
    successful = 0
    failed = 0
    total_groups = 0
    total_tables = 0
    
    for pdf_path in pdf_files:
        try:
            print(f"Processing {pdf_path.name}...")
            output_file = output_dir / f"{pdf_path.stem}_enriched.md"
            
            results = enricher.process_pdf(str(pdf_path), str(output_file))
            
            groups = len(results)
            tables = sum(len(group['tables']) for group in results)
            
            print(f"  ✓ Created {groups} groups with {tables} tables")
            
            successful += 1
            total_groups += groups
            total_tables += tables
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed += 1
    
    # Summary
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"Files processed: {successful}")
    print(f"Files failed: {failed}")
    print(f"Table groups created: {total_groups}")
    print(f"Total tables processed: {total_tables}")
    print(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    main()