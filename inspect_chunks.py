#!/usr/bin/env python3
"""
Simple script to view and analyze the processed chunks inspection file
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

def load_inspection_file(file_path: str = "processed_chunks_inspection.json") -> Dict[str, Any]:
    """Load the chunks inspection file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Inspection file not found: {file_path}")
        print("💡 Run 'python main.py' first to generate the inspection file")
        return None
    except Exception as e:
        print(f"❌ Error loading inspection file: {e}")
        return None

def print_summary(data: Dict[str, Any]):
    """Print a summary of the chunks"""
    metadata = data.get('metadata', {})
    
    print("📊 CHUNKS INSPECTION SUMMARY")
    print("=" * 50)
    print(f"📄 Total chunks: {metadata.get('total_chunks', 0)}")
    print(f"🏢 Companies: {', '.join(metadata.get('companies', []))}")
    print(f"📅 Years: {', '.join(metadata.get('years', []))}")
    print()
    
    chunk_types = metadata.get('chunk_types', {})
    print("📋 Chunk Types:")
    for chunk_type, count in chunk_types.items():
        print(f"  - {chunk_type}: {count}")
    print()

def find_table_chunks(data: Dict[str, Any]) -> list:
    """Find all table chunks for inspection"""
    table_chunks = []
    for chunk in data.get('chunks', []):
        if chunk.get('source_info', {}).get('chunk_type') == 'table':
            table_chunks.append(chunk)
    return table_chunks

def print_table_chunks_summary(table_chunks: list):
    """Print summary of table chunks and their AI descriptions"""
    print(f"📊 TABLE CHUNKS ANALYSIS ({len(table_chunks)} found)")
    print("=" * 50)
    
    enriched_count = 0
    for i, chunk in enumerate(table_chunks, 1):
        content_info = chunk.get('content_info', {})
        source_info = chunk.get('source_info', {})
        is_enriched = content_info.get('enriched', False)
        
        if is_enriched:
            enriched_count += 1
        
        print(f"\n📋 Table {i}:")
        print(f"  🏢 Company: {source_info.get('company', 'unknown')}")
        print(f"  📅 Year: {source_info.get('year', 'unknown')}")
        print(f"  📄 File: {source_info.get('source_file', 'unknown')}")
        print(f"  📏 Size: {content_info.get('character_count', 0)} chars")
        print(f"  🤖 AI Enriched: {'✅ Yes' if is_enriched else '❌ No'}")
        
        # Show content preview
        preview = chunk.get('preview', {})
        content_preview = preview.get('content_preview', 'No preview')
        print(f"  📝 Content Preview: {content_preview}")
        
        # Show AI description if available
        if is_enriched:
            description_preview = preview.get('description_preview', 'No description')
            print(f"  🤖 AI Description: {description_preview}")
    
    print(f"\n📈 ENRICHMENT STATISTICS:")
    print(f"  - Total table chunks: {len(table_chunks)}")
    print(f"  - AI enriched: {enriched_count}")
    print(f"  - Not enriched: {len(table_chunks) - enriched_count}")
    print(f"  - Enrichment rate: {(enriched_count/len(table_chunks)*100):.1f}%" if table_chunks else "  - Enrichment rate: 0%")

def interactive_chunk_viewer(data: Dict[str, Any]):
    """Interactive viewer for detailed chunk inspection"""
    chunks = data.get('chunks', [])
    
    while True:
        print(f"\n🔍 INTERACTIVE CHUNK VIEWER")
        print(f"Available chunks: 1-{len(chunks)} (or 'q' to quit)")
        
        user_input = input("Enter chunk number to view details: ").strip()
        
        if user_input.lower() in ['q', 'quit', 'exit']:
            break
            
        try:
            chunk_num = int(user_input)
            if 1 <= chunk_num <= len(chunks):
                chunk = chunks[chunk_num - 1]
                print_chunk_details(chunk)
            else:
                print(f"❌ Invalid chunk number. Please enter 1-{len(chunks)}")
        except ValueError:
            print("❌ Please enter a valid number or 'q' to quit")

def print_chunk_details(chunk: Dict[str, Any]):
    """Print detailed information about a specific chunk"""
    print("\n" + "="*60)
    print(f"📋 CHUNK #{chunk.get('chunk_id', 'unknown')} DETAILS")
    print("="*60)
    
    source_info = chunk.get('source_info', {})
    content_info = chunk.get('content_info', {})
    content = chunk.get('content', {})
    
    print(f"🏢 Company: {source_info.get('company', 'unknown')}")
    print(f"📅 Year: {source_info.get('year', 'unknown')}")
    print(f"📄 Source File: {source_info.get('source_file', 'unknown')}")
    print(f"📋 Type: {source_info.get('chunk_type', 'unknown')}")
    print(f"📏 Character Count: {content_info.get('character_count', 0)}")
    print(f"📊 Has Table: {content_info.get('has_table', False)}")
    print(f"🤖 AI Enriched: {content_info.get('enriched', False)}")
    
    print(f"\n📝 RAW CONTENT:")
    print("-" * 40)
    print(content.get('raw_content', 'No content'))
    
    if content_info.get('enriched', False) and content.get('ai_description'):
        print(f"\n🤖 AI DESCRIPTION:")
        print("-" * 40)
        print(content.get('ai_description', 'No description'))

def main():
    """Main function"""
    print("🔍 Chunks Inspection Tool")
    print("=" * 30)
    
    # Load inspection data
    data = load_inspection_file()
    if not data:
        return
    
    # Print summary
    print_summary(data)
    
    # Analyze table chunks specifically
    table_chunks = find_table_chunks(data)
    print_table_chunks_summary(table_chunks)
    
    # Interactive viewer
    print(f"\n💡 Use the interactive viewer to examine specific chunks in detail")
    user_input = input("Start interactive viewer? (y/n): ").strip().lower()
    if user_input in ['y', 'yes']:
        interactive_chunk_viewer(data)
    
    print("\n✅ Inspection complete!")

if __name__ == "__main__":
    main()