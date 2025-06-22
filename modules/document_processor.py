"""
Document processing module for table detection and enhancement
"""
import re
from typing import List
from pathlib import Path
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

class DocumentProcessor:
    def __init__(self, llm, chunk_size=1000, chunk_overlap=200):
        self.llm = llm
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def detect_tables(self, text: str) -> List[str]:
        """Detect tables in markdown text"""
        tables = []
        
        # Markdown tables
        table_pattern = r'(\|[^\n]*\|\s*\n\|[-\s\|:]+\|\s*\n(?:\|[^\n]*\|\s*\n?)*)'
        tables.extend(re.findall(table_pattern, text, re.MULTILINE))
        
        # Financial data blocks
        financial_pattern = r'((?:in Mio\. €|in %|in Tsd\.|€ Mio\.|Mio\. EUR)[^\n]*\n(?:[^\n]*\d+[^\n]*\n?)+)'
        tables.extend(re.findall(financial_pattern, text, re.MULTILINE))
        
        return tables
    
    def describe_table(self, table_text: str) -> str:
        """Generate table description using LLM"""
        prompt = ChatPromptTemplate.from_template(
            "Beschreibe diese Tabelle/Daten in 1-2 Sätzen auf Deutsch:\n{table}\n\nBeschreibung:"
        )
        
        try:
            chain = prompt | self.llm | StrOutputParser()
            description = chain.invoke({"table": table_text})
            return description.strip()
        except Exception as e:
            return "Tabelle mit Finanz- oder Geschäftsdaten."
    
    def process_document(self, file_path: str) -> List[Document]:
        """Process single markdown file with table enhancement"""
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Get filename info
        filename = Path(file_path).stem
        company = filename.replace('_2023', '').replace('_', ' ')
        
        # Detect and describe tables
        tables = self.detect_tables(content)
        
        # Add table descriptions to content
        enhanced_content = content
        table_descriptions = []
        
        for table in tables:
            description = self.describe_table(table)
            enhanced_content += f"\n\n[Tabellenbeschreibung]: {description}\n"
            table_descriptions.append({
                'table': table[:200] + "...",  # Truncate for display
                'description': description
            })
        
        # Split into chunks
        chunks = self.text_splitter.split_text(enhanced_content)
        
        # Create documents
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    'company': company, 
                    'source': filename,
                    'chunk_id': i,
                    'table_count': len(tables)
                }
            )
            documents.append(doc)
        
        return documents, table_descriptions
    
    def process_all_documents(self, markdown_folder: str) -> tuple[List[Document], dict]:
        """Process all markdown files"""
        
        markdown_files = list(Path(markdown_folder).glob('*.md'))
        
        all_documents = []
        processing_stats = {
            'files_processed': 0,
            'total_chunks': 0,
            'total_tables': 0,
            'file_details': []
        }
        
        for file_path in markdown_files:
            try:
                documents, table_descriptions = self.process_document(str(file_path))
                all_documents.extend(documents)
                
                processing_stats['files_processed'] += 1
                processing_stats['total_chunks'] += len(documents)
                processing_stats['total_tables'] += len(table_descriptions)
                processing_stats['file_details'].append({
                    'filename': file_path.name,
                    'chunks': len(documents),
                    'tables': len(table_descriptions),
                    'table_descriptions': table_descriptions
                })
                
            except Exception as e:
                processing_stats['file_details'].append({
                    'filename': file_path.name,
                    'error': str(e)
                })
        
        return all_documents, processing_stats
