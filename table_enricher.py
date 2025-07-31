"""
Table Enricher - Extracts tables from PDFs and adds AI descriptions

This module processes PDFs to:
1. Extract tables with surrounding context
2. Group consecutive tables together
3. Generate natural language descriptions using Gemini AI
4. Save enriched content for better LLM understanding
"""

import logging
import google.generativeai as genai
import markdownify
from unstructured.partition.pdf import partition_pdf
from typing import List, Optional, Dict, Any


class TableEnricher:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        
        # Setup Gemini if API key provided
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model = None
            self.logger.warning("No API key provided - AI descriptions disabled")
    
    def process_pdf(self, pdf_path: str, output_path: str) -> List[Dict[str, Any]]:
        """Process a PDF and save enriched table content"""
        self.logger.info(f"Processing {pdf_path}")
        
        # Extract all chunks (text and tables)
        chunks = self._extract_chunks(pdf_path)
        
        # Group tables with context
        table_groups = self._group_tables(chunks)
        
        # Enrich with descriptions
        enriched_groups = []
        for group in table_groups:
            enriched = self._enrich_group(group)
            enriched_groups.append(enriched)
        
        # Save to file
        self._save_content(enriched_groups, output_path)
        
        self.logger.info(f"Created {len(enriched_groups)} table groups with {sum(len(g['tables']) for g in enriched_groups)} total tables")
        return enriched_groups
    
    def _extract_chunks(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text and table chunks from PDF"""
        try:
            elements = partition_pdf(
                filename=pdf_path,
                strategy="hi_res",
                infer_table_structure=True,
                languages=['deu']
            )
        except Exception as e:
            self.logger.error(f"Failed to parse PDF: {e}")
            raise
        
        chunks = []
        for i, element in enumerate(elements):
            page_num = getattr(element.metadata, 'page_number', None)
            
            if element.category == "Table":
                # Extract table content
                table_content = self._extract_table_content(element)
                if table_content:
                    chunks.append({
                        'type': 'table',
                        'content': table_content,
                        'page': page_num,
                        'index': i
                    })
            else:
                # Extract text content
                if element.text and element.text.strip():
                    chunks.append({
                        'type': 'text',
                        'content': element.text.strip(),
                        'page': page_num,
                        'index': i
                    })
        
        self.logger.info(f"Extracted {len(chunks)} chunks ({len([c for c in chunks if c['type'] == 'table'])} tables)")
        return chunks
    
    def _extract_table_content(self, element) -> Optional[str]:
        """Extract table content as markdown"""
        # Try HTML first (better formatting)
        html_content = getattr(element.metadata, 'text_as_html', None)
        if html_content:
            try:
                return markdownify.markdownify(html_content).strip()
            except:
                pass
        
        # Fallback to plain text
        if element.text and element.text.strip():
            return element.text.strip()
        
        return None
    
    def _group_tables(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group consecutive tables with surrounding text context"""
        groups = []
        i = 0
        
        while i < len(chunks):
            if chunks[i]['type'] == 'table':
                # Start new table group
                group_tables = [chunks[i]]
                start_idx = i
                i += 1
                
                # Collect consecutive tables (max 5 per group)
                while i < len(chunks) and chunks[i]['type'] == 'table' and len(group_tables) < 5:
                    group_tables.append(chunks[i])
                    i += 1
                
                # Find context
                prev_text = self._find_previous_text(chunks, start_idx)
                next_text = self._find_next_text(chunks, i - 1)
                
                groups.append({
                    'tables': group_tables,
                    'previous_text': prev_text,
                    'next_text': next_text,
                    'start_page': group_tables[0]['page'],
                    'end_page': group_tables[-1]['page']
                })
            else:
                i += 1
        
        self.logger.info(f"Created {len(groups)} table groups")
        return groups
    
    def _find_previous_text(self, chunks: List[Dict[str, Any]], table_start: int) -> Optional[str]:
        """Find closest preceding text chunk"""
        for i in range(table_start - 1, -1, -1):
            if chunks[i]['type'] == 'text':
                return chunks[i]['content']
        return None
    
    def _find_next_text(self, chunks: List[Dict[str, Any]], table_end: int) -> Optional[str]:
        """Find closest following text chunk"""
        for i in range(table_end + 1, len(chunks)):
            if chunks[i]['type'] == 'text':
                return chunks[i]['content']
        return None
    
    def _enrich_group(self, group: Dict[str, Any]) -> Dict[str, Any]:
        """Add AI description to table group"""
        if self.model:
            description = self._generate_ai_description(group)
        else:
            description = "AI-Beschreibung nicht verfügbar (kein API-Schlüssel)"
        
        return {
            'tables': group['tables'],
            'previous_text': group['previous_text'],
            'next_text': group['next_text'],
            'pages': f"{group['start_page']}-{group['end_page']}" if group['start_page'] != group['end_page'] else str(group['start_page']),
            'ai_description': description,
            'table_count': len(group['tables'])
        }
    
    def _generate_ai_description(self, group: Dict[str, Any]) -> str:
        """Generate AI description for table group"""
        # Build context
        prev_context = group['previous_text'][:500] + "..." if group['previous_text'] and len(group['previous_text']) > 500 else group['previous_text'] or "N/A"
        next_context = group['next_text'][:500] + "..." if group['next_text'] and len(group['next_text']) > 500 else group['next_text'] or "N/A"
        
        # Build tables content
        tables_content = "\n\n".join([
            f"=== TABELLE {i+1} (Seite {table['page']}) ===\n{table['content']}"
            for i, table in enumerate(group['tables'])
        ])
        
        prompt = f"""
Analysiere die folgenden Tabellen und ihren Kontext. Erstelle eine natürliche, verständliche Beschreibung.

KONTEXT VOR DEN TABELLEN:
{prev_context}

TABELLENGRUPPE ({len(group['tables'])} Tabellen):
{tables_content}

KONTEXT NACH DEN TABELLEN:
{next_context}

Bitte erstelle eine präzise Beschreibung (max. 200 Wörter), die erklärt:
1. Was diese Tabellen zeigen
2. Wichtige Erkenntnisse oder Muster
3. Bezug zum umgebenden Kontext
4. Beziehung zwischen den Tabellen (falls mehrere)

Antworte auf Deutsch in einem zusammenhängenden Absatz.
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            self.logger.error(f"AI description failed: {e}")
            return f"Fehler bei der KI-Beschreibung: {str(e)}"
    
    def _save_content(self, enriched_groups: List[Dict[str, Any]], output_path: str):
        """Save enriched content to markdown file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Angereicherte Tabellen mit KI-Beschreibungen\n\n")
            
            for i, group in enumerate(enriched_groups, 1):
                f.write(f"## Tabellengruppe {i}\n\n")
                f.write(f"**Seite(n):** {group['pages']}  \n")
                f.write(f"**Anzahl Tabellen:** {group['table_count']}\n\n")
                
                f.write("### KI-Beschreibung\n")
                f.write(f"{group['ai_description']}\n\n")
                
                f.write("### Kontext\n")
                f.write("**Vorheriger Text:**\n")
                f.write(f"{group['previous_text'] or 'Nicht verfügbar'}\n\n")
                
                f.write("**Nachfolgender Text:**\n")
                f.write(f"{group['next_text'] or 'Nicht verfügbar'}\n\n")
                
                f.write("### Tabellen\n")
                for j, table in enumerate(group['tables'], 1):
                    f.write(f"#### Tabelle {j} (Seite {table['page']})\n")
                    f.write(f"{table['content']}\n\n")
                
                f.write("---\n\n")
        
        self.logger.info(f"Saved enriched content to {output_path}")