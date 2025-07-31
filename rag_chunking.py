import os
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import tiktoken
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
import hashlib
import json
from dataclasses import dataclass
import requests
from datetime import datetime


@dataclass
class ExtractedTable:
    """Datenklasse für extrahierte Tabellen"""
    raw_content: str
    before_context: str
    after_context: str
    position: int
    source_file: str


class LLMTableProcessor:
    """
    LLM-basierter Tabellenprozessor für kontextuelle Anreicherung
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        
    def generate_contextual_description(self, table: ExtractedTable) -> str:
        """
        Generiert kontextuelle Beschreibung mit LLM
        """
        if not self.api_key:
            # Fallback ohne LLM
            return self._generate_rule_based_description(table)
        
        prompt = f"""
        Du bist ein Experte für Finanzanalyse. Analysiere die folgende Tabelle aus einem Geschäftsbericht und erstelle eine präzise, kontextuelle Beschreibung.
        
        KONTEXT DAVOR:
        {table.before_context}
        
        TABELLE:
        {table.raw_content}
        
        KONTEXT DANACH:
        {table.after_context}
        
        Erstelle eine präzise Beschreibung die:
        1. Den Tabelleninhalt und Zweck erklärt
        2. Wichtige Kennzahlen und Trends hervorhebt
        3. Den Geschäftskontext berücksichtigt
        4. Suchrelevante Schlüsselbegriffe verwendet
        
        Antwort (nur die Beschreibung, max 200 Wörter):
        """
        
        try:
            return self._call_llm(prompt)
        except Exception as e:
            print(f"LLM-Aufruf fehlgeschlagen: {e}")
            return self._generate_rule_based_description(table)
    
    def standardize_table_format(self, table: ExtractedTable) -> str:
        """
        Standardisiert Tabellenformat mit LLM
        """
        if not self.api_key:
            return self._clean_table_format(table.raw_content)
        
        prompt = f"""
        Konvertiere die folgende Tabelle in ein sauberes, einheitliches Markdown-Format:
        
        ORIGINAL TABELLE:
        {table.raw_content}
        
        Anforderungen:
        1. Verwende standard Markdown-Tabellensyntax
        2. Stelle sicher dass alle Spalten korrekt ausgerichtet sind
        3. Entferne überflüssige Zeichen und Formatierung
        4. Behalte alle Daten und Zahlen bei
        5. Füge aussagekräftige Header hinzu falls fehlend
        
        Antwort (nur die formatierte Tabelle):
        """
        
        try:
            return self._call_llm(prompt)
        except Exception as e:
            print(f"Format-Standardisierung fehlgeschlagen: {e}")
            return self._clean_table_format(table.raw_content)
    
    def _call_llm(self, prompt: str) -> str:
        """
        Placeholder für LLM-Aufruf - kann mit OpenAI API, Ollama etc. implementiert werden
        """
        # Hier würde der echte LLM-Aufruf stehen
        # Für jetzt verwenden wir rule-based fallback
        raise Exception("LLM nicht verfügbar")
    
    def _generate_rule_based_description(self, table: ExtractedTable) -> str:
        """
        Rule-based Fallback für Tabellenbeschreibung
        """
        content = table.raw_content.lower()
        context = f"{table.before_context} {table.after_context}".lower()
        
        description_parts = []
        
        # Identifiziere Tabellentyp
        if 'umsatz' in content or 'revenue' in content:
            description_parts.append("Umsatz- und Ertragskennzahlen")
        if 'ebit' in content or 'gewinn' in content:
            description_parts.append("Profitabilitätskennzahlen")
        if 'mitarbeiter' in content or 'personal' in content:
            description_parts.append("Personalzahlen")
        if 'bilanz' in content or 'vermögen' in content:
            description_parts.append("Bilanzpositionen")
        if 'cashflow' in content:
            description_parts.append("Cashflow-Kennzahlen")
        
        # Zeitbezug
        if '2023' in content and '2022' in content:
            description_parts.append("Jahresvergleich 2023 vs 2022")
        elif '2023' in content:
            description_parts.append("Kennzahlen für 2023")
        
        # Geschäftsbereich aus Kontext
        if 'automotive' in context:
            description_parts.append("Automotive-Bereich")
        if 'tires' in context or 'reifen' in context:
            description_parts.append("Reifen-Geschäft")
        if 'contitech' in context:
            description_parts.append("ContiTech-Segment")
        
        if description_parts:
            description = f"Tabelle mit {', '.join(description_parts[:3])}. "
        else:
            description = "Geschäftskennzahlen-Tabelle. "
        
        # Extrahiere wichtige Werte
        lines = table.raw_content.split('\n')
        important_values = []
        for line in lines[:8]:  # Erste 8 Zeilen
            if '|' in line and '---' not in line:  # Skip separator lines
                cells = [cell.strip() for cell in line.split('|')]
                # Remove empty first/last cells if present
                if cells and not cells[0]:
                    cells = cells[1:]
                if cells and not cells[-1]:
                    cells = cells[:-1]
                
                if len(cells) >= 2 and cells[0] and cells[1]:
                    # Skip header row
                    if cells[0].lower() in ['mio €', 'mio', '€', 'jahr', 'year']:
                        continue
                    # Suche nach numerischen Werten
                    for cell in cells[1:]:
                        # Bereinige Zellinhalt aggressiv
                        clean_cell = cell.replace('< /dev/null', '').replace('null', '').strip()
                        clean_cell = clean_cell.replace('<', '').replace('/dev/', '').replace('\\n', ' ')
                        clean_cell = ' '.join(clean_cell.split())  # Normalisiere Whitespace
                        if any(char.isdigit() for char in clean_cell) and len(clean_cell) > 0:
                            clean_label = cells[0].replace('< /dev/null', '').replace('null', '').strip()
                            clean_label = clean_label.replace('<', '').replace('/dev/', '').replace('\\n', ' ')
                            clean_label = ' '.join(clean_label.split())
                            important_values.append(f"{clean_label}: {clean_cell}")
                            break
        
        if important_values:
            description += f"Wichtige Kennzahlen: {'; '.join(important_values[:3])}."
        
        # Finale Bereinigung der Beschreibung
        clean_description = description.replace('< /dev/null', '').replace('null', '')
        clean_description = clean_description.replace('<', '').replace('/dev/', '').replace('\\n', ' ')
        clean_description = ' '.join(clean_description.split())
        
        return clean_description
    
    def _clean_table_format(self, raw_content: str) -> str:
        """
        Reinigt und standardisiert Tabellenformat ohne LLM
        """
        lines = raw_content.strip().split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if '|' in line:
                # Bereinige Tabellenzeilen
                cells = [cell.strip() for cell in line.split('|')]
                # Entferne leere erste/letzte Zellen falls vorhanden
                if cells and not cells[0]:
                    cells = cells[1:]
                if cells and not cells[-1]:
                    cells = cells[:-1]
                
                # Bereinige Zellinhalt
                cleaned_cells = []
                for cell in cells:
                    # Entferne problematische Zeichen und normalisiere
                    cell = cell.replace('< /dev/null', '').replace('null', '').strip()
                    # Entferne weitere problematische Zeichen
                    cell = cell.replace('<', '').replace('/dev/', '').replace('\\n', ' ')
                    # Normalisiere Whitespace
                    cell = ' '.join(cell.split())
                    if cell:  # Nur nicht-leere Zellen
                        cleaned_cells.append(cell)
                
                if cleaned_cells:  # Nur nicht-leere Zeilen
                    cleaned_line = '| ' + ' | '.join(cleaned_cells) + ' |'
                    cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines)


class PreciseTableExtractor:
    """
    Präziser Tabellen-Extraktor mit sauberer Trennung
    """
    
    def __init__(self, table_context_size: int = 500):
        self.table_context_size = table_context_size
        self.llm_processor = LLMTableProcessor()
        
    def extract_all_tables(self, content: str, source_file: str) -> List[ExtractedTable]:
        """
        Step 1: Precise Extraction - Cleanly extract all tables from document
        """
        tables = []
        
        # Pattern für Tabellen-Start/Ende Marker
        table_pattern = r'--- Tabelle Start ---(.*?)--- Tabelle Ende ---'
        
        # Finde alle Tabellen mit ihren Positionen
        for match_idx, match in enumerate(re.finditer(table_pattern, content, re.DOTALL)):
            start_pos = match.start()
            end_pos = match.end()
            table_content = match.group(1).strip()
            
            # Extrahiere Kontext vor und nach der Tabelle
            context_start = max(0, start_pos - self.table_context_size)
            before_context = content[context_start:start_pos].strip()
            
            context_end = min(len(content), end_pos + self.table_context_size)
            after_context = content[end_pos:context_end].strip()
            
            # Erstelle ExtractedTable Objekt
            extracted_table = ExtractedTable(
                raw_content=table_content,
                before_context=before_context,
                after_context=after_context,
                position=match_idx,
                source_file=source_file
            )
            
            tables.append(extracted_table)
        
        return tables
    
    def process_table_with_llm(self, table: ExtractedTable) -> Dict[str, Any]:
        """
        Steps 2-4: Contextual Enrichment, Format Standardization, Unified Embedding
        """
        # Step 2: Contextual Enrichment
        contextual_description = self.llm_processor.generate_contextual_description(table)
        
        # Step 3: Format Standardization  
        standardized_markdown = self.llm_processor.standardize_table_format(table)
        
        # Step 4: Unified Embedding - Combine description with formatted table
        # Bereinige alle Inhalte von problematischen Zeichen
        clean_description = contextual_description.replace('< /dev/null', '').replace('null', '')
        clean_description = clean_description.replace('<', '').replace('/dev/', '').replace('\\n', ' ')
        clean_description = ' '.join(clean_description.split())
        
        clean_markdown = standardized_markdown.replace('< /dev/null', '').replace('null', '')
        clean_markdown = clean_markdown.replace('<', '').replace('/dev/', '')
        
        clean_context = table.before_context[:200].replace('< /dev/null', '').replace('null', '')
        clean_context = clean_context.replace('<', '').replace('/dev/', '')
        
        unified_content = f"""{clean_description}

TABELLE:
{clean_markdown}

DATENQUELLE: {table.source_file}
KONTEXT: {clean_context}..."""
        
        return {
            'content': unified_content,
            'type': 'table_llm_enhanced',
            'metadata': {
                'source_file': table.source_file,
                'table_position': table.position,
                'has_table': True,
                'contextual_description': contextual_description,
                'standardized_format': standardized_markdown,
                'context_before': table.before_context[:200],  
                'context_after': table.after_context[:200],
                'extraction_timestamp': datetime.now().isoformat()
            }
        }
        
    def count_tokens(self, text: str) -> int:
        """Zählt Tokens in einem Text"""
        return len(self.tokenizer.encode(text))
    
    def extract_tables_and_context(self, content: str) -> List[Dict[str, Any]]:
        """
        Extrahiert Tabellen mit ihrem Kontext für bessere Auffindbarkeit
        """
        chunks = []
        
        # Pattern für Tabellen-Start/Ende Marker
        table_pattern = r'--- Tabelle Start ---(.*?)--- Tabelle Ende ---'
        
        # Finde alle Tabellen
        tables = re.finditer(table_pattern, content, re.DOTALL)
        last_end = 0
        
        for match in tables:
            start_pos = match.start()
            end_pos = match.end()
            table_content = match.group(1).strip()
            
            # Kontext vor der Tabelle
            context_start = max(0, start_pos - self.table_context_size)
            before_context = content[context_start:start_pos].strip()
            
            # Kontext nach der Tabelle
            context_end = min(len(content), end_pos + self.table_context_size)
            after_context = content[end_pos:context_end].strip()
            
            # Text vor der Tabelle chunken
            if last_end < start_pos:
                text_before = content[last_end:start_pos].strip()
                if text_before:
                    text_chunks = self._chunk_text(text_before)
                    chunks.extend(text_chunks)
            
            # Verbesserte Tabellen-Verarbeitung
            table_data = self._parse_table_structure(table_content)
            
            if table_data and table_data.get('headers'):
                # Strukturierte Tabelle - erstelle erweiterte Repräsentation
                enhanced_chunks = self._create_enhanced_table_chunks(
                    table_data, before_context, after_context
                )
                
                # Prüfe Größe und teile bei Bedarf auf
                for chunk in enhanced_chunks:
                    if self.count_tokens(chunk['content']) > self.max_chunk_size:
                        split_chunks = self._chunk_large_table(table_content, before_context, after_context)
                        chunks.extend(split_chunks)
                    else:
                        chunks.append(chunk)
            else:
                # Fallback für unparsbare Tabellen
                full_table_chunk = f"{before_context}\n\n--- TABELLE ---\n{table_content}\n--- ENDE TABELLE ---\n\n{after_context}".strip()
                
                if self.count_tokens(full_table_chunk) > self.max_chunk_size:
                    table_chunks = self._chunk_large_table(table_content, before_context, after_context)
                    chunks.extend(table_chunks)
                else:
                    chunks.append({
                        'content': full_table_chunk,
                        'type': 'table',
                        'metadata': {
                            'has_table': True,
                            'table_rows': table_content.count('|'),
                            'context_before': before_context[:100],
                            'context_after': after_context[:100]
                        }
                    })
            
            last_end = end_pos
        
        # Verbleibender Text nach der letzten Tabelle
        if last_end < len(content):
            remaining_text = content[last_end:].strip()
            if remaining_text:
                text_chunks = self._chunk_text(remaining_text)
                chunks.extend(text_chunks)
        
        return chunks
    
    def _chunk_large_table(self, table_content: str, before_context: str, after_context: str) -> List[Dict[str, Any]]:
        """
        Teilt große Tabellen in kleinere Chunks auf, behält dabei Header
        """
        chunks = []
        
        # Parse Tabelle zu strukturierten Daten
        table_data = self._parse_table_structure(table_content)
        
        if table_data and table_data.get('headers') and table_data.get('rows'):
            # Strukturierte Verarbeitung
            chunks.extend(self._create_enhanced_table_chunks(
                table_data, before_context, after_context, is_part=True
            ))
        else:
            # Fallback: ursprüngliche Methode
            chunks.extend(self._create_simple_table_chunks(
                table_content, before_context, after_context
            ))
        
        return chunks
    
    def _parse_table_structure(self, table_content: str) -> Dict[str, Any]:
        """
        Parst Markdown-Tabelle in strukturierte Daten
        """
        lines = [line.strip() for line in table_content.split('\n') if line.strip()]
        
        if len(lines) < 2:
            return {}
        
        headers = []
        rows = []
        separator_found = False
        
        for i, line in enumerate(lines):
            if '|' not in line:
                continue
                
            cells = [cell.strip() for cell in line.split('|')[1:-1]]  # Entferne erste/letzte leere Zelle
            
            if not separator_found and '---' in line:
                separator_found = True
                continue
            elif not separator_found:
                headers = cells
            else:
                if cells and any(cell for cell in cells):  # Ignoriere leere Zeilen
                    rows.append(cells)
        
        return {
            'headers': headers,
            'rows': rows,
            'raw_content': table_content
        }
    
    def _create_enhanced_table_chunks(self, table_data: Dict[str, Any], 
                                    before_context: str, after_context: str, 
                                    is_part: bool = False) -> List[Dict[str, Any]]:
        """
        Erstellt verbesserte Tabellen-Chunks mit mehreren Repräsentationen
        """
        headers = table_data.get('headers', [])
        rows = table_data.get('rows', [])
        
        # 1. Semantische Beschreibung erstellen
        semantic_desc = self._create_table_semantic_description(table_data, before_context, after_context)
        
        # 2. Key-Value Repräsentation
        key_value_text = self._create_key_value_representation(table_data)
        
        # 3. Vollständige Tabelle als Text
        full_table_text = self._create_readable_table_text(table_data)
        
        # Kombiniere alle Repräsentationen
        combined_content = f"{semantic_desc}\n\n{full_table_text}\n\n{key_value_text}"
        
        if before_context:
            combined_content = f"{before_context}\n\n{combined_content}"
        if after_context:
            combined_content = f"{combined_content}\n\n{after_context}"
        
        chunk_type = 'table_part_enhanced' if is_part else 'table_enhanced'
        
        return [{
            'content': combined_content,
            'type': chunk_type,
            'metadata': {
                'has_table': True,
                'is_table_part': is_part,
                'table_rows': len(rows),
                'table_columns': len(headers),
                'context_before': before_context[:100],
                'context_after': after_context[:100],
                'semantic_tags': self._extract_semantic_tags(table_data)
            }
        }]
    
    def _create_table_semantic_description(self, table_data: Dict[str, Any], 
                                         before_context: str, after_context: str) -> str:
        """Erstellt semantische Beschreibung der Tabelle"""
        headers = table_data.get('headers', [])
        rows = table_data.get('rows', [])
        
        description = "TABELLEN-INHALT: "
        
        # Identifiziere Tabellentyp basierend auf Headers
        if any('umsatz' in str(h).lower() for h in headers):
            description += "Finanzielle Kennzahlen mit Umsatzdaten. "
        if any('ebit' in str(h).lower() for h in headers):
            description += "EBIT und Ergebniskennzahlen. "
        if any('mitarbeiter' in str(h).lower() for h in headers):
            description += "Personalzahlen und Mitarbeiterdaten. "
        if any(str(h).isdigit() or '20' in str(h) for h in headers):
            description += "Jahresvergleich mit zeitlichen Daten. "
        
        description += f"Die Tabelle enthält {len(rows)} Datenzeilen mit {len(headers)} Spalten. "
        
        if headers:
            description += f"Spalten: {', '.join(headers)}. "
        
        # Wichtige Werte aus ersten Zeilen
        important_values = []
        for row in rows[:3]:  # Nur erste 3 Zeilen
            if len(row) >= 2:
                label = row[0].strip()
                value = row[1].strip() if len(row) > 1 else ''
                if label and value:
                    important_values.append(f"{label}: {value}")
        
        if important_values:
            description += f"Wichtige Werte: {'; '.join(important_values)}."
        
        return description
    
    def _create_key_value_representation(self, table_data: Dict[str, Any]) -> str:
        """Erstellt Key-Value Repräsentation für bessere Suchbarkeit"""
        headers = table_data.get('headers', [])
        rows = table_data.get('rows', [])
        
        kv_text = "DATEN-WERTE:\n"
        
        for row in rows:
            if len(row) >= 2:  # Mindestens Label und Wert
                label = row[0].strip()
                if label:
                    values = [cell.strip() for cell in row[1:] if cell.strip()]
                    if values:
                        kv_text += f"{label}: {' | '.join(values)}\n"
        
        return kv_text
    
    def _create_readable_table_text(self, table_data: Dict[str, Any]) -> str:
        """Erstellt lesbare Textrepräsentation der Tabelle"""
        headers = table_data.get('headers', [])
        rows = table_data.get('rows', [])
        
        table_text = "TABELLE:\n"
        
        if headers:
            table_text += "Spalten: " + " | ".join(headers) + "\n\n"
        
        for i, row in enumerate(rows):
            row_text = f"Zeile {i+1}: "
            if headers and len(row) == len(headers):
                pairs = [f"{h}: {v}" for h, v in zip(headers, row) if v.strip()]
                row_text += " | ".join(pairs)
            else:
                row_text += " | ".join([cell for cell in row if cell.strip()])
            table_text += row_text + "\n"
        
        return table_text
    
    def _extract_semantic_tags(self, table_data: Dict[str, Any]) -> List[str]:
        """Extrahiert semantische Tags für Metadaten"""
        tags = []
        headers = table_data.get('headers', [])
        rows = table_data.get('rows', [])
        
        # Header-basierte Tags
        header_text = ' '.join(headers).lower()
        if 'umsatz' in header_text:
            tags.append('umsatz')
        if 'ebit' in header_text:
            tags.append('ebit')
        if 'gewinn' in header_text:
            tags.append('gewinn')
        if 'mitarbeiter' in header_text:
            tags.append('mitarbeiter')
        if any(str(h).isdigit() for h in headers):
            tags.append('jahresvergleich')
        
        # Inhalts-basierte Tags
        if rows:
            first_column = [row[0] if row else '' for row in rows[:5]]
            first_col_text = ' '.join(first_column).lower()
            
            financial_keywords = ['dividende', 'cashflow', 'eigenkapital', 'schulden', 'marge']
            for keyword in financial_keywords:
                if keyword in first_col_text:
                    tags.append(keyword)
        
        return list(set(tags))
    
    def _create_simple_table_chunks(self, table_content: str, before_context: str, 
                                   after_context: str) -> List[Dict[str, Any]]:
        """Fallback für Tabellen die nicht geparst werden können"""
        lines = table_content.split('\n')
        
        # Identifiziere Header (erste 2-3 Zeilen normalerweise)
        header_lines = []
        data_lines = []
        
        for i, line in enumerate(lines):
            if i < 3 or '---' in line or not line.strip():
                header_lines.append(line)
            elif '|' in line:
                data_lines.append(line)
        
        header_text = '\n'.join(header_lines)
        
        # Teile Datenzeilen in Gruppen
        chunks = []
        current_chunk_lines = []
        current_size = len(before_context) + len(header_text) + len(after_context)
        
        for line in data_lines:
            line_tokens = self.count_tokens(line)
            
            if current_size + line_tokens > self.max_chunk_size and current_chunk_lines:
                # Erstelle Chunk
                chunk_table = header_text + '\n' + '\n'.join(current_chunk_lines)
                full_chunk = f"{before_context}\n\n--- TABELLE (TEIL) ---\n{chunk_table}\n--- ENDE TABELLE ---\n\n{after_context}".strip()
                
                chunks.append({
                    'content': full_chunk,
                    'type': 'table_part',
                    'metadata': {
                        'has_table': True,
                        'is_table_part': True,
                        'table_rows': len(current_chunk_lines),
                        'context_before': before_context[:100],
                        'context_after': after_context[:100]
                    }
                })
                
                current_chunk_lines = []
                current_size = len(before_context) + len(header_text) + len(after_context)
            
            current_chunk_lines.append(line)
            current_size += line_tokens
        
        # Letzter Chunk
        if current_chunk_lines:
            chunk_table = header_text + '\n' + '\n'.join(current_chunk_lines)
            full_chunk = f"{before_context}\n\n--- TABELLE (TEIL) ---\n{chunk_table}\n--- ENDE TABELLE ---\n\n{after_context}".strip()
            
            chunks.append({
                'content': full_chunk,
                'type': 'table_part',
                'metadata': {
                    'has_table': True,
                    'is_table_part': True,
                    'table_rows': len(current_chunk_lines),
                    'context_before': before_context[:100],
                    'context_after': after_context[:100]
                }
            })
        
        return chunks
    
    def _chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Chunking für reinen Text (ohne Tabellen)
        """
        chunks = []
        sentences = re.split(r'[.!?]\s+', text)
        
        current_chunk = ""
        current_size = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_tokens = self.count_tokens(sentence)
            
            if current_size + sentence_tokens > self.max_chunk_size and current_chunk:
                chunks.append({
                    'content': current_chunk.strip(),
                    'type': 'text',
                    'metadata': {
                        'has_table': False,
                        'word_count': len(current_chunk.split()),
                        'char_count': len(current_chunk)
                    }
                })
                
                # Overlap: behalte letzten Teil
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
                current_size = self.count_tokens(current_chunk)
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_size += sentence_tokens
        
        # Letzter Chunk
        if current_chunk.strip():
            chunks.append({
                'content': current_chunk.strip(),
                'type': 'text',
                'metadata': {
                    'has_table': False,
                    'word_count': len(current_chunk.split()),
                    'char_count': len(current_chunk)
                }
            })
        
        return chunks
    
    def extract_and_process_content(self, content: str, source_file: str) -> List[Dict[str, Any]]:
        """
        Hauptmethode: Kombiniert alle Schritte der präzisen Tabellenverarbeitung
        """
        all_chunks = []
        
        # Step 1: Extrahiere alle Tabellen präzise
        extracted_tables = self.extract_all_tables(content, source_file)
        
        print(f"  -> {len(extracted_tables)} Tabellen extrahiert")
        
        # Steps 2-4: Verarbeite jede Tabelle mit LLM
        for table in extracted_tables:
            try:
                processed_chunk = self.process_table_with_llm(table)
                all_chunks.append(processed_chunk)
                print(f"    -> Tabelle {table.position} verarbeitet")
            except Exception as e:
                print(f"    -> Fehler bei Tabelle {table.position}: {e}")
                # Fallback auf einfache Verarbeitung
                fallback_chunk = self._create_fallback_chunk(table)
                all_chunks.append(fallback_chunk)
        
        # Verarbeite verbleibenden Text (ohne Tabellen)
        text_chunks = self._extract_remaining_text(content, extracted_tables, source_file)
        all_chunks.extend(text_chunks)
        
        return all_chunks
    
    def _create_fallback_chunk(self, table: ExtractedTable) -> Dict[str, Any]:
        """Fallback wenn LLM-Verarbeitung fehlschlägt"""
        # Einfache rule-based Beschreibung
        description = self.llm_processor._generate_rule_based_description(table)
        
        # Einfache Formatbereinigung
        cleaned_format = self.llm_processor._clean_table_format(table.raw_content)
        
        unified_content = f"""{description}

TABELLE:
{cleaned_format}

DATENQUELLE: {table.source_file}"""
        
        return {
            'content': unified_content,
            'type': 'table_fallback',
            'metadata': {
                'source_file': table.source_file,
                'table_position': table.position,
                'has_table': True,
                'context_before': table.before_context[:200],
                'context_after': table.after_context[:200],
                'processing_mode': 'fallback'
            }
        }
    
    def _extract_remaining_text(self, content: str, extracted_tables: List[ExtractedTable], source_file: str) -> List[Dict[str, Any]]:
        """Extrahiert den verbleibenden Text zwischen den Tabellen"""
        if not extracted_tables:
            # Kein Tabelleninhalt, verarbeite gesamten Text
            return self._chunk_text_content(content, source_file)
        
        text_chunks = []
        table_pattern = r'--- Tabelle Start ---.*?--- Tabelle Ende ---'
        
        # Entferne alle Tabellen aus dem Content
        text_only_content = re.sub(table_pattern, '[TABELLE_ENTFERNT]', content, flags=re.DOTALL)
        
        # Teile Text in Abschnitte und verarbeite sie
        text_sections = text_only_content.split('[TABELLE_ENTFERNT]')
        
        for i, section in enumerate(text_sections):
            section = section.strip()
            if section and len(section) > 50:  # Reduziere Mindestlänge für mehr Textabschnitte
                section_chunks = self._chunk_text_content(section, source_file, section_id=i)
                text_chunks.extend(section_chunks)
                print(f"    -> Textabschnitt {i}: {len(section_chunks)} Chunks erstellt")
        
        return text_chunks
    
    def _chunk_text_content(self, text: str, source_file: str, section_id: int = 0) -> List[Dict[str, Any]]:
        """Chunking für reinen Textinhalt"""
        # Einfaches Sentence-based Chunking
        sentences = re.split(r'[.!?]\s+', text)
        
        chunks = []
        current_chunk = ""
        max_chunk_length = 1000  # Zeichen
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if len(current_chunk + sentence) > max_chunk_length and current_chunk:
                chunks.append({
                    'content': current_chunk.strip(),
                    'type': 'text',
                    'metadata': {
                        'source_file': source_file,
                        'section_id': section_id,
                        'has_table': False,
                        'char_count': len(current_chunk),
                        'processing_mode': 'text_chunking'
                    }
                })
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Letzter Chunk
        if current_chunk.strip():
            chunks.append({
                'content': current_chunk.strip(),
                'type': 'text',
                'metadata': {
                    'source_file': source_file,
                    'section_id': section_id,
                    'has_table': False,
                    'char_count': len(current_chunk),
                    'processing_mode': 'text_chunking'
                }
            })
        
        return chunks


class EnhancedRAGEmbeddingSystem:
    """
    Embedding und Retrieval System für tabellenreiche Dokumente
    """
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/distiluse-base-multilingual-cased",
                 index_path: str = "enhanced_faiss_index"):
        self.model = SentenceTransformer(model_name)
        self.index_path = index_path
        self.table_extractor = PreciseTableExtractor()
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
        
    def process_documents(self, markdown_dir: str) -> None:
        """
        Verarbeitet alle Markdown-Dateien mit präziser Tabellenextraktion
        """
        print("Verarbeite Markdown-Dateien mit LLM-basierter Tabellenverarbeitung...")
        
        all_chunks = []
        markdown_files = list(Path(markdown_dir).glob("*.md"))
        
        for md_file in markdown_files:
            print(f"Verarbeite: {md_file.name}")
            
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Präzise Tabellenextraktion und -verarbeitung
            doc_chunks = self.table_extractor.extract_and_process_content(content, md_file.name)
            
            # Metadata erweitern
            for chunk in doc_chunks:
                chunk['metadata']['file_hash'] = self._get_file_hash(str(md_file))
            
            all_chunks.extend(doc_chunks)
        
        # Statistiken ausgeben
        table_chunks = [c for c in all_chunks if c['metadata']['has_table']]
        text_chunks = [c for c in all_chunks if not c['metadata']['has_table']]
        
        print(f"Insgesamt {len(all_chunks)} Chunks erstellt:")
        print(f"  -> {len(table_chunks)} Tabellen-Chunks")
        print(f"  -> {len(text_chunks)} Text-Chunks")
        
        # Beispiel für verbessertes Tabellen-Chunk
        if table_chunks:
            example_chunk = table_chunks[0]
            print(f"\nBeispiel Tabellen-Chunk (erste 300 Zeichen):")
            print(f"Type: {example_chunk['type']}")
            print(f"Content: {example_chunk['content'][:300]}...")
        
        # Embeddings erstellen
        print("\nErstelle Embeddings...")
        self._create_embeddings(all_chunks)
        
        # Index speichern
        self._save_index()
        print(f"Verbesserter Index gespeichert in: {self.index_path}")
    
    def _create_embeddings(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Erstellt Embeddings für alle Chunks
        """
        self.chunks = chunks
        self.chunk_metadata = [chunk['metadata'] for chunk in chunks]
        
        # Embeddings in Batches erstellen
        texts = [chunk['content'] for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        
        # FAISS Index erstellen
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product für Kosinus-Ähnlichkeit
        
        # Normalisiere Embeddings für Kosinus-Ähnlichkeit
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
    
    def search(self, query: str, top_k: int = 5, score_threshold: float = 0.1, prefer_tables: bool = False) -> List[Dict[str, Any]]:
        """
        Sucht ähnliche Chunks für eine Anfrage
        """
        if self.index is None:
            self._load_index()
        
        # Query Embedding
        query_embedding = self.model.encode([query]).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Suche mit mehr Kandidaten wenn Tabellen bevorzugt werden
        search_k = top_k * 3 if prefer_tables else top_k
        scores, indices = self.index.search(query_embedding, search_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= score_threshold:
                chunk = self.chunks[idx]
                result = {
                    'content': chunk['content'],
                    'score': float(score),
                    'type': chunk['type'],
                    'metadata': self.chunk_metadata[idx]
                }
                results.append(result)
        
        # Filtere und priorisiere Ergebnisse
        if prefer_tables:
            table_results = [r for r in results if r['metadata']['has_table']]
            text_results = [r for r in results if not r['metadata']['has_table']]
            
            # Kombiniere: Tabellen zuerst, dann Text
            final_results = table_results[:top_k//2] + text_results[:top_k//2]
            results = final_results[:top_k]
        else:
            results = results[:top_k]
        
        return results
    
    def search_tables_specifically(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Speziell für Tabellen-Suche optimiert
        """
        return self.search(query, top_k=top_k, score_threshold=0.05, prefer_tables=True)
    
    def _save_index(self) -> None:
        """Speichert FAISS Index und Metadaten"""
        os.makedirs(self.index_path, exist_ok=True)
        
        # FAISS Index
        faiss.write_index(self.index, f"{self.index_path}/index.faiss")
        
        # Chunks und Metadata
        with open(f"{self.index_path}/chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)
        
        with open(f"{self.index_path}/metadata.pkl", 'wb') as f:
            pickle.dump(self.chunk_metadata, f)
    
    def _load_index(self) -> None:
        """Lädt gespeicherten Index"""
        if not os.path.exists(f"{self.index_path}/index.faiss"):
            raise FileNotFoundError(f"Index nicht gefunden in {self.index_path}")
        
        self.index = faiss.read_index(f"{self.index_path}/index.faiss")
        
        with open(f"{self.index_path}/chunks.pkl", 'rb') as f:
            self.chunks = pickle.load(f)
        
        with open(f"{self.index_path}/metadata.pkl", 'rb') as f:
            self.chunk_metadata = pickle.load(f)
    
    def _get_file_hash(self, filepath: str) -> str:
        """Erstellt Hash für Datei-Versionierung"""
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()[:8]


def main():
    """
    Hauptfunktion - verarbeitet Markdown-Dateien zu RAG-System
    """
    current_dir = Path.cwd()
    markdown_dir = current_dir / "Extrahierter_Text_Markdown"
    
    if not markdown_dir.exists():
        print(f"Fehler: Ordner {markdown_dir} nicht gefunden")
        return
    
    # Initialisiere verbessertes RAG System
    rag_system = EnhancedRAGEmbeddingSystem()
    
    # Verarbeite Dokumente
    rag_system.process_documents(str(markdown_dir))
    
    # Verbesserte Test-Suchen
    print("\n" + "="*60)
    print("Verbesserte Test-Suchen:")
    print("="*60)
    
    test_queries = [
        ("Umsatz 2023", True),
        ("EBIT Kennzahlen", True),
        ("Mitarbeiter Anzahl", True),
        ("Dividende pro Aktie", True),
        ("Continental Automotive", False)
    ]
    
    for query, is_table_query in test_queries:
        print(f"\nSuche: '{query}' {'(Tabellen-fokussiert)' if is_table_query else ''}")
        print("-" * 50)
        
        if is_table_query:
            results = rag_system.search_tables_specifically(query, top_k=3)
        else:
            results = rag_system.search(query, top_k=3, score_threshold=0.1)
        
        if not results:
            print("Keine Ergebnisse gefunden")
            continue
            
        for i, result in enumerate(results):
            print(f"  {i+1}. Score: {result['score']:.3f} | Type: {result['type']}")
            print(f"     Content: {result['content'][:200]}...")
            if result['metadata']['has_table']:
                tags = result['metadata'].get('semantic_tags', [])
                print(f"     -> Tabelle mit Tags: {tags}")


# Backward compatibility alias
RAGEmbeddingSystem = EnhancedRAGEmbeddingSystem

if __name__ == "__main__":
    main()