from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from pathlib import Path
import re
import os

def extract_company_name(filename):
    """Extrahiert den Unternehmensnammen aus dem Dateinamen"""
    name = filename.replace('.md', '')
    
    # Spezielle Mappings für bessere Erkennung
    company_mappings = {
        'knorrbremse': 'Knorr-Bremse',
        'knorr_bremse': 'Knorr-Bremse', 
        'daimler_truck': 'Daimler Truck',
        'porsche_automobil': 'Porsche SE',
        'continental': 'Continental',
        'bmw': 'BMW',
        'vw': 'Volkswagen',
        'traton': 'TRATON'
    }
    
    name_lower = name.lower().replace('_', '_').replace(' ', '_')
    
    # Prüfe Mappings
    for key, mapped_name in company_mappings.items():
        if key in name_lower:
            return mapped_name
    
    # Fallback
    words = name.replace('_', ' ').replace('-', ' ').split()
    return ' '.join(word.capitalize() for word in words)

def enhanced_financial_chunker(text, company_name, chunk_size=1000, chunk_overlap=150):
    """
    Verbesserter Chunker speziell für Finanzberichte mit besserer Tabellen-Behandlung
    """
    
    # 1. Tabellen mit mehr Kontext extrahieren 
    table_pattern = r'--- Tabelle Start ---(.*?)--- Tabelle Ende ---'
    tables = []
    table_contexts = []
    
    for match in re.finditer(table_pattern, text, re.DOTALL):
        table_content = match.group(1).strip()
        
        # Suche nach Kontext vor der Tabelle (vorherige 500 Zeichen)
        start_pos = match.start()
        context_start = max(0, start_pos - 500)
        pre_context = text[context_start:start_pos].strip()
        
        # Suche nach Kontext nach der Tabelle (nächste 300 Zeichen)
        end_pos = match.end()
        post_context = text[end_pos:end_pos + 300].strip()
        
        tables.append(table_content)
        table_contexts.append({
            'pre_context': pre_context[-200:] if len(pre_context) > 200 else pre_context,
            'post_context': post_context[:200] if len(post_context) > 200 else post_context
        })
    
    # Text ohne Tabellen
    text_without_tables = re.sub(table_pattern, '<<TABLE_PLACEHOLDER>>', text, flags=re.DOTALL)
    
    # 2. Text-Chunking mit finanziellen Trennwörtern
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n\n",
            "\n\n", 
            "\n---",  # Oft vor wichtigen Abschnitten
            "\n##",   # Überschriften Level 2
            "\n#",    # Überschriften Level 1
            "\n",     
            ". ",     
            "; ",     
            " ",      
            ""        
        ]
    )
    
    raw_chunks = text_splitter.split_text(text_without_tables)
    
    # 3. Text-Chunks mit Finanz-Keywords anreichern
    text_chunks = []
    for chunk in raw_chunks:
        if len(chunk.strip()) >= 400:  # Mindestlänge reduziert
            
            # Erkenne wichtige Finanz-Keywords
            keywords = extract_financial_keywords(chunk)
            
            # Erstelle erweiterten Chunk
            enhanced_chunk = create_enhanced_text_chunk(chunk, company_name, keywords)
            text_chunks.append(enhanced_chunk)
    
    # 4. Tabellen mit vollem Kontext verarbeiten
    table_chunks = []
    for i, (table_content, context) in enumerate(zip(tables, table_contexts)):
        
        enhanced_table = create_enhanced_table_chunk(
            table_content, company_name, i+1, context
        )
        table_chunks.append(enhanced_table)
    
    return text_chunks, table_chunks

def extract_financial_keywords(text):
    """Extrahiert wichtige Finanz-Keywords aus dem Text"""
    
    # Listen wichtiger Finanz-Begriffe
    revenue_terms = ['umsatz', 'erlös', 'einnahmen', 'revenue', 'sales']
    profit_terms = ['gewinn', 'verlust', 'ebitda', 'ebit', 'ergebnis', 'profit']
    financial_terms = ['bilanz', 'cashflow', 'liquidität', 'dividende', 'aktie']
    
    text_lower = text.lower()
    found_keywords = []
    
    # Suche nach Jahren (2019-2025)
    years = re.findall(r'\b(20[12][0-9])\b', text)
    found_keywords.extend(years)
    
    # Suche nach Währungsbeträgen
    amounts = re.findall(r'\b\d+[.,]?\d*\s*(?:mio|milliarden?|tsd|€|euro|usd|dollar)\b', text_lower)
    found_keywords.extend(amounts[:3])  # Max 3 Beträge
    
    # Suche nach Finanz-Keywords
    for term_group in [revenue_terms, profit_terms, financial_terms]:
        for term in term_group:
            if term in text_lower:
                found_keywords.append(term)
    
    return list(set(found_keywords))  # Duplikate entfernen

def create_enhanced_text_chunk(chunk, company_name, keywords):
    """Erstellt einen angereicherten Text-Chunk"""
    
    # Basis-Header
    header = f"Unternehmen: {company_name}\n"
    header += f"Dokumenttyp: Geschäftsbericht\n"
    
    # Keywords hinzufügen falls vorhanden
    if keywords:
        header += f"Schlüsselwörter: {', '.join(keywords[:5])}\n"  # Max 5 Keywords
    
    header += "\nInhalt:\n"
    
    return header + chunk.strip()

def create_enhanced_table_chunk(table_content, company_name, table_num, context):
    """Erstellt einen angereicherten Tabellen-Chunk mit Kontext"""
    
    # Bereinige Tabelle
    cleaned_table = clean_table_advanced(table_content)
    
    # Erkenne Tabellen-Typ basierend auf Inhalt
    table_type = identify_table_type(cleaned_table)
    
    # Header mit mehr Informationen
    header = f"Unternehmen: {company_name}\n"
    header += f"Dokumenttyp: Finanztabelle ({table_type})\n"
    header += f"Tabellen-ID: {table_num}\n"
    
    # Füge Kontext hinzu
    full_content = header + "\n"
    
    if context['pre_context']:
        full_content += f"Kontext davor: ...{context['pre_context']}...\n\n"
    
    full_content += f"TABELLENDATEN:\n{cleaned_table}\n"
    
    if context['post_context']:
        full_content += f"\nKontext danach: ...{context['post_context']}..."
    
    return full_content

def identify_table_type(table_content):
    """Identifiziert den Typ der Tabelle basierend auf Inhalt"""
    
    content_lower = table_content.lower()
    
    if any(term in content_lower for term in ['umsatz', 'erlös', 'revenue']):
        return "Umsatztabelle"
    elif any(term in content_lower for term in ['gewinn', 'verlust', 'ebitda', 'ebit']):
        return "Ergebnistabelle" 
    elif any(term in content_lower for term in ['bilanz', 'aktiva', 'passiva']):
        return "Bilanztabelle"
    elif any(term in content_lower for term in ['cashflow', 'geldfluss']):
        return "Cashflow-Tabelle"
    elif any(term in content_lower for term in ['mitarbeiter', 'personal']):
        return "Mitarbeitertabelle"
    else:
        return "Finanzkennzahlen"

def clean_table_advanced(table_text):
    """Erweiterte Tabellen-Bereinigung mit besserer Formatierung"""
    
    lines = [line.strip() for line in table_text.split('\n') if line.strip()]
    cleaned_lines = []
    
    for line in lines:
        if '|' in line and not line.startswith('---'):
            cells = [cell.strip() for cell in line.split('|')]
            if len(cells) > 2:
                # Bessere Zell-Bereinigung
                clean_cells = []
                for cell in cells[1:-1]:  # Entferne erste/letzte leere Zelle
                    if not cell:
                        clean_cells.append("—")
                    else:
                        # Entferne Zeilenumbrüche in Zellen
                        cell_clean = cell.replace('\n', ' ').strip()
                        clean_cells.append(cell_clean)
                
                if clean_cells:  # Nur wenn Zellen vorhanden
                    cleaned_lines.append('| ' + ' | '.join(clean_cells) + ' |')
    
    return '\n'.join(cleaned_lines)

def create_faiss_index_from_markdown():
    """
    Hauptfunktion - Erstellt FAISS-Index aus Markdown-Dateien
    """
    current_dir = Path.cwd()
    markdown_dir = current_dir / "Extrahierter_Text_Markdown"
    
    if not markdown_dir.exists():
        print(f"[FEHLER] Ordner fehlt: {markdown_dir}")
        return
    
    markdown_files = list(markdown_dir.glob("*.md"))
    if not markdown_files:
        print("Keine Markdown-Dateien gefunden!")
        return
    
    print(f"📄 Gefunden: {len(markdown_files)} Markdown-Dateien")
    print(f"⚙️ Erweiterte Chunk-Parameter: 1000 Zeichen, 150 Overlap, mit Kontext")
    
    all_documents = []
    stats = {'total': 0, 'text': 0, 'tables': 0}
    
    for md_file in markdown_files:
        print(f"→ Verarbeite: {md_file.name}")
        
        company_name = extract_company_name(md_file.name)
        print(f"  📈 Unternehmen: {company_name}")
        
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                text = f.read()
            
            # Erweiterte Chunk-Erstellung
            text_chunks, table_chunks = enhanced_financial_chunker(
                text, company_name, chunk_size=1000, chunk_overlap=150
            )
            
            # Text-Dokumente hinzufügen
            for i, chunk in enumerate(text_chunks):
                all_documents.append(Document(
                    page_content=chunk,
                    metadata={
                        "source": md_file.name,
                        "company": company_name,
                        "chunk_type": "text",
                        "chunk_id": i,
                        "char_count": len(chunk)
                    }
                ))
                stats['text'] += 1
            
            # Tabellen-Dokumente hinzufügen  
            for i, chunk in enumerate(table_chunks):
                all_documents.append(Document(
                    page_content=chunk,
                    metadata={
                        "source": md_file.name,
                        "company": company_name,
                        "chunk_type": "table", 
                        "chunk_id": i,
                        "char_count": len(chunk)
                    }
                ))
                stats['tables'] += 1
            
            print(f"  ✓ {len(text_chunks)} Text + {len(table_chunks)} Tabellen")
            
        except Exception as e:
            print(f"  ✗ Fehler: {e}")
    
    stats['total'] = len(all_documents)
    
    print(f"\n📊 Finale Statistiken:")
    print(f"   Gesamt: {stats['total']} Chunks")
    print(f"   Text: {stats['text']} Chunks") 
    print(f"   Tabellen: {stats['tables']} Chunks")
    
    if not all_documents:
        print("❌ Keine Chunks erstellt!")
        return
    
    # Deutsches Embedding-Modell
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    print(f"\n🔄 Erstelle FAISS-Index mit {model_name}...")
    
    # FAISS-Index erstellen und speichern
    faiss_index = FAISS.from_documents(all_documents, embedding=embeddings)
    faiss_index.save_local("faiss_index")
    
    print(f"✅ FAISS-Index gespeichert!")
    
    # Übersicht
    companies = set(d.metadata['company'] for d in all_documents)
    print(f"\n🏢 Erfasste Unternehmen ({len(companies)}):")
    for company in sorted(companies):
        company_chunks = len([d for d in all_documents if d.metadata['company'] == company])
        text_chunks = len([d for d in all_documents if d.metadata['company'] == company and d.metadata['chunk_type'] == 'text'])
        table_chunks = len([d for d in all_documents if d.metadata['company'] == company and d.metadata['chunk_type'] == 'table'])
        print(f"   • {company}: {company_chunks} Chunks ({text_chunks} Text, {table_chunks} Tabellen)")

if __name__ == "__main__":
    create_faiss_index_from_markdown()