from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

with open("Extrahierter_Text_Markdown/Daimler_Truck_Holding_2023.md", "r", encoding="utf-8") as f:
    text = f.read()

def simple_financial_chunker(text, chunk_size=1000, chunk_overlap=80):
    table_pattern = r'--- Tabelle Start ---(.*?)--- Tabelle Ende ---'
    tables = []
    
    for match in re.finditer(table_pattern, text, re.DOTALL):
        tables.append(match.group(1).strip())
    
    text_without_tables = re.sub(table_pattern, '<<TABLE_PLACEHOLDER>>', text, flags=re.DOTALL)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n\n",
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )
    
    text_chunks = text_splitter.split_text(text_without_tables)
    
    table_chunks = []
    for i, table_content in enumerate(tables):
        cleaned_table = clean_table_simple(table_content)
        table_chunks.append(f"📊 **TABELLE {i+1}**\n\n{cleaned_table}")
    
    return text_chunks + table_chunks

def clean_table_simple(table_text):
    lines = [line.strip() for line in table_text.split('\n') if line.strip()]
    
    cleaned_lines = []
    for line in lines:
        if '|' in line and not line.startswith('---'):
            cells = [cell.strip() for cell in line.split('|')]
            if len(cells) > 2:
                cells = [cell if cell else "—" for cell in cells]
                cleaned_lines.append('| ' + ' | '.join(cells[1:-1]) + ' |')
    
    return '\n'.join(cleaned_lines)

chunks = simple_financial_chunker(text, chunk_size=1000, chunk_overlap=80)

print(f"✅ {len(chunks)} Chunks erstellt")

table_chunks = [chunk for chunk in chunks if chunk.startswith('📊')]
text_chunks = [chunk for chunk in chunks if not chunk.startswith('📊')]

print(f"📄 Text-Chunks: {len(text_chunks)}")
print(f"📊 Tabellen-Chunks: {len(table_chunks)}")

if table_chunks:
    print("\n--- BEISPIEL TABELLEN-CHUNK ---")
    print(table_chunks[0][:400] + "...")

if text_chunks:
    print("\n--- BEISPIEL TEXT-CHUNK ---")  
    print(text_chunks[0][:300] + "...")