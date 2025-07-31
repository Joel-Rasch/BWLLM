# Document Processing Pipeline - University Demo

A complete RAG pipeline that processes PDF documents, extracts tables with AI enhancement, and embeds everything into a searchable vector database.

## Features

### Document Processing
- **Automatic PDF Processing**: Scans `Geschaeftsberichte/` folder for new PDFs
- **Table Extraction**: Extracts tables with surrounding context
- **AI Enhancement**: Uses Google Gemini to create natural language descriptions of tables

### Vector Database
- **Milvus Lite Integration**: Embeds documents into local file-based vector database
- **Smart Metadata**: Extracts company, year, document type from filenames
- **Chunking Strategy**: Intelligent text chunking for optimal retrieval
- **Filtered Search**: Search by company, year, or document type

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up API Key (optional)
```bash
# Create .env file
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

### 3. Process Documents (No server setup needed!)
```bash
# Step 1: Process PDFs and create enriched content
python main.py

# Step 2: Embed into vector database
python embed_to_milvus.py

# Step 3: Search the database
python search_milvus.py
```

## Workflow

1. **Document Processing** (`main.py`)
   - Place PDFs in `Geschaeftsberichte/` folder
   - Creates enriched markdown files in `Extrahierter_Text_Markdown/`

2. **Vector Embedding** (`embed_to_milvus.py`)
   - Chunks documents intelligently
   - Extracts metadata (company, year, type)
   - Embeds using SentenceTransformers
   - Stores in Milvus Lite (local file: `milvus_lite.db`)

3. **Search & Retrieval** (`search_milvus.py`)
   - Semantic search across all documents
   - Filter by company, year, or document type
   - Interactive search interface

## Example Searches

```python
# General search
searcher.search("Umsatz und Gewinn")

# Company-specific search
searcher.search("financial performance", company="Continental")

# Year-specific search
searcher.search("revenue growth", year=2023)

# Document type filter
searcher.search("tables", doc_type="enriched_tables")
```

## Files

- `main.py` - Document processing orchestrator
- `table_enricher.py` - Table extraction and AI enhancement
- `embed_to_milvus.py` - Vector database embedding
- `search_milvus.py` - Search interface with filtering
- `requirements.txt` - All dependencies

## Demo Notes

- Works without API key (basic descriptions only)
- Automatically extracts metadata from filenames
- Supports German and English content
- Interactive search with metadata filtering