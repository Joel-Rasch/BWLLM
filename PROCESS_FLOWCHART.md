# 🔄 RAG Document Processing Flowchart

## 📊 Complete Pipeline Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   📁 PDF Files  │    │  🔍 Extraction  │    │  ✂️ Chunking    │    │ 🤖 Enrichment  │    │  🧮 Embeddings  │
│                 │    │                 │    │                 │    │                 │    │                 │
│ • Company_Year  │───▶│ • Text Content  │───▶│ • Text Chunks   │───▶│ • AI Analysis   │───▶│ • Vector Repr.  │
│ • .pdf format   │    │ • Table Data    │    │ • Table Chunks  │    │ • Descriptions  │    │ • 384 dimensions│
│ • Business Rpts │    │ • Metadata      │    │ • Metadata      │    │ • Context       │    │ • Normalized    │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
                                                                                                       │
                                                                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  💬 User Query  │    │  🎯 Smart       │    │  📊 Similarity  │    │  🤖 LLM         │    │  💾 FAISS       │
│                 │    │  Filtering      │    │  Search         │    │  Generation     │    │  Storage        │
│ • Natural Lang  │◀───│ • Auto Company  │◀───│ • Vector Match  │◀───│ • Context Aware │◀───│ • Index File    │
│ • Multi-domain  │    │ • Auto Year     │    │ • Top-K Results │    │ • Source Cited  │    │ • Metadata      │
│ • German Text   │    │ • Transparent   │    │ • Filtered      │    │ • German Output │    │ • Fast Retrieval│
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔍 Detailed Processing Steps

### Step 1: PDF Content Extraction
```
📄 Continental_2023.pdf
           │
           ▼
   ┌───────────────┐
   │  unstructured │ ──┐
   │    library    │   │
   └───────────────┘   │
           │           │
           ▼           │
   ┌───────────────┐   │ ┌─────────────┐
   │     Tables    │───┤ │    Text     │
   │   • Financial │   │ │  • Paragraphs│
   │   • Statistics│   │ │  • Headers   │
   │   • Data grids│   │ │  • Summaries │
   └───────────────┘   │ └─────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Raw Content    │
              │  with Structure │
              └─────────────────┘
```

### Step 2: Content Chunking Strategy
```
┌─────────────────────────────────────┐
│           Raw Content               │
│  • Long documents                   │
│  • Mixed content types              │
│  • Tables and text                  │
└─────────────────┬───────────────────┘
                  │
                  ▼
         ┌─────────────────┐
         │  Text Splitter  │
         │  • 1000 chars   │
         │  • 200 overlap  │
         │  • Semantic     │
         └─────────┬───────┘
                   │
                   ▼
    ┌──────────────┴──────────────┐
    │                             │
    ▼                             ▼
┌───────────┐               ┌───────────┐
│Text Chunks│               │Table Chunks│
│• Coherent │               │• Structured│
│• Metadata │               │• Enhanced  │
│• Context  │               │• Tabular   │
└───────────┘               └───────────┘
```

### Step 3: AI Enrichment Process
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Table Chunk   │    │  Gemini API     │    │  Enhanced Chunk │
│                 │    │                 │    │                 │
│ Raw table data  │───▶│ • Context anal. │───▶│ + AI description│
│ Numbers & text  │    │ • Explanation   │    │ + Human-readable│
│ Basic metadata  │    │ • Summarization │    │ + Better context│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                   ▲
                                   │
                            ┌─────────────┐
                            │ API Key     │
                            │ Required    │
                            └─────────────┘
```

### Step 4: Vector Embedding Generation
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Text Chunks    │    │ Sentence Trans- │    │  Vector Embed.  │
│                 │    │ former Model    │    │                 │
│ "Umsatz stieg   │───▶│                 │───▶│ [0.1, -0.3, .. │
│  um 15% auf     │    │ all-MiniLM-L6v2 │    │  0.8, 0.2, ..] │
│  €2.5 Mrd."     │    │ (384 dims)      │    │ 384 dimensions  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Step 5: FAISS Index Construction
```
┌─────────────────┐    ┌─────────────────┐
│   Embeddings    │    │  FAISS Index    │
│                 │    │                 │
│ Vector Arrays   │───▶│ • Fast Search   │
│ [chunk1_vec]    │    │ • Cosine Sim.   │
│ [chunk2_vec]    │    │ • IndexFlatIP   │
│ [chunk3_vec]    │    │ • Normalized    │
│ ...             │    │                 │
└─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Metadata PKL   │
                       │                 │
                       │ • Company info  │
                       │ • Year info     │
                       │ • Content text  │
                       │ • Chunk types   │
                       └─────────────────┘
```

## 🎯 Query Processing with Smart Filtering

### Query Analysis Flow
```
👤 "Wie hoch war der Umsatz von Continental 2023?"
                    │
                    ▼
            ┌───────────────┐
            │ Query Analyzer│
            │               │
            │ • NLP parsing │
            │ • Entity extr.│
            │ • Pattern match│
            └───────┬───────┘
                    │
                    ▼
     ┌──────────────┴──────────────┐
     │                             │
     ▼                             ▼
┌─────────────┐            ┌─────────────┐
│ Company:    │            │ Year:       │
│ Continental │            │ 2023        │
└─────────────┘            └─────────────┘
```

### Smart Filtering Application
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Full Corpus    │    │  Smart Filter   │    │ Filtered Corpus │
│                 │    │                 │    │                 │
│ • All companies │───▶│ Company =       │───▶│ Only Continental│
│ • All years     │    │ "Continental"   │    │ Only 2023       │
│ • 10,000 chunks │    │ Year = "2023"   │    │ 500 chunks     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │ Similarity      │
                                              │ Search          │
                                              │ • More precise  │
                                              │ • Faster        │
                                              │ • Relevant      │
                                              └─────────────────┘
```

## 📊 Data Flow Architecture

```
                               🎯 Smart RAG System Architecture
                                          
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   Input     │  │ Processing  │  │   Storage   │  │  Retrieval  │  │   Output    │
│   Layer     │  │   Layer     │  │    Layer    │  │    Layer    │  │   Layer     │
│             │  │             │  │             │  │             │  │             │
│ • PDF Files │─▶│ • Extract   │─▶│ • FAISS     │─▶│ • Query     │─▶│ • Answers   │
│ • Business  │  │ • Chunk     │  │ • Metadata  │  │ • Filter    │  │ • Sources   │
│   Reports   │  │ • Enrich    │  │ • Vectors   │  │ • Search    │  │ • Context   │
│ • Multi-co. │  │ • Embed     │  │ • Persist   │  │ • Generate  │  │ • Citations │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
      │                │                │                │                │
      ▼                ▼                ▼                ▼                ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ File System │  │ Python      │  │ Binary      │  │ AI Models   │  │ Streamlit   │
│ Structure   │  │ Processing  │  │ Storage     │  │ & Logic     │  │ Interface   │
│             │  │             │  │             │  │             │  │             │
│ Named PDFs  │  │ Unstructured│  │ .bin/.pkl   │  │ Transformer │  │ Web UI      │
│ Organized   │  │ LangChain   │  │ Fast Access │  │ Gemini LLM  │  │ Interactive │
│ Accessible  │  │ Custom Code │  │ Scalable    │  │ Query Anal. │  │ User-Friend │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
```

## 🔧 Component Integration

```
main.py (Orchestrator)
         │
         ├─── src/processing/
         │    ├─── table_extractor.py    📄 PDF → Content
         │    ├─── chunk_processor.py    ✂️ Content → Chunks  
         │    └─── chunk_enricher.py     🤖 Chunks → Enhanced
         │
         ├─── src/vector/
         │    ├─── embedder.py           🧮 Text → Vectors
         │    └─── search_faiss.py       🔍 Vector Operations
         │
         └─── src/rag/
              └─── retriever.py          🎯 Query → Answer
                   ├─── QueryAnalyzer   🧠 NLP Analysis
                   ├─── FAISSRetriever  📊 Vector Search
                   └─── RAGSystem       🤖 Complete Pipeline

app.py (User Interface)
         │
         ├─── Streamlit Components
         ├─── Interactive Chat
         ├─── Statistics Dashboard  
         └─── Process Transparency
```