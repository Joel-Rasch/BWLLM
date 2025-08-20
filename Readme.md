# RAG Chatbot für Finanzberichte

Ein Retrieval-Augmented Generation (RAG) Chatbot-System zur Analyse deutscher Finanzberichte von Unternehmen wie BMW, Continental und Daimler.

## 🚀 Features

- **Intelligente Dokumentensuche**: FAISS-basierte Vektorsuche in Finanzberichten
- **Multimodale Verarbeitung**: Automatische Konvertierung von Tabellen zu Fließtext
- **Interaktive Web-UI**: Streamlit-basierte Chatbot-Oberfläche
- **Firmenspezifische Abfragen**: Suche nach bestimmten Unternehmen und Jahren
- **Kontextanzeige**: Transparente Darstellung der verwendeten Quellen

## 📋 Voraussetzungen

- Python 3.10
- Google API Key für Gemini (für LLM-Funktionen)

## 🛠️ Installation

### 1. Repository klonen
```bash
git clone <repository-url>
cd rag-chatbot
```

### 2. Virtual Environment erstellen
```bash
python -m venv BWLLM_venv
source BWLLM_venv/bin/activate  # Linux/Mac
# oder
BWLLM_venv\Scripts\activate     # Windows
```

### 3. Dependencies installieren
```bash
pip install -r requirements.txt
```

### 4. Erforderliche Python-Pakete
Falls keine `requirements.txt` vorhanden, installiere folgende Pakete:
```bash
pip install streamlit
pip install langchain
pip install langchain-community
pip install langchain-google-genai
pip install faiss-cpu  # oder faiss-gpu für GPU-Support
pip install sentence-transformers
pip install python-dotenv
pip install pathlib
```

### 5. Umgebungsvariablen konfigurieren
Erstelle eine `.env` Datei im Projektroot:
```env
GOOGLE_API_KEY=your_google_api_key_here
```
## 📁 Projektstruktur

```
├── app.py                      # Hauptanwendung (Streamlit)
├── rag_system.py              # RAG-Pipeline
├── variable_loader.py         # Globale Variablen-Loader
├── create_faiss.ipynb         # FAISS-Index erstellen
├── process_markdown.ipynb     # Datenverarbeitung
├── test_rag.ipynb            # System-Tests
├── .env                       # Umgebungsvariablen
└── data/
    ├── raw/                   # Rohe Markdown-Dateien
    ├── text/                  # Extrahierte Texte
    ├── table/                 # Extrahierte Tabellen
    ├── tabletext/            # KI-generierte Tabellenbeschreibungen
    └── processed/            # Verarbeitete finale Dateien
```

## 🔧 Setup-Prozess

### Schritt 1: Datenverarbeitung

1. **Rohdaten vorbereiten:**
   - Platziere Markdown-Dateien im Format `{firma}_{jahr}.md` im `data/raw/` Ordner
   - Beispiel: `bmw_2023.md`, `continental_2022.md`

2. **Datenverarbeitung ausführen:**
   ```bash
   jupyter notebook process_markdown.ipynb
   ```
   - Führe alle Zellen nacheinander aus
   - Dieser Schritt extrahiert Texte und Tabellen
   - Konvertiert Tabellen zu Fließtext mit Gemini AI

### Schritt 2: FAISS-Index erstellen

```bash
jupyter notebook create_faiss.ipynb
```
- Führe alle Zellen aus
- Erstellt einen FAISS-Vektorindex aus allen verarbeiteten Dokumenten
- Speichert den Index als `faiss_index/`

### Schritt 3: System testen

```bash
jupyter notebook test_rag.ipynb
```
- Teste das RAG-System mit Beispielabfragen
- Überprüfe ob alle Komponenten korrekt funktionieren

### Schritt 4: Chatbot starten

```bash
streamlit run app.py
```

Die Anwendung ist dann unter `http://localhost:8501` verfügbar.

## 💡 Verwendung

### Chatbot-Interface

1. **Firmenspezifische Fragen stellen:**
   - "Wie viel Umsatz hat BMW 2023 gemacht?"
   - "Was war der Gewinn von Continental 2022?"
   - "Hat BMW 2022 mehr Umsatz als Continental gemacht?"

2. **Kontext einsehen:**
   - Klicke auf "🔍 Kontext Anschauen" um die verwendeten Quellen zu sehen
   - Zeigt Document-IDs, Metadaten und Inhalte

### Testseite

- Nutze die Testseite zur Überprüfung der Suchfunktionalität
- Zeigt direkte FAISS-Suchergebnisse ohne LLM-Verarbeitung

## ⚙️ Konfiguration

### Anpassbare Parameter

**In `create_faiss.ipynb`:**
- `chunk_size`: Größe der Textchunks (Standard: 200)
- `chunk_overlap`: Überlappung zwischen Chunks (Standard: 0)
- `model_name`: Embedding-Modell (Standard: "sentence-transformers/all-MiniLM-L6-v2")

**In `rag_system.py`:**
- `k`: Anzahl der abgerufenen Dokumente (Standard: 5)
- `temperature`: LLM-Kreativität (Standard: 1)

### Neue Firmen hinzufügen

1. Füge Markdown-Dateien im Format `{firma}_{jahr}.md` zu `data/raw/` hinzu
2. Führe `process_markdown.ipynb` aus
3. Erstelle den FAISS-Index neu mit `create_faiss.ipynb`
4. Starte die Anwendung neu

## 📊 Datenformat

### Erwartetes Markdown-Format

```markdown
# Firmenbericht 2023

Normaler Text...

--- Tabelle Start ---
| Kennzahl | Wert | Vorjahr |
|----------|------|---------|
| Umsatz   | 100€ | 95€     |
--- Tabelle Ende ---

Weiterer Text...
```

### Metadaten-Struktur

Jedes Dokument enthält:
- `company`: Firmenname (extrahiert aus Dateiname)
- `year`: Jahr (extrahiert aus Dateiname)
- `chunk`: Chunk-Nummer innerhalb des Dokuments

---
