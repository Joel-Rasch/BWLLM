# RAG Streamlit App - Benutzerhandbuch

## 🚀 Schnellstart

### 1. App starten
```bash
streamlit run app.py
```

### 2. API Key konfigurieren
**Option A: .env Datei (Empfohlen)**
```bash
# Kopieren Sie die Beispiel-Datei
cp .env.example .env
# Bearbeiten Sie .env und fügen Sie Ihren API Key hinzu
GOOGLE_API_KEY=ihr_gemini_api_key
```

**Option B: Manuelle Eingabe**
- Geben Sie Ihren Google Gemini API Key in der Seitenleiste der App ein

### 3. Frage stellen
- Geben Sie Ihre Frage in das Textfeld ein
- Klicken Sie auf "🔍 Suchen"
- Warten Sie auf die Antwort mit Quellen

## 🎯 Features

### 💬 Chat-Interface
- **Natürliche Fragen**: Stellen Sie Fragen in natürlicher deutscher Sprache
- **Sofortige Antworten**: Erhalten Sie intelligente Antworten basierend auf Ihren Dokumenten
- **Quellen-Transparenz**: Sehen Sie genau, welche Dokumente für die Antwort verwendet wurden
- **Chat-Verlauf**: Alle Fragen und Antworten werden gespeichert

### 🔍 Intelligente Filter
- **Unternehmen**: Beschränken Sie die Suche auf bestimmte Unternehmen
- **Jahr**: Filtern Sie nach spezifischen Jahren
- **Anzahl Quellen**: Stellen Sie ein, wie viele Quellen angezeigt werden sollen

### 🔄 Prozess-Transparenz
Die App zeigt Ihnen transparent, was im Hintergrund passiert:

1. ⚙️ **System-Initialisierung**: RAG System wird mit Ihren Einstellungen geladen
2. 🔍 **Dokumenten-Suche**: Relevante Dokumente werden im FAISS Index gesucht  
3. 📝 **Kontext-Aufbereitung**: Gefundene Dokumente werden formatiert
4. 🤖 **Antwort-Generierung**: Gemini LLM generiert die Antwort
5. 📊 **Ergebnis-Präsentation**: Antwort und Quellen werden angezeigt

### 📊 Statistiken & Visualisierungen
- **System-Übersicht**: Gesamtzahl der Dokumente, Unternehmen, Jahre
- **Verteilungsdiagramme**: Visualisierung der Dokumentenverteilung
- **Quellen-Statistiken**: Detaillierte Analyse der verwendeten Quellen
- **Relevanz-Scores**: Bewertung der Quellenrelevanz

## 💡 Beispiel-Fragen

### Finanzielle Kennzahlen
- "Wie hoch war der Umsatz von Continental 2023?"
- "Welche Unternehmen hatten den höchsten Gewinn?"
- "Zeige mir die Entwicklung der Kosten über die Jahre"

### Tabellen-Analysen
- "Welche Tabellen gibt es über Finanzen?"
- "Erkläre mir die wichtigsten Zahlen aus den Bilanzen"
- "Was zeigen die Gewinn- und Verlustrechnungen?"

### Vergleichende Analysen
- "Vergleiche die Performance von verschiedenen Unternehmen"
- "Wie haben sich die Kennzahlen zwischen 2022 und 2023 entwickelt?"
- "Welche Trends sind in den Geschäftsberichten erkennbar?"

## 🔧 Erweiterte Funktionen

### Filter-Kombinationen
- **Einzelunternehmen**: Konzentrieren Sie sich auf ein bestimmtes Unternehmen
- **Zeiträume**: Analysieren Sie spezifische Jahre oder Zeiträume
- **Dokumenttypen**: Unterscheiden Sie zwischen Text und Tabellen-Inhalten

### Antwort-Qualität
- **Relevanz-Scores**: Jede Quelle wird mit einem Relevanz-Score bewertet
- **Kontext-Länge**: Optimale Kontextlänge für beste Antwortqualität
- **Multi-Source**: Antworten basieren auf mehreren Quellen für Vollständigkeit

### Export & Teilen
- **Verlauf**: Alle Fragen und Antworten bleiben während der Session gespeichert
- **Quellen-Details**: Vollständige Quellenangaben mit Dateinamen und Seitenzahlen
- **Copy-Paste**: Einfaches Kopieren von Antworten und Quellen

## ⚠️ Wichtige Hinweise

### System-Voraussetzungen
- FAISS Index muss existieren (`faiss_index.bin`)
- Metadata muss vorhanden sein (`chunks_metadata.pkl`)
- Führen Sie zuerst `python main.py` aus, um die Datenbank zu erstellen

### API-Nutzung
- Benötigt gültigen Google Gemini API Key
- API-Calls kosten Geld - achten Sie auf Ihr Kontingent
- Lange Dokumente können mehr Tokens verbrauchen

### Performance
- Erste Abfrage kann länger dauern (Model Loading)
- Nachfolgende Abfragen sind deutlich schneller
- Große Filter-Sets können die Suche verlangsamen

## 🐛 Troubleshooting

### Häufige Probleme

**"FAISS Index nicht gefunden"**
```bash
# Lösung: Pipeline ausführen
python main.py
```

**"API Key Fehler"**
- Überprüfen Sie Ihren Google Gemini API Key
- Stellen Sie sicher, dass das Kontingent nicht überschritten ist
- Prüfen Sie die API-Berechtigung

**"Keine Ergebnisse gefunden"**
- Versuchen Sie andere Suchbegriffe
- Entfernen Sie Filter (Unternehmen/Jahr)
- Prüfen Sie, ob relevante Dokumente in der Datenbank sind

**App lädt nicht**
```bash
# Abhängigkeiten installieren
pip install -r requirements.txt

# App neu starten
streamlit run app.py
```

### Support
- Überprüfen Sie die Logs in der App
- Aktivieren Sie "Prozess-Schritte anzeigen" für Debugging
- Konsultieren Sie die Terminal-Ausgabe für detaillierte Fehlermeldungen

## 🎨 Interface-Übersicht

### Hauptbereiche
1. **Seitenleiste**: Einstellungen, Filter, API Key
2. **Chat-Tab**: Hauptinterface für Fragen und Antworten
3. **Statistiken-Tab**: Übersicht über das System und die Daten
4. **Hilfe-Tab**: Dokumentation und Beispiele

### Navigation
- **Tabs**: Wechseln zwischen Chat, Statistiken und Hilfe
- **Expandable Sections**: Zusätzliche Details bei Bedarf
- **Filter-Updates**: Automatische Aktualisierung bei Filteränderungen
- **Verlauf**: Chronologische Darstellung aller Interaktionen