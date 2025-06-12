#!/usr/bin/env python3
"""
Setup-Skript für das RAG-System
Führt alle notwendigen Vorbereitungen durch, damit app.py direkt gestartet werden kann.
"""

import os
import sys
from pathlib import Path
from embed_kontext import create_faiss_index_from_markdown
import subprocess

def check_requirements():
    """Prüft ob alle Requirements installiert sind"""
    print("🔍 Prüfe Requirements...")
    
    requirements_file = Path("requirements.txt")
    if requirements_file.exists():
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                         check=True, capture_output=True)
            print("✅ Requirements installiert")
        except subprocess.CalledProcessError as e:
            print(f"❌ Fehler bei Requirements-Installation: {e}")
            return False
    else:
        print("⚠️ requirements.txt nicht gefunden - installiere Basis-Pakete...")
        packages = [
            "streamlit",
            "langchain",
            "langchain-google-genai",
            "faiss-cpu",
            "sentence-transformers",
            "python-dotenv",
            "unstructured[pdf]",
            "markdownify"
        ]
        
        for package in packages:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", package], 
                             check=True, capture_output=True)
                print(f"  ✅ {package}")
            except subprocess.CalledProcessError:
                print(f"  ❌ {package}")
                return False
    
    return True

def check_env_file():
    """Prüft ob .env Datei existiert"""
    print("\n🔍 Prüfe .env Datei...")
    
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️ .env Datei nicht gefunden!")
        print("Erstelle .env Template...")
        
        with open(".env", "w") as f:
            f.write("# Google AI API Key für Gemini\n")
            f.write("GOOGLE_API_KEY=your_google_api_key_here\n")
        
        print("✅ .env Template erstellt")
        print("❗ WICHTIG: Trage deinen Google AI API Key in die .env Datei ein!")
        return False
    else:
        # Prüfe ob API Key gesetzt ist
        with open(".env", "r") as f:
            content = f.read()
            if "your_google_api_key_here" in content or "GOOGLE_API_KEY=" not in content:
                print("⚠️ Google API Key nicht konfiguriert!")
                print("❗ Trage deinen Google AI API Key in die .env Datei ein!")
                return False
        
        print("✅ .env Datei gefunden und konfiguriert")
        return True

def extract_pdfs():
    """Extrahiert PDFs zu Markdown falls noch nicht geschehen"""
    print("\n🔍 Prüfe PDF-Extraktion...")
    
    pdf_dir = Path("Geschaeftsberichte")
    markdown_dir = Path("Extrahierter_Text_Markdown")
    
    if not pdf_dir.exists():
        print("⚠️ Geschaeftsberichte Ordner nicht gefunden!")
        print("📁 Erstelle Ordner - bitte PDFs dort hineinlegen")
        pdf_dir.mkdir(exist_ok=True)
        return False
    
    # Prüfe ob PDFs vorhanden sind
    pdf_files = list(pdf_dir.glob("*.pdf")) + list(pdf_dir.glob("*.PDF"))
    if not pdf_files:
        print("⚠️ Keine PDF-Dateien im Geschaeftsberichte Ordner gefunden!")
        return False
    
    # Prüfe ob Markdown-Dateien bereits existieren
    if markdown_dir.exists():
        md_files = list(markdown_dir.glob("*.md"))
        if len(md_files) >= len(pdf_files):
            print("✅ Markdown-Dateien bereits vorhanden")
            return True
    
    print(f"🔄 Extrahiere {len(pdf_files)} PDF-Dateien...")
    
    try:
        from Textextraktion import main as extract_main
        extract_main()
        print("✅ PDF-Extraktion abgeschlossen")
        return True
    except Exception as e:
        print(f"❌ Fehler bei PDF-Extraktion: {e}")
        return False

def create_embeddings():
    """Erstellt FAISS-Embeddings"""
    print("\n🔍 Prüfe FAISS-Index...")
    
    index_path = Path("faiss_index")
    if index_path.exists() and (index_path / "index.faiss").exists():
        print("✅ FAISS-Index bereits vorhanden")
        return True
    
    markdown_dir = Path("Extrahierter_Text_Markdown")
    if not markdown_dir.exists() or not list(markdown_dir.glob("*.md")):
        print("❌ Keine Markdown-Dateien für Embedding-Erstellung gefunden!")
        return False
    
    print("🔄 Erstelle FAISS-Index...")
    
    try:
        from embed_kontext import create_faiss_index_from_markdown
        create_faiss_index_from_markdown()
        print("✅ FAISS-Index erstellt")
        return True
    except Exception as e:
        print(f"❌ Fehler bei FAISS-Index-Erstellung: {e}")
        return False

def main():
    """Hauptfunktion des Setup-Skripts"""
    print("🚀 RAG-System Setup")
    print("=" * 50)
    
    success = True
    
    # 1. Requirements prüfen
    if not check_requirements():
        success = False
    
    # 2. .env Datei prüfen
    env_ok = check_env_file()
    if not env_ok:
        success = False
    
    # 3. PDFs extrahieren
    if not extract_pdfs():
        success = False
    
    # 4. Embeddings erstellen
    if success and env_ok:  # Nur wenn alles andere OK ist
        if not create_embeddings():
            success = False
    
    print("\n" + "=" * 50)
    
    if success:
        print("✅ Setup erfolgreich abgeschlossen!")
        print("\n🎉 Du kannst jetzt die App starten mit:")
        print("   streamlit run app.py")
    else:
        print("❌ Setup unvollständig!")
        print("\n🔧 Bitte behebe die oben genannten Probleme und führe setup.py erneut aus.")
    
    print("\n📁 Ordnerstruktur:")
    print("   ./Geschaeftsberichte/     - Hier PDFs hineinlegen")
    print("   ./Extrahierter_Text_Markdown/  - Extrahierte Markdown-Dateien")
    print("   ./faiss_index/           - FAISS-Embeddings")
    print("   ./.env                   - API Keys")

if __name__ == "__main__":
    main()