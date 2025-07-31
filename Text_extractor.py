import os
import time
import concurrent.futures
from pathlib import Path
from unstructured.partition.pdf import partition_pdf
import markdownify


def extract_pdf_to_markdown_content(pdf_path):
    """
    Schnelle und robuste Textextraktion aus PDF mit Tabellen als Markdown
    """
    if not os.path.exists(pdf_path):
        return None

    filename = os.path.basename(pdf_path)
    all_text_elements = []

    try:
        print(f"-> Verarbeite: {filename}")

        # pdf wird analysiert
        pdf_elements = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",  # schneller als hi_res
            infer_table_structure=True,
            languages=['deu']
        )

        for el in pdf_elements:
            # Tabelle -> als HTML -> in Markdown umwandeln
            if el.category == "Table" and getattr(el.metadata, 'text_as_html', None):
                try:
                    md_table = markdownify.markdownify(el.metadata.text_as_html)
                    all_text_elements.append(f"\n--- Tabelle Start ---\n{md_table}\n--- Tabelle Ende ---\n")
                except Exception:
                    all_text_elements.append(f"\n--- Tabelle Start (Fallback) ---\n{el.text}\n--- Tabelle Ende ---\n")
            else:
                all_text_elements.append(el.text.strip())

        return "\n".join(all_text_elements) if all_text_elements else None

    except Exception as e:
        print(f"[FEHLER bei {filename}]: {e}")
        return None


def process_single_pdf(pdf_info):
    """Verarbeitet eine einzelne PDF-Datei"""
    pdf_path, output_dir = pdf_info
    filename = os.path.basename(pdf_path)
    markdown_content = extract_pdf_to_markdown_content(pdf_path)

    if markdown_content:
        md_filename = os.path.splitext(filename)[0] + ".md"
        output_path = os.path.join(output_dir, md_filename)

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            return f"[OK] {filename} -> {md_filename}"
        except Exception as e:
            return f"[FEHLER] Fehler beim Speichern {filename}: {e}"
    else:
        return f"[FEHLER] Keine Inhalte extrahiert aus {filename}"


def main():
    """Hauptfunktion – keine Hardcodierung, läuft im aktuellen Ordner"""
    current_dir = Path.cwd()
    input_dir = current_dir / "Geschaeftsberichte"
    output_dir = current_dir / "Extrahierter_Text_Markdown"

    if not input_dir.exists():
        print(f"[FEHLER] Ordner fehlt: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # PDFs sammeln
    pdf_files = [
        (str(pdf_file), str(output_dir))
        for pattern in ("*.pdf", "*.PDF")
        for pdf_file in input_dir.glob(pattern)
    ]

    print(f"\nPDF-Dateien gefunden: {len(pdf_files)}")

    if not pdf_files:
        return

    print(f"Starte Verarbeitung mit max. 4 Threads...\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(pdf_files))) as executor:
        results = list(executor.map(process_single_pdf, pdf_files))

    print("\nErgebnisse:\n" + "-" * 40)
    for result in results:
        print(result)

    print(f"\nFertig. Markdown-Dateien in: {output_dir}")


if __name__ == "__main__":
    main()