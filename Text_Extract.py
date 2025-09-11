import os
import time
import concurrent.futures
from pathlib import Path
from unstructured.partition.pdf import partition_pdf
import markdownify


def extract_pdf_to_markdown_content(pdf_path):
    """
    Textextraktion aus PDF mit Tabellen als Markdown
    """
    filename = os.path.basename(pdf_path)
    all_text_elements = []

    print(f"Verarbeite: {filename}")

    pdf_elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        infer_table_structure=True,
    )

    for el in pdf_elements:

        if el.category == "Table" and getattr(el.metadata, 'text_as_html', None):
            try:
                md_table = markdownify.markdownify(el.metadata.text_as_html)
                all_text_elements.append(f"\n--- Tabelle Start ---\n{md_table}\n--- Tabelle Ende ---\n")
            except Exception:
                all_text_elements.append(f"\n--- Tabelle Start (Fallback) ---\n{el.text}\n--- Tabelle Ende ---\n")
        else:
            all_text_elements.append(el.text.strip())

    return "\n".join(all_text_elements) if all_text_elements else None


def process_single_pdf(pdf_info):
    """Verarbeitet eine einzelne PDF-Datei"""
    pdf_path, output_dir = pdf_info
    filename = os.path.basename(pdf_path)
    markdown_content = extract_pdf_to_markdown_content(pdf_path)

    if markdown_content:
        md_filename = os.path.splitext(filename)[0] + ".md"
        output_path = os.path.join(output_dir, md_filename)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        return f"{filename} zu {md_filename}"

    else:
        return f"Keine Inhalte extrahiert aus {filename}"


def main():
    current_dir = Path.cwd()
    input_dir = current_dir / "data/raw"
    output_dir = current_dir / "data/processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = [
        (str(pdf_file), str(output_dir))
        for pattern in ("*.pdf", "*.PDF")
        for pdf_file in input_dir.glob(pattern)
    ]

    if not pdf_files:
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(pdf_files))) as executor:
        results = list(executor.map(process_single_pdf, pdf_files))

    print("Ergebnisse:")
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
