import os
import json
import logging
import concurrent.futures
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Union, Dict
from enum import Enum
import dotenv

import google.generativeai as genai
import markdownify
from unstructured.partition.pdf import partition_pdf


class ChunkType(Enum):
    TEXT = "text"
    TABLE = "table"


@dataclass
class BaseChunk:
    content: str
    page_number: Optional[int]
    element_index: int
    chunk_type: ChunkType


@dataclass
class TextChunk(BaseChunk):
    chunk_type: ChunkType = ChunkType.TEXT


@dataclass
class TableChunk(BaseChunk):
    raw_html: Optional[str] = None
    chunk_type: ChunkType = ChunkType.TABLE


@dataclass
class TableGroup:
    tables: List[TableChunk]
    previous_text: Optional[TextChunk]
    next_text: Optional[TextChunk]
    start_index: int
    end_index: int


@dataclass
class EnrichedTableGroup:
    table_group: TableGroup
    natural_description: str
    enriched_content: str


@dataclass
class ProcessingResult:
    pdf_filename: str
    success: bool
    error_message: Optional[str] = None
    table_groups_count: int = 0
    total_tables_count: int = 0
    output_path: Optional[str] = None


class Config:
    CONTEXT_WINDOW = 3
    MAX_PAGE_CONTENT_LENGTH = 1000
    GEMINI_MODEL = 'gemini-1.5-flash'
    DEFAULT_OUTPUT_ENCODING = 'utf-8'
    MAX_TABLES_PER_GROUP = 5
    INPUT_DIR = "Geschaeftsberichte"
    OUTPUT_DIR = "Extrahierter_Text_Markdown"
    PROCESSING_LOG_FILE = "table_processing_log.json"
    MAX_CONCURRENT_WORKERS = 2


class ProcessingTracker:
    def __init__(self, log_file_path: str):
        self.log_file_path = Path(log_file_path)
        self.processed_files = self._load_processing_log()
        self.logger = logging.getLogger(__name__)
    
    def _load_processing_log(self) -> Dict[str, dict]:
        if self.log_file_path.exists():
            try:
                with open(self.log_file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_processing_log(self):
        try:
            with open(self.log_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.processed_files, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save processing log: {e}")
    
    def is_file_processed(self, pdf_path: Path) -> bool:
        file_key = str(pdf_path)
        file_stat = pdf_path.stat()
        
        if file_key in self.processed_files:
            recorded_mtime = self.processed_files[file_key].get('modified_time')
            return recorded_mtime == file_stat.st_mtime
        return False
    
    def mark_file_processed(self, pdf_path: Path, result: ProcessingResult):
        file_key = str(pdf_path)
        file_stat = pdf_path.stat()
        
        self.processed_files[file_key] = {
            'modified_time': file_stat.st_mtime,
            'processed_at': str(Path().cwd()),
            'success': result.success,
            'table_groups_count': result.table_groups_count,
            'total_tables_count': result.total_tables_count,
            'output_path': result.output_path
        }
        self._save_processing_log()
    
    def get_unprocessed_files(self, input_dir: Path) -> List[Path]:
        pdf_files = []
        for pattern in ("*.pdf", "*.PDF"):
            pdf_files.extend(input_dir.glob(pattern))
        
        unprocessed = [f for f in pdf_files if not self.is_file_processed(f)]
        self.logger.info(f"Found {len(unprocessed)} unprocessed files out of {len(pdf_files)} total")
        return unprocessed


class PDFChunkExtractor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def extract_chunks_from_pdf(self, pdf_path: str) -> List[Union[TextChunk, TableChunk]]:
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        self.logger.info(f"Extracting chunks from {pdf_file.name}")
        
        try:
            pdf_elements = partition_pdf(
                filename=pdf_path,
                strategy="hi_res",
                infer_table_structure=True,
                languages=['deu']
            )
        except Exception as e:
            self.logger.error(f"Failed to parse PDF {pdf_path}: {e}")
            raise

        chunks = []
        for i, element in enumerate(pdf_elements):
            page_num = getattr(element.metadata, 'page_number', None)
            
            if element.category == "Table":
                table_chunk = self._create_table_chunk(element, page_num, i)
                if table_chunk:
                    chunks.append(table_chunk)
            else:
                text_chunk = self._create_text_chunk(element, page_num, i)
                if text_chunk:
                    chunks.append(text_chunk)
        
        self.logger.info(f"Extracted {len(chunks)} chunks ({len([c for c in chunks if c.chunk_type == ChunkType.TABLE])} tables)")
        return chunks

    def _create_table_chunk(self, element, page_num: Optional[int], index: int) -> Optional[TableChunk]:
        raw_html = getattr(element.metadata, 'text_as_html', None)
        
        if raw_html:
            try:
                markdown_content = markdownify.markdownify(raw_html)
            except Exception:
                markdown_content = element.text if element.text else None
        else:
            markdown_content = element.text if element.text else None
        
        if markdown_content and markdown_content.strip():
            return TableChunk(
                content=markdown_content.strip(),
                page_number=page_num,
                element_index=index,
                raw_html=raw_html
            )
        return None

    def _create_text_chunk(self, element, page_num: Optional[int], index: int) -> Optional[TextChunk]:
        if element.text and element.text.strip():
            return TextChunk(
                content=element.text.strip(),
                page_number=page_num,
                element_index=index
            )
        return None

    def group_tables_with_context(self, chunks: List[Union[TextChunk, TableChunk]]) -> List[TableGroup]:
        table_groups = []
        i = 0
        
        while i < len(chunks):
            if chunks[i].chunk_type == ChunkType.TABLE:
                group = self._extract_table_group(chunks, i)
                table_groups.append(group)
                i = group.end_index + 1
            else:
                i += 1
        
        self.logger.info(f"Created {len(table_groups)} table groups")
        return table_groups

    def _extract_table_group(self, chunks: List[Union[TextChunk, TableChunk]], start_idx: int) -> TableGroup:
        tables = []
        current_idx = start_idx
        
        while (current_idx < len(chunks) and 
               chunks[current_idx].chunk_type == ChunkType.TABLE and 
               len(tables) < Config.MAX_TABLES_PER_GROUP):
            tables.append(chunks[current_idx])
            current_idx += 1
        
        previous_text = self._find_previous_text_chunk(chunks, start_idx)
        next_text = self._find_next_text_chunk(chunks, current_idx - 1)
        
        return TableGroup(
            tables=tables,
            previous_text=previous_text,
            next_text=next_text,
            start_index=start_idx,
            end_index=current_idx - 1
        )

    def _find_previous_text_chunk(self, chunks: List[Union[TextChunk, TableChunk]], 
                                 table_start_idx: int) -> Optional[TextChunk]:
        for i in range(table_start_idx - 1, -1, -1):
            if chunks[i].chunk_type == ChunkType.TEXT:
                return chunks[i]
        return None

    def _find_next_text_chunk(self, chunks: List[Union[TextChunk, TableChunk]], 
                             table_end_idx: int) -> Optional[TextChunk]:
        for i in range(table_end_idx + 1, len(chunks)):
            if chunks[i].chunk_type == ChunkType.TEXT:
                return chunks[i]
        return None


class GeminiDescriptionGenerator:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Gemini API key is required")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(Config.GEMINI_MODEL)
        self.logger = logging.getLogger(__name__)
    
    def generate_table_group_description(self, table_group: TableGroup) -> str:
        prompt = self._build_enrichment_prompt(table_group)
        
        try:
            response = self.model.generate_content(prompt)
            description = response.text.strip()
            self.logger.debug(f"Generated description for table group with {len(table_group.tables)} tables")
            return description
        except Exception as e:
            error_msg = f"Fehler bei der Beschreibungsgenerierung: {str(e)}"
            self.logger.error(error_msg)
            return error_msg

    def _build_enrichment_prompt(self, table_group: TableGroup) -> str:
        previous_context = table_group.previous_text.content if table_group.previous_text else "N/A"
        next_context = table_group.next_text.content if table_group.next_text else "N/A"
        
        previous_context_preview = previous_context[:Config.MAX_PAGE_CONTENT_LENGTH]
        if len(previous_context) > Config.MAX_PAGE_CONTENT_LENGTH:
            previous_context_preview += "..."
            
        next_context_preview = next_context[:Config.MAX_PAGE_CONTENT_LENGTH]
        if len(next_context) > Config.MAX_PAGE_CONTENT_LENGTH:
            next_context_preview += "..."
        
        tables_content = "\n\n".join([
            f"=== TABELLE {i+1} (Seite {table.page_number}) ===\n{table.content}"
            for i, table in enumerate(table_group.tables)
        ])
        
        return f"""
        Analyze the following table group and its context to create a comprehensive natural language description.
        
        CONTEXT BEFORE TABLES:
        {previous_context_preview}
        
        TABLE GROUP ({len(table_group.tables)} tables):
        {tables_content}
        
        CONTEXT AFTER TABLES:
        {next_context_preview}
        
        Please provide:
        1. A clear, natural language description of what these tables show collectively
        2. The key insights or patterns visible across all tables in the group
        3. How this table group relates to the surrounding context
        4. Any notable trends, comparisons, or significant values across the tables
        5. The relationship between the individual tables in this group
        
        Format your response as a comprehensive description that would be useful for an LLM to understand the tables' content and significance.
        Respond in German if the context is in German, otherwise respond in English.
        Keep the description concise but informative (max 300 words).
        """


class TableContextEnricher:
    def __init__(self, gemini_api_key: str):
        self.extractor = PDFChunkExtractor()
        self.generator = GeminiDescriptionGenerator(gemini_api_key)
        self.logger = logging.getLogger(__name__)
    
    def process_single_pdf(self, pdf_path: Path, output_dir: Path) -> ProcessingResult:
        pdf_filename = pdf_path.name
        
        try:
            output_filename = pdf_path.stem + "_enriched_tables.md"
            output_path = output_dir / output_filename
            
            enriched_groups = self.process_pdf_with_enriched_tables(str(pdf_path), str(output_path))
            
            total_tables = sum(len(group.table_group.tables) for group in enriched_groups)
            
            return ProcessingResult(
                pdf_filename=pdf_filename,
                success=True,
                table_groups_count=len(enriched_groups),
                total_tables_count=total_tables,
                output_path=str(output_path)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to process {pdf_filename}: {e}")
            return ProcessingResult(
                pdf_filename=pdf_filename,
                success=False,
                error_message=str(e)
            )
    
    def process_pdf_with_enriched_tables(self, pdf_path: str, output_path: Optional[str] = None) -> List[EnrichedTableGroup]:
        chunks = self.extractor.extract_chunks_from_pdf(pdf_path)
        table_groups = self.extractor.group_tables_with_context(chunks)
        
        enriched_groups = []
        self.logger.info(f"Enriching {len(table_groups)} table groups with AI descriptions")
        
        for group in table_groups:
            enriched_group = self._enrich_table_group(group)
            enriched_groups.append(enriched_group)
        
        if output_path:
            self._save_enriched_content(enriched_groups, output_path)
        
        return enriched_groups

    def _enrich_table_group(self, table_group: TableGroup) -> EnrichedTableGroup:
        description = self.generator.generate_table_group_description(table_group)
        enriched_content = self._format_enriched_table_group(table_group, description)
        
        return EnrichedTableGroup(
            table_group=table_group,
            natural_description=description,
            enriched_content=enriched_content
        )

    def _format_enriched_table_group(self, table_group: TableGroup, description: str) -> str:
        pages = set(table.page_number for table in table_group.tables if table.page_number)
        page_info = f"Seiten {', '.join(map(str, sorted(pages)))}" if pages else "Unbekannte Seite"
        
        tables_section = "\n\n".join([
            f"--- TABELLE {i+1} ---\n{table.content}"
            for i, table in enumerate(table_group.tables)
        ])
        
        return f"""
=== TABELLENGRUPPE ({len(table_group.tables)} Tabellen, {page_info}) ===

KI-GENERIERTE BESCHREIBUNG:
{description}

KONTEXT VOR DEN TABELLEN:
{table_group.previous_text.content if table_group.previous_text else 'N/A'}

TABELLEN:
{tables_section}

KONTEXT NACH DEN TABELLEN:
{table_group.next_text.content if table_group.next_text else 'N/A'}

=== ENDE TABELLENGRUPPE ===
"""

    def _save_enriched_content(self, enriched_groups: List[EnrichedTableGroup], output_path: str):
        try:
            with open(output_path, 'w', encoding=Config.DEFAULT_OUTPUT_ENCODING) as f:
                f.write("# Angereicherte Tabellengruppen mit KI-Beschreibungen\n\n")
                
                for i, group in enumerate(enriched_groups, 1):
                    f.write(f"## Tabellengruppe {i}\n\n")
                    f.write(group.enriched_content)
                    f.write("\n\n" + "="*80 + "\n\n")
                    
            self.logger.info(f"Saved enriched content to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save enriched content: {e}")
            raise


class DocumentProcessor:
    def __init__(self, gemini_api_key: str):
        self.enricher = TableContextEnricher(gemini_api_key)
        self.tracker = ProcessingTracker(Config.PROCESSING_LOG_FILE)
        self.logger = logging.getLogger(__name__)
        
        self.input_dir = Path(Config.INPUT_DIR)
        self.output_dir = Path(Config.OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True)
    
    def process_all_new_documents(self) -> List[ProcessingResult]:
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        
        unprocessed_files = self.tracker.get_unprocessed_files(self.input_dir)
        
        if not unprocessed_files:
            self.logger.info("No new documents to process")
            return []
        
        self.logger.info(f"Processing {len(unprocessed_files)} new documents")
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=Config.MAX_CONCURRENT_WORKERS) as executor:
            future_to_file = {
                executor.submit(self.enricher.process_single_pdf, pdf_file, self.output_dir): pdf_file
                for pdf_file in unprocessed_files
            }
            
            for future in concurrent.futures.as_completed(future_to_file):
                pdf_file = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    self.tracker.mark_file_processed(pdf_file, result)
                    
                    if result.success:
                        self.logger.info(f"Successfully processed {result.pdf_filename}")
                    else:
                        self.logger.error(f"Failed to process {result.pdf_filename}: {result.error_message}")
                        
                except Exception as e:
                    error_result = ProcessingResult(
                        pdf_filename=pdf_file.name,
                        success=False,
                        error_message=str(e)
                    )
                    results.append(error_result)
                    self.tracker.mark_file_processed(pdf_file, error_result)
                    self.logger.error(f"Exception processing {pdf_file.name}: {e}")
        
        return results


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    dotenv.load_dotenv()
    api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        logger.error("GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set")
        print("Bitte setze die Umgebungsvariable GOOGLE_API_KEY oder GEMINI_API_KEY")
        return
    
    try:
        processor = DocumentProcessor(api_key)
        results = processor.process_all_new_documents()
        
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        print(f"\n{'='*60}")
        print(f"VERARBEITUNGSÜBERSICHT")
        print(f"{'='*60}")
        print(f"Verarbeitete Dokumente: {len(successful)}")
        print(f"Fehlgeschlagene Dokumente: {len(failed)}")
        
        if successful:
            total_groups = sum(r.table_groups_count for r in successful)
            total_tables = sum(r.total_tables_count for r in successful)
            print(f"Gesamt Tabellengruppen: {total_groups}")
            print(f"Gesamt Tabellen: {total_tables}")
            
            print(f"\nERFOLGREICH VERARBEITET:")
            for result in successful:
                print(f"  ✓ {result.pdf_filename} -> {result.table_groups_count} Gruppen, {result.total_tables_count} Tabellen")
        
        if failed:
            print(f"\nFEHLGESCHLAGEN:")
            for result in failed:
                print(f"  ✗ {result.pdf_filename}: {result.error_message}")
        
        print(f"\nAusgabeordner: {Config.OUTPUT_DIR}")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"Unerwarteter Fehler: {e}")


if __name__ == "__main__":
    main()