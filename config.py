import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import dotenv


@dataclass
class ProcessingConfig:
    """Configuration settings for document processing"""
    input_dir: str = "Geschaeftsberichte"
    output_dir: str = "Extrahierter_Text_Markdown"
    processing_log_file: str = "processing_log.json"
    max_concurrent_workers: int = 2
    enable_logging: bool = True
    log_level: str = "INFO"


@dataclass
class TableEnricherConfig:
    """Configuration for table enrichment with AI"""
    context_window: int = 3
    max_page_content_length: int = 1000
    gemini_model: str = 'gemini-1.5-flash'
    max_tables_per_group: int = 5
    enable_ai_descriptions: bool = True


@dataclass
class TextExtractorConfig:
    """Configuration for basic text extraction"""
    strategy: str = "hi_res"
    infer_table_structure: bool = True
    languages: list = None
    max_threads: int = 4
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = ['deu']


@dataclass
class GlobalConfig:
    """Main configuration class containing all module configs"""
    processing: ProcessingConfig
    table_enricher: TableEnricherConfig
    text_extractor: TextExtractorConfig
    
    # API Keys
    gemini_api_key: Optional[str] = None
    
    # File encoding
    default_encoding: str = 'utf-8'
    
    def __post_init__(self):
        # Load environment variables
        dotenv.load_dotenv()
        
        # Try to get API key from environment
        if not self.gemini_api_key:
            self.gemini_api_key = (
                os.getenv('GOOGLE_API_KEY') or 
                os.getenv('GEMINI_API_KEY')
            )
    
    @classmethod
    def create_default(cls) -> 'GlobalConfig':
        """Create default configuration"""
        return cls(
            processing=ProcessingConfig(),
            table_enricher=TableEnricherConfig(),
            text_extractor=TextExtractorConfig()
        )
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        # Check required directories
        input_path = Path(self.processing.input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_path}")
        
        # Create output directory if it doesn't exist
        output_path = Path(self.processing.output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Check API key for AI features
        if self.table_enricher.enable_ai_descriptions and not self.gemini_api_key:
            raise ValueError("Gemini API key required for AI table descriptions")
        
        return True