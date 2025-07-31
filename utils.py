import json
import logging
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ProcessingResult:
    """Result of processing a single document"""
    filename: str
    processor_name: str
    success: bool
    processing_time: float
    error_message: str = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ProcessingTracker:
    """Tracks processed files to avoid reprocessing"""
    
    def __init__(self, log_file_path: str):
        self.log_file_path = Path(log_file_path)
        self.processed_files = self._load_processing_log()
        self.logger = logging.getLogger(__name__)
    
    def _load_processing_log(self) -> Dict[str, dict]:
        if self.log_file_path.exists():
            try:
                with open(self.log_file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load processing log: {e}")
                return {}
        return {}
    
    def _save_processing_log(self):
        try:
            with open(self.log_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.processed_files, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save processing log: {e}")
    
    def is_file_processed(self, file_path: Path, processor_name: str) -> bool:
        file_key = f"{processor_name}:{str(file_path)}"
        file_stat = file_path.stat()
        
        if file_key in self.processed_files:
            recorded_mtime = self.processed_files[file_key].get('modified_time')
            return recorded_mtime == file_stat.st_mtime
        return False
    
    def mark_file_processed(self, file_path: Path, result: ProcessingResult):
        file_key = f"{result.processor_name}:{str(file_path)}"
        file_stat = file_path.stat()
        
        self.processed_files[file_key] = {
            'modified_time': file_stat.st_mtime,
            'processed_at': datetime.now().isoformat(),
            'success': result.success,
            'processing_time': result.processing_time,
            'metadata': result.metadata
        }
        self._save_processing_log()
    
    def get_unprocessed_files(self, input_dir: Path, processor_name: str, 
                             file_patterns: List[str] = None) -> List[Path]:
        if file_patterns is None:
            file_patterns = ["*.pdf", "*.PDF"]
        
        all_files = []
        for pattern in file_patterns:
            all_files.extend(input_dir.glob(pattern))
        
        unprocessed = [
            f for f in all_files 
            if not self.is_file_processed(f, processor_name)
        ]
        
        self.logger.info(
            f"Found {len(unprocessed)} unprocessed files for {processor_name} "
            f"out of {len(all_files)} total"
        )
        return unprocessed


def setup_logging(level: str = "INFO", log_file: str = None):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers
    )


def create_output_filename(input_file: Path, processor_name: str, 
                          extension: str = ".md") -> str:
    """Create standardized output filename"""
    base_name = input_file.stem
    return f"{base_name}_{processor_name}{extension}"


def safe_file_operation(operation, *args, **kwargs):
    """Safely execute file operations with error handling"""
    try:
        return operation(*args, **kwargs), None
    except Exception as e:
        return None, str(e)