"""
Shared utility functions for the RAG system
"""
import logging
import sys
from pathlib import Path
from typing import Optional, List


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def validate_file_exists(file_path: str) -> Path:
    """
    Validate that a file exists
    
    Args:
        file_path: Path to file
        
    Returns:
        Path object if file exists
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return path


def validate_directory_exists(directory: str) -> Path:
    """
    Validate that a directory exists
    
    Args:
        directory: Path to directory
        
    Returns:
        Path object if directory exists
        
    Raises:
        FileNotFoundError: If directory doesn't exist
    """
    path = Path(directory)
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    return path


def find_files_by_extension(directory: str, extension: str) -> List[Path]:
    """
    Find all files with specific extension in directory
    
    Args:
        directory: Directory to search
        extension: File extension (e.g., '.pdf', '.md')
        
    Returns:
        List of Path objects
    """
    dir_path = validate_directory_exists(directory)
    pattern = f"*{extension}" if not extension.startswith('.') else f"*{extension}"
    return list(dir_path.glob(pattern))


def safe_create_directory(directory: str) -> Path:
    """
    Safely create a directory if it doesn't exist
    
    Args:
        directory: Directory path
        
    Returns:
        Path object
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in MB
    """
    path = validate_file_exists(file_path)
    return path.stat().st_size / (1024 * 1024)


def truncate_text(text: str, max_length: int = 200) -> str:
    """
    Truncate text to maximum length with ellipsis
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."