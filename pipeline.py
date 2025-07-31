import logging
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from config import GlobalConfig
from utils import ProcessingTracker, ProcessingResult
from base_processor import BaseProcessor, processor_registry


class ProcessingPipeline:
    """Main processing pipeline that orchestrates all document processors"""
    
    def __init__(self, config: GlobalConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.tracker = ProcessingTracker(config.processing.processing_log_file)
        
        # Setup directories
        self.input_dir = Path(config.processing.input_dir)
        self.output_dir = Path(config.processing.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Validate configuration
        config.validate()
    
    def register_processors(self):
        """Register all available processors"""
        from processors.text_extractor import TextExtractorProcessor
        from processors.table_enricher import TableEnricherProcessor
        
        # Register text extractor
        text_processor = TextExtractorProcessor(self.config.text_extractor)
        text_processor.set_tracker(self.tracker)
        processor_registry.register(text_processor)
        
        # Register table enricher (if API key available)
        if self.config.gemini_api_key and self.config.table_enricher.enable_ai_descriptions:
            table_processor = TableEnricherProcessor(
                self.config.table_enricher, 
                self.config.gemini_api_key
            )
            table_processor.set_tracker(self.tracker)
            processor_registry.register(table_processor)
        else:
            self.logger.warning("Table enricher not registered - missing API key or disabled")
    
    def run_all_processors(self, processors: Optional[List[str]] = None) -> Dict[str, List[ProcessingResult]]:
        """Run all specified processors (or all registered if None)"""
        if processors is None:
            processors = processor_registry.get_processor_names()
        
        if not processors:
            self.logger.warning("No processors to run")
            return {}
        
        self.logger.info(f"Starting processing pipeline with processors: {processors}")
        start_time = datetime.now()
        
        all_results = {}
        
        for processor_name in processors:
            processor = processor_registry.get_processor(processor_name)
            if not processor:
                self.logger.error(f"Processor not found: {processor_name}")
                continue
            
            self.logger.info(f"Running processor: {processor_name}")
            
            if self.config.processing.max_concurrent_workers > 1:
                results = self._run_processor_concurrent(processor)
            else:
                results = self._run_processor_sequential(processor)
            
            all_results[processor_name] = results
        
        end_time = datetime.now()
        self.logger.info(f"Pipeline completed in {end_time - start_time}")
        
        return all_results
    
    def _run_processor_sequential(self, processor: BaseProcessor) -> List[ProcessingResult]:
        """Run processor sequentially"""
        return processor.process_files(self.input_dir, self.output_dir)
    
    def _run_processor_concurrent(self, processor: BaseProcessor) -> List[ProcessingResult]:
        """Run processor with concurrent execution"""
        # Get files to process
        file_list = self.tracker.get_unprocessed_files(
            self.input_dir, 
            processor.name, 
            processor.get_file_patterns()
        )
        
        files_to_process = [f for f in file_list if processor.should_process_file(f)]
        
        if not files_to_process:
            return []
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.processing.max_concurrent_workers
        ) as executor:
            future_to_file = {
                executor.submit(processor.process_single_file, file_path, self.output_dir): file_path
                for file_path in files_to_process
            }
            
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    
                    # Track the processing result
                    processor.tracker.mark_file_processed(file_path, result)
                    results.append(result)
                    
                    if result.success:
                        self.logger.info(f"Successfully processed {file_path.name} with {processor.name}")
                    else:
                        self.logger.error(f"Failed to process {file_path.name}: {result.error_message}")
                        
                except Exception as e:
                    error_result = ProcessingResult(
                        filename=file_path.name,
                        processor_name=processor.name,
                        success=False,
                        processing_time=0,
                        error_message=str(e)
                    )
                    
                    processor.tracker.mark_file_processed(file_path, error_result)
                    results.append(error_result)
                    self.logger.error(f"Exception processing {file_path.name}: {e}")
        
        return results
    
    def get_pipeline_summary(self, all_results: Dict[str, List[ProcessingResult]]) -> Dict:
        """Generate comprehensive pipeline summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_processors': len(all_results),
            'processor_summaries': {},
            'overall_stats': {
                'total_files': 0,
                'total_successful': 0,
                'total_failed': 0,
                'total_processing_time': 0
            }
        }
        
        for processor_name, results in all_results.items():
            processor = processor_registry.get_processor(processor_name)
            if processor:
                proc_summary = processor.get_summary(results)
                summary['processor_summaries'][processor_name] = proc_summary
                
                # Update overall stats
                summary['overall_stats']['total_files'] += proc_summary['total_files']
                summary['overall_stats']['total_successful'] += proc_summary['successful']
                summary['overall_stats']['total_failed'] += proc_summary['failed']
                summary['overall_stats']['total_processing_time'] += proc_summary['total_processing_time']
        
        # Calculate overall success rate
        total_files = summary['overall_stats']['total_files']
        if total_files > 0:
            summary['overall_stats']['success_rate'] = (
                summary['overall_stats']['total_successful'] / total_files
            )
        else:
            summary['overall_stats']['success_rate'] = 0
        
        return summary
    
    def print_summary(self, all_results: Dict[str, List[ProcessingResult]]):
        """Print formatted summary to console"""
        summary = self.get_pipeline_summary(all_results)
        
        print(f"\n{'='*80}")
        print(f"DOCUMENT PROCESSING PIPELINE SUMMARY")
        print(f"{'='*80}")
        print(f"Zeitpunkt: {summary['timestamp']}")
        print(f"Verarbeitete Prozessoren: {summary['total_processors']}")
        
        overall = summary['overall_stats']
        print(f"\nGESAMTÜBERSICHT:")
        print(f"  Dateien gesamt: {overall['total_files']}")
        print(f"  Erfolgreich: {overall['total_successful']}")
        print(f"  Fehlgeschlagen: {overall['total_failed']}")
        print(f"  Erfolgsrate: {overall['success_rate']:.1%}")
        print(f"  Verarbeitungszeit: {overall['total_processing_time']:.2f}s")
        
        print(f"\nPROZESSOR DETAILS:")
        for proc_name, proc_summary in summary['processor_summaries'].items():
            print(f"  {proc_name.upper()}:")
            print(f"    Dateien: {proc_summary['total_files']}")
            print(f"    Erfolgreich: {proc_summary['successful']}")
            print(f"    Fehlgeschlagen: {proc_summary['failed']}")
            print(f"    Erfolgsrate: {proc_summary['success_rate']:.1%}")
            print(f"    Ø Zeit/Datei: {proc_summary['average_processing_time']:.2f}s")
        
        # Show failed files if any
        failed_files = []
        for processor_name, results in all_results.items():
            for result in results:
                if not result.success:
                    failed_files.append((processor_name, result))
        
        if failed_files:
            print(f"\nFEHLGESCHLAGENE DATEIEN:")
            for proc_name, result in failed_files:
                print(f"  ✗ {result.filename} ({proc_name}): {result.error_message}")
        
        print(f"\nAusgabeordner: {self.output_dir}")
        print(f"{'='*80}")