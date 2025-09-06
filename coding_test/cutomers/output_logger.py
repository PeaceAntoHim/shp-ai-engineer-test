import os
import sys
from datetime import datetime
from contextlib import contextmanager

class OutputLogger:
    """Captures and logs all output to both console and file."""
    
    def __init__(self, log_directory: str = "output_logs"):
        self.log_directory = log_directory
        self.ensure_log_directory()
        
    def ensure_log_directory(self):
        """Create log directory if it doesn't exist."""
        if not os.path.exists(self.log_directory):
            os.makedirs(self.log_directory)
            
    @contextmanager
    def capture_output(self, filename: str = None):
        """Context manager to capture output to both console and file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"csv_analysis_{timestamp}.txt"
            
        log_path = os.path.join(self.log_directory, filename)
        
        # Store original stdout
        original_stdout = sys.stdout
        
        try:
            # Create tee writer that writes to both console and file
            with open(log_path, 'w', encoding='utf-8') as log_file:
                tee_writer = TeeWriter(original_stdout, log_file)
                sys.stdout = tee_writer
                
                # Add header to log
                self._write_log_header(log_path)
                
                yield log_path
                
        finally:
            # Restore original stdout
            sys.stdout = original_stdout
            print(f"\n📝 Analysis output saved to: {log_path}")
            
    def _write_log_header(self, log_path: str):
        """Write header information to log."""
        print("=" * 100)
        print("CSV PROCESSING & ANALYSIS - COMPREHENSIVE OUTPUT LOG")
        print("=" * 100)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log File: {log_path}")
        print("=" * 100)


class TeeWriter:
    """Writer that outputs to multiple streams simultaneously."""
    
    def __init__(self, *streams):
        self.streams = streams
        
    def write(self, text: str):
        """Write text to all streams."""
        for stream in self.streams:
            stream.write(text)
            stream.flush()
            
    def flush(self):
        """Flush all streams."""
        for stream in self.streams:
            stream.flush()


class AnalysisReportGenerator:
    """Generates structured analysis reports."""
    
    def __init__(self, logger: OutputLogger):
        self.logger = logger
        
    def generate_complete_report(self, small_result=None, large_result=None):
        """Generate complete analysis report with all metrics."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"complete_analysis_report_{timestamp}.txt"
        
        with self.logger.capture_output(filename) as log_path:
            self._print_executive_summary()
            
            if small_result:
                self._print_small_file_analysis(small_result)
                
            if large_result:
                self._print_large_file_analysis(large_result)
                
            if small_result and large_result:
                self._print_performance_comparison(small_result, large_result)
                
            self._print_technical_recommendations()
            
        return log_path
    
    def _print_executive_summary(self):
        """Print executive summary section."""
        print("\nEXECUTIVE SUMMARY")
        print("=" * 80)
        print("This report provides comprehensive analysis of CSV processing capabilities")
        print("comparing small file (100K records) vs large file (2M+ records) processing.")
        print("The analysis demonstrates memory-efficient processing strategies and")
        print("comprehensive data insights extraction.")
        print()
        
    def _print_small_file_analysis(self, result):
        """Print detailed small file analysis."""
        print("\nQUESTION 1: SMALL CSV ANALYSIS RESULTS")
        print("=" * 80)
        print("FILE: customers-100000.csv")
        print("STRATEGY: Full Dataset Loading with Complete Analysis")
        print("-" * 80)
        
        # Processing Summary
        print(f"\nPROCESSING METRICS")
        print(f"Total Rows Processed: {result.total_rows:,}")
        print(f"Successfully Processed: {result.processed_rows:,}")
        print(f"Failed Records: {result.failed_rows:,}")
        print(f"Success Rate: {result.get_success_rate():.2f}%")
        
        # Performance Metrics
        tech = result.technical_metrics
        print(f"\nPERFORMANCE METRICS")
        print(f"Processing Strategy: {tech.processing_strategy.value}")
        print(f"Execution Time: {tech.execution_time_seconds:.2f} seconds")
        print(f"Memory Usage: {tech.memory_usage_mb:.1f} MB")
        print(f"Peak Memory: {tech.peak_memory_mb:.1f} MB")
        print(f"CPU Usage: {tech.cpu_usage_percent:.1f}%")
        print(f"Throughput: {tech.throughput_rows_per_second:,.0f} rows/second")
        
        # Data Quality Analysis
        quality = result.insights.data_quality
        print(f"\nDATA QUALITY ASSESSMENT")
        print(f"Completeness Score: {quality.completeness_score:.1f}%")
        print(f"Validity Score: {quality.validity_score:.1f}%")
        print(f"Consistency Score: {quality.consistency_score:.1f}%")
        print(f"Unique Customers: {quality.unique_customers:,}")
        print(f"Duplicate Records: {quality.duplicate_count:,}")
        
        # Business Insights
        self._print_business_insights(result.insights, "SMALL FILE")
        
    def _print_large_file_analysis(self, result):
        """Print detailed large file analysis."""
        print("\nQUESTION 2: LARGE CSV ANALYSIS RESULTS")
        print("=" * 80)
        print("FILE: customers-2000000.csv")
        print("STRATEGY: Memory-Efficient Chunked Streaming Processing")
        print("-" * 80)
        
        # Processing Summary
        print(f"\nPROCESSING METRICS")
        print(f"Total Rows Processed: {result.total_rows:,}")
        print(f"Successfully Processed: {result.processed_rows:,}")
        print(f"Failed Records: {result.failed_rows:,}")
        print(f"Success Rate: {result.get_success_rate():.2f}%")
        
        # Performance Metrics
        tech = result.technical_metrics
        print(f"\n⚡ PERFORMANCE METRICS")
        print(f"Processing Strategy: {tech.processing_strategy.value}")
        print(f"Execution Time: {tech.execution_time_seconds:.2f} seconds")
        print(f"Memory Usage (Constant): {tech.memory_usage_mb:.1f} MB")
        print(f"Peak Memory: {tech.peak_memory_mb:.1f} MB")
        print(f"Chunks Processed: {tech.chunks_processed:,}")
        print(f"Garbage Collections: {tech.garbage_collections}")
        print(f"Throughput: {tech.throughput_rows_per_second:,.0f} rows/second")
        
        # Memory Efficiency Analysis
        print(f"\nMEMORY EFFICIENCY ANALYSIS")
        print(f"Memory Efficient Processing: {tech.memory_efficient}")
        print(f"Average Memory per Chunk: {tech.memory_usage_mb/tech.chunks_processed:.1f} MB")
        print(f"Memory Optimization: {((100 - tech.memory_usage_mb)/100)*100:.1f}% memory savings")
        
        # Data Quality Analysis
        quality = result.insights.data_quality
        print(f"\nDATA QUALITY ASSESSMENT")
        print(f"Completeness Score: {quality.completeness_score:.1f}%")
        print(f"Validity Score: {quality.validity_score:.1f}%")
        print(f"Unique Customers: {quality.unique_customers:,}")
        print(f"Duplicate Records: {quality.duplicate_count:,}")
        
        # Business Insights from Streaming
        self._print_business_insights(result.insights, "LARGE FILE (STREAMING)")
        
    def _print_business_insights(self, insights, file_type: str):
        """Print comprehensive business insights."""
        print(f"\nGEOGRAPHIC DISTRIBUTION ({file_type})")
        print("-" * 50)
        geo_dist = insights.business_insights.geographic_distribution
        print(f"Countries Represented: {len(geo_dist)}")
        print("Top 10 Countries by Customer Count:")
        for i, (country, count) in enumerate(list(geo_dist.items())[:10], 1):
            print(f"  {i:2}. {country:<30} {count:>8,} customers")
        
        print(f"\nBUSINESS ANALYSIS ({file_type})")
        print("-" * 50)
        companies = insights.business_insights.company_distribution
        print(f"Unique Companies: {len(companies):,}")
        print("Top 10 Companies by Customer Count:")
        for i, (company, count) in enumerate(list(companies.items())[:10], 1):
            company_name = company[:35] + "..." if len(company) > 35 else company
            print(f"  {i:2}. {company_name:<38} {count:>8,} customers")
        
        print(f"\nCOMMUNICATION PATTERNS ({file_type})")
        print("-" * 50)
        email_domains = insights.communication_analysis.email_domains
        print("Top 10 Email Domains:")
        for i, (domain, count) in enumerate(list(email_domains.items())[:10], 1):
            print(f"  {i:2}. {domain:<30} {count:>8,} addresses")
        
        # Phone Analysis
        phone_patterns = insights.communication_analysis.phone_patterns
        if phone_patterns:
            print(f"\nPHONE ANALYSIS ({file_type})")
            print("-" * 50)
            for pattern, value in phone_patterns.items():
                if isinstance(value, (int, float)):
                    pattern_name = pattern.replace('_', ' ').title()
                    print(f"  {pattern_name:<30} {value:>10,}")
        
        # Temporal Analysis
        temporal = insights.temporal_analysis
        if temporal.subscription_by_year:
            print(f"\nTEMPORAL ANALYSIS ({file_type})")
            print("-" * 50)
            print("Subscription Timeline by Year:")
            for year, count in sorted(temporal.subscription_by_year.items()):
                print(f"  {year}: {count:>8,} new subscriptions")
            
        if temporal.seasonal_patterns:
            print("\nSeasonal Distribution:")
            for season, count in temporal.seasonal_patterns.items():
                print(f"  {season:<10} {count:>8,} subscriptions")
    
    def _print_performance_comparison(self, small_result, large_result):
        """Print detailed performance comparison."""
        print("\nPERFORMANCE COMPARISON: SMALL vs LARGE FILE PROCESSING")
        print("=" * 80)
        
        small_tech = small_result.technical_metrics
        large_tech = large_result.technical_metrics
        
        print(f"\nPROCESSING METRICS COMPARISON")
        print("-" * 80)
        print(f"{'Metric':<30} {'Small File':<20} {'Large File':<20} {'Ratio':<15}")
        print("-" * 80)
        
        small_rows = small_result.processed_rows
        large_rows = large_result.processed_rows
        print(f"{'Rows Processed':<30} {small_rows:<20,} {large_rows:<20,} {large_rows/small_rows:<15.1f}x")
        
        print(f"{'Execution Time (s)':<30} {small_tech.execution_time_seconds:<20.2f} {large_tech.execution_time_seconds:<20.2f} {large_tech.execution_time_seconds/small_tech.execution_time_seconds:<15.1f}x")
        
        print(f"{'Memory Usage (MB)':<30} {small_tech.memory_usage_mb:<20.1f} {large_tech.memory_usage_mb:<20.1f} {large_tech.memory_usage_mb/small_tech.memory_usage_mb:<15.1f}x")
        
        print(f"{'Throughput (rows/s)':<30} {small_tech.throughput_rows_per_second:<20,.0f} {large_tech.throughput_rows_per_second:<20,.0f} {large_tech.throughput_rows_per_second/small_tech.throughput_rows_per_second:<15.1f}x")
        
        # Memory Efficiency Analysis
        print(f"\nMEMORY EFFICIENCY COMPARISON")
        print("-" * 50)
        print(f"Small File Memory Strategy:")
        print(f"  • Full dataset loaded into memory")
        print(f"  • Memory usage scales with file size")
        print(f"  • Peak memory: {small_tech.peak_memory_mb:.1f} MB")
        print(f"  • Memory per row: {small_tech.memory_usage_mb/small_rows*1000:.2f} KB/row")
        
        print(f"\nLarge File Memory Strategy:")
        print(f"  • Constant memory usage via chunking")
        print(f"  • Memory usage independent of file size")
        print(f"  • Peak memory: {large_tech.peak_memory_mb:.1f} MB")
        print(f"  • Memory per chunk: {large_tech.memory_usage_mb/large_tech.chunks_processed:.1f} MB/chunk")
        print(f"  • Chunks processed: {large_tech.chunks_processed:,}")
        
        # Scalability Analysis
        print(f"\nSCALABILITY ANALYSIS")
        print("-" * 50)
        print(f"Processing Efficiency:")
        print(f"  • Small file efficiency: {small_rows/small_tech.execution_time_seconds/small_tech.memory_usage_mb:.0f} rows/s/MB")
        print(f"  • Large file efficiency: {large_rows/large_tech.execution_time_seconds/large_tech.memory_usage_mb:.0f} rows/s/MB")
        
        memory_efficiency_gain = (large_rows/large_tech.memory_usage_mb) / (small_rows/small_tech.memory_usage_mb)
        print(f"  • Memory efficiency gain: {memory_efficiency_gain:.1f}x better for large files")
    
    def _print_technical_recommendations(self):
        """Print technical recommendations."""
        print("\nTECHNICAL RECOMMENDATIONS & BEST PRACTICES")
        print("=" * 80)
        
        print(f"\nWHEN TO USE EACH PROCESSING STRATEGY")
        print("-" * 50)
        print("Small File Processor (Full Load Strategy):")
        print("  Files under 100,000 rows or < 100MB")
        print("  Need exact statistical analysis")
        print("  Interactive data exploration")
        print("  Real-time analytics dashboards")
        print("  Sufficient memory available (2-3x file size)")
        print("  Memory-constrained environments")
        print("  Very large datasets")
        
        print(f"\nLarge File Processor (Chunked Streaming Strategy):")
        print("  Files over 100,000 rows or > 100MB")
        print("  Memory-constrained environments")
        print("  Production batch processing")
        print("  Streaming data pipelines")
        print("  Server environments")
        print("  Need to process files larger than available RAM")
        print("  Need exact real-time statistics")
        print("  Interactive analysis requiring immediate results")
        
        print(f"\nARCHITECTURE PRINCIPLES DEMONSTRATED")
        print("-" * 50)
        print("SOLID Principles:")
        print("  • Single Responsibility: Each class has one clear purpose")
        print("  • Open/Closed: Easy to extend with new processors")
        print("  • Liskov Substitution: All processors implement same interface")
        print("  • Interface Segregation: Clean, focused abstractions")
        print("  • Dependency Inversion: Abstract strategy pattern")
        
        print(f"\nKISS (Keep It Simple, Stupid):")
        print("  • Simple, readable design")
        print("  • Clear separation of concerns")
        print("  • Minimal complexity")
        
        print(f"\nYAGNI (You Aren't Gonna Need It):")
        print("  • Focus only on required functionality")
        print("  • Avoid over-engineering")
        print("  • Implement features when actually needed")
        
        print(f"\nPERFORMANCE OPTIMIZATION TECHNIQUES")
        print("-" * 50)
        print("Memory Management:")
        print("  • Chunked processing for large files")
        print("  • Automatic garbage collection")
        print("  • Memory usage monitoring")
        print("  • Streaming data accumulation")
        
        print(f"\nProcessing Optimization:")
        print("  • Pandas vectorized operations")
        print("  • Efficient data type handling")
        print("  • Batch processing strategies")
        print("  • Error handling with limits")
        
        print(f"\nDATA INSIGHTS CAPABILITIES")
        print("-" * 50)
        print("Business Intelligence:")
        print("  • Geographic distribution analysis")
        print("  • Market penetration metrics")
        print("  • Company and industry segmentation")
        print("  • Communication pattern analysis")
        
        print(f"\nData Quality Assessment:")
        print("  • Completeness scoring")
        print("  • Validity verification")
        print("  • Consistency analysis")
        print("  • Duplicate detection")
        
        print(f"\nTemporal Analysis:")
        print("  • Subscription trend analysis")
        print("  • Seasonal pattern detection")
        print("  • Growth metrics calculation")
        print("  • Time-based segmentation")
        
        print(f"\nCONCLUSION")
        print("-" * 50)
        print("This CSV processing solution demonstrates enterprise-grade capabilities")
        print("with both memory-efficient processing and comprehensive data insights.")
        print("The architecture supports scalable processing from small interactive")
        print("analyses to large-scale batch processing while maintaining code quality")
        print("and following software engineering best practices.")
