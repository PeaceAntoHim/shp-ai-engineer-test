
import asyncio
import logging
from datetime import datetime
from csv_runner_factory import CSVProcessorFactory
from config import settings
from models import CSVProcessingResult
from output_logger import OutputLogger, AnalysisReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CSVAnalysisRunner:
    """Main runner for CSV analysis with comprehensive reporting."""
    def __init__(self):
        self.output_logger = OutputLogger()
        self.report_generator = AnalysisReportGenerator(self.output_logger)

    async def analyze_small_csv(self) -> CSVProcessingResult | None:
        """Analyze small CSV file (customers-100000.csv)."""
        print("\n" + "="*80)
        print("QUESTION 1: SMALL CSV ANALYSIS (customers-100000.csv)")
        print("="*80)

        try:
            result = await CSVProcessorFactory.create_and_process(settings.small_csv_path)

            self._print_comprehensive_report(result, "SMALL FILE ANALYSIS")
            self._print_detailed_insights(result)

            return result

        except Exception as e:
            logger.error(f"Small CSV analysis failed: {e}")
            print(f"Error processing small CSV: {e}")
            return None

    async def analyze_large_csv(self) -> CSVProcessingResult | None:
        """Analyze large CSV file (customers-2000000.csv)."""
        print("\n" + "="*80)
        print("QUESTION 2: LARGE CSV ANALYSIS (customers-2000000.csv)")
        print("Memory-Efficient Processing")
        print("="*80)

        try:
            result = await CSVProcessorFactory.create_and_process(settings.large_csv_path)

            self._print_comprehensive_report(result, "LARGE FILE ANALYSIS")
            self._print_memory_efficiency_metrics(result)

            return result

        except Exception as e:
            logger.error(f"Large CSV analysis failed: {e}")
            print(f"Error processing large CSV: {e}")
            return None

    def _print_comprehensive_report(self, result, title: str):
        """Print comprehensive analysis report."""
        print(f"\n {title}")
        print("-" * 60)

        # Basic Processing Metrics
        print(f"PROCESSING SUMMARY")
        print(f"   File: {result.file_path}")
        print(f"   Total Rows: {result.total_rows:,}")
        print(f"   Successfully Processed: {result.processed_rows:,}")
        print(f"   Failed Records: {result.failed_rows:,}")
        print(f"   Success Rate: {result.get_success_rate():.2f}%")

        # Performance Metrics
        tech = result.technical_metrics
        print(f"\nPERFORMANCE METRICS")
        print(f"   Strategy: {tech.processing_strategy.value}")
        print(f"   Execution Time: {tech.execution_time_seconds:.2f} seconds")
        print(f"   Memory Usage: {tech.memory_usage_mb:.1f} MB")
        print(f"   Peak Memory: {tech.peak_memory_mb:.1f} MB")
        print(f"   CPU Usage: {tech.cpu_usage_percent:.1f}%")
        print(f"   Throughput: {tech.throughput_rows_per_second:,.0f} rows/second")

        if tech.chunks_processed > 0:
            print(f"   Chunks Processed: {tech.chunks_processed:,}")
            print(f"   Garbage Collections: {tech.garbage_collections}")

        # Data Quality Metrics
        quality = result.insights.data_quality
        print(f"\nDATA QUALITY ASSESSMENT")
        print(f"   Completeness Score: {quality.completeness_score:.1f}%")
        print(f"   Validity Score: {quality.validity_score:.1f}%")
        print(f"   Consistency Score: {quality.consistency_score:.1f}%")
        print(f"   Unique Customers: {quality.unique_customers:,}")
        print(f"   Duplicates Found: {quality.duplicate_count:,}")

        # Missing Values Analysis
        print(f"\nMISSING VALUES ANALYSIS")
        top_missing = sorted(quality.missing_values.items(), key=lambda x: x[1], reverse=True)[:5]
        for column, count in top_missing:
            if count > 0:
                percentage = (count / result.total_rows) * 100
                print(f"   {column}: {count:,} missing ({percentage:.1f}%)")

    def _print_detailed_insights(self, result):
        """Print detailed insights for small file analysis."""
        insights = result.insights

        # Geographic Analysis
        print(f"\nGEOGRAPHIC INSIGHTS")
        geo_dist = insights.business_insights.geographic_distribution
        print(f"   Countries Represented: {len(geo_dist)}")
        print(f"   Top Countries:")
        for i, (country, count) in enumerate(list(geo_dist.items())[:5], 1):
            percentage = (count / result.total_rows) * 100
            print(f"   {i}. {country}: {count:,} customers ({percentage:.1f}%)")

        # Market Penetration
        if insights.business_insights.market_penetration:
            print(f"\nMARKET PENETRATION")
            for country, penetration in list(insights.business_insights.market_penetration.items())[:3]:
                print(f"   {country}: {penetration:.2f}% market share")

        # Business Intelligence
        print(f"\nBUSINESS INTELLIGENCE")
        companies = insights.business_insights.company_distribution
        print(f"   Unique Companies: {len(companies)}")
        print(f"   Top Companies:")
        for i, (company, count) in enumerate(list(companies.items())[:3], 1):
            print(f"   {i}. {company}: {count:,} customers")

        # Industry Segments
        if insights.business_insights.industry_segments:
            print(f"\nINDUSTRY SEGMENTS")
            for segment, count in insights.business_insights.industry_segments.items():
                print(f"   {segment}: {count:,} companies")

        # Communication Analysis
        print(f"\nCOMMUNICATION ANALYSIS")
        email_domains = insights.communication_analysis.email_domains
        print(f"   Email Domains (Top 5):")
        for domain, count in list(email_domains.items())[:5]:
            print(f"   • {domain}: {count:,}")

        # Phone Analysis
        phone_patterns = insights.communication_analysis.phone_patterns
        if phone_patterns:
            print(f"\nPHONE ANALYSIS")
            for pattern, value in phone_patterns.items():
                if isinstance(value, (int, float)):
                    print(f"   {pattern.replace('_', ' ').title()}: {value:,}")

        # Temporal Analysis
        temporal = insights.temporal_analysis
        if temporal.subscription_by_year:
            print(f"\nTEMPORAL ANALYSIS")
            print(f"   Subscription Timeline:")
            for year, count in list(temporal.subscription_by_year.items())[:5]:
                print(f"   {year}: {count:,} subscriptions")

        if temporal.seasonal_patterns:
            print(f"   Seasonal Patterns:")
            for season, count in temporal.seasonal_patterns.items():
                print(f"   {season}: {count:,} subscriptions")

        # Data Structure Analysis
        columns_info = insights.columns_info
        print(f"\nDATA STRUCTURE")
        print(f"   Total Columns: {columns_info['total_columns']}")
        print(f"   Data Types:")
        for col, dtype in list(columns_info['data_types'].items())[:5]:
            print(f"   • {col}: {dtype}")

    def _print_memory_efficiency_metrics(self, result):
        """Print memory efficiency metrics for large file processing."""
        tech = result.technical_metrics
        insights = result.insights

        print(f"\nMEMORY EFFICIENCY ANALYSIS")
        print(f"   Memory Efficient Processing: {tech.memory_efficient}")
        print(f"   Constant Memory Usage: ~{tech.memory_usage_mb:.1f} MB")
        print(f"   Chunks Processed: {tech.chunks_processed:,}")
        print(f"   Garbage Collections: {tech.garbage_collections}")
        print(f"   Avg Memory per Chunk: {tech.memory_usage_mb/tech.chunks_processed:.1f} MB")

        # Streaming Analysis Results
        print(f"\nSTREAMING ANALYSIS RESULTS")
        quality = insights.data_quality
        print(f"   Total Records Analyzed: {quality.total_rows:,}")
        print(f"   Unique Customers: {quality.unique_customers:,}")
        print(f"   Data Completeness: {quality.completeness_score:.1f}%")

        # Top Geographic Markets (from streaming)
        geo_dist = insights.business_insights.geographic_distribution
        if geo_dist:
            print(f"\nTOP MARKETS (Streaming Analysis)")
            for i, (country, count) in enumerate(list(geo_dist.items())[:5], 1):
                print(f"   {i}. {country}: {count:,} customers")

        # Communication Patterns (from streaming)
        email_domains = insights.communication_analysis.email_domains
        if email_domains:
            print(f"\nEMAIL PATTERNS (Streaming Analysis)")
            for domain, count in list(email_domains.items())[:5]:
                print(f"   • {domain}: {count:,}")

        # Temporal Trends (from streaming)
        temporal = insights.temporal_analysis
        if temporal.subscription_by_year:
            print(f"\nGROWTH TRENDS (Streaming Analysis)")
            yearly_data = temporal.subscription_by_year
            for year, count in list(yearly_data.items())[:3]:
                print(f"   {year}: {count:,} new subscriptions")

    def print_comparison_summary(self, small_result, large_result):
        """Print comparison between small and large file processing."""
        if not small_result or not large_result:
            print("\nCannot compare - one or both analyses failed")
            return

        print("\n" + "="*80)
        print("PROCESSING COMPARISON: SMALL vs LARGE FILE")
        print("="*80)

        # Performance Comparison
        small_tech = small_result.technical_metrics
        large_tech = large_result.technical_metrics

        print(f"\nPERFORMANCE COMPARISON")
        print(f"{'Metric':<25} {'Small File':<20} {'Large File':<20} {'Difference':<15}")
        print("-" * 80)
        print(f"{'Rows Processed':<25} {small_tech.throughput_rows_per_second*small_tech.execution_time_seconds:<20,.0f} {large_tech.throughput_rows_per_second*large_tech.execution_time_seconds:<20,.0f} {large_tech.throughput_rows_per_second*large_tech.execution_time_seconds/small_tech.throughput_rows_per_second/small_tech.execution_time_seconds:<15.1f}x")
        print(f"{'Execution Time (s)':<25} {small_tech.execution_time_seconds:<20.2f} {large_tech.execution_time_seconds:<20.2f} {large_tech.execution_time_seconds/small_tech.execution_time_seconds:<15.1f}x")
        print(f"{'Memory Usage (MB)':<25} {small_tech.memory_usage_mb:<20.1f} {large_tech.memory_usage_mb:<20.1f} {large_tech.memory_usage_mb/small_tech.memory_usage_mb:<15.1f}x")
        print(f"{'Throughput (rows/s)':<25} {small_tech.throughput_rows_per_second:<20,.0f} {large_tech.throughput_rows_per_second:<20,.0f} {large_tech.throughput_rows_per_second/small_tech.throughput_rows_per_second:<15.1f}x")

        # Strategy Differences
        print(f"\nPROCESSING STRATEGY DIFFERENCES")
        print(f"Small File Strategy: {small_tech.processing_strategy.value}")
        print(f"• Full dataset loaded into memory")
        print(f"• Complete statistical analysis")
        print(f"• Exact calculations and insights")
        print(f"• Higher memory usage ({small_tech.memory_usage_mb:.1f} MB)")

        print(f"\nLarge File Strategy: {large_tech.processing_strategy.value}")
        print(f"• Chunked streaming processing")
        print(f"• Accumulated statistical analysis")
        print(f"• Approximate insights from streaming")
        print(f"• Constant memory usage (~{large_tech.memory_usage_mb:.1f} MB)")
        print(f"• {large_tech.chunks_processed} chunks processed")

        # When to Use Each
        print(f"\nWHEN TO USE EACH APPROACH")
        print(f"Small File Processor:")
        print(f"Files < {settings.small_file_threshold:,} rows")
        print(f"Need exact statistical analysis")
        print(f"Have sufficient RAM (2-3x file size)")
        print(f"Interactive data exploration")
        print(f"Real-time analytics dashboards")

        print(f"\nLarge File Processor:")
        print(f"Files > {settings.small_file_threshold:,} rows")
        print(f"Limited memory environments")
        print(f"Production batch processing")
        print(f"Streaming data pipelines")
        print(f"Server environments with memory constraints")

    async def run_with_logging(self):
        """Run complete analysis with automatic output logging."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"csv_analysis_output_{timestamp}.txt"
        
        # Capture all console output to file
        with self.output_logger.capture_output(log_filename) as log_path:
            print("Enhanced CSV Processing & Analysis Suite")
            print("Demonstrating SOLID, KISS, and YAGNI principles")
            print(f"Processing files:")
            print(f"• Small: {settings.small_csv_path}")
            print(f"• Large: {settings.large_csv_path}")
            print(f"• Output logging to: {log_path}")
            
            # Question 1: Small CSV Analysis
            small_result = await self.analyze_small_csv()
            
            # Question 2: Large CSV Analysis
            large_result = await self.analyze_large_csv()
            
            # Comparison Summary
            self.print_comparison_summary(small_result, large_result)
            
            print(f"\nAnalysis Complete - {datetime.now()}")
            
        # Generate additional structured report
        if small_result or large_result:
            report_path = self.report_generator.generate_complete_report(small_result, large_result)
            print(f"\nDetailed structured report: {report_path}")
            
        return {
            'small_result': small_result,
            'large_result': large_result,
            'console_log': log_path,
            'structured_report': report_path if small_result or large_result else None
        }


async def main():
    """Main execution function with automated logging."""
    runner = CSVAnalysisRunner()
    results = await runner.run_with_logging()
    
    print(f"\nANALYSIS COMPLETE!")
    print(f"Generated output files:")
    print(f"   • Console output: {results['console_log']}")
    if results['structured_report']:
        print(f"   • Structured report: {results['structured_report']}")


if __name__ == "__main__":
    asyncio.run(main())