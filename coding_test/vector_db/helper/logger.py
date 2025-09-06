import os
import sys
import time
import psutil
from datetime import datetime
from contextlib import contextmanager
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class VectorDBMetrics:
    """Performance and operational metrics for vector database"""
    
    # Processing Metrics
    operation_type: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    execution_time: float = 0.0
    
    # Database Metrics
    vectors_processed: int = 0
    vectors_added: int = 0
    vectors_searched: int = 0
    vectors_deleted: int = 0
    search_queries: int = 0
    
    # Performance Metrics
    memory_usage_mb: float = 0.0
    peak_memory_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # Vector Processing Metrics
    average_vector_dimension: float = 0.0
    total_similarity_calculations: int = 0
    average_similarity_calculation_time: float = 0.0
    
    # Search Metrics
    average_search_time: float = 0.0
    average_results_per_search: float = 0.0
    highest_similarity_score: float = 0.0
    lowest_similarity_score: float = 1.0
    
    # Database Connection Metrics
    database_connections: int = 0
    database_queries: int = 0
    database_errors: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'operation_type': self.operation_type,
            'execution_time': self.execution_time,
            'vectors_processed': self.vectors_processed,
            'vectors_added': self.vectors_added,
            'vectors_searched': self.vectors_searched,
            'vectors_deleted': self.vectors_deleted,
            'search_queries': self.search_queries,
            'memory_usage_mb': self.memory_usage_mb,
            'peak_memory_mb': self.peak_memory_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'average_vector_dimension': self.average_vector_dimension,
            'total_similarity_calculations': self.total_similarity_calculations,
            'average_similarity_calculation_time': self.average_similarity_calculation_time,
            'average_search_time': self.average_search_time,
            'average_results_per_search': self.average_results_per_search,
            'highest_similarity_score': self.highest_similarity_score,
            'lowest_similarity_score': self.lowest_similarity_score,
            'database_connections': self.database_connections,
            'database_queries': self.database_queries,
            'database_errors': self.database_errors
        }


class VectorDBLogger:
    """Comprehensive logging system for vector database operations"""
    
    def __init__(self, log_directory: str = "output_logs"):
        self.log_directory = log_directory
        self.metrics = VectorDBMetrics()
        self.operation_logs: List[Dict[str, Any]] = []
        self.ensure_log_directory()
        
        # Performance tracking
        self.process = psutil.Process()
        self.initial_memory = self._get_memory_usage_mb()
        self.peak_memory = self.initial_memory
        
    def ensure_log_directory(self):
        """Create log directory if it doesn't exist"""
        if not os.path.exists(self.log_directory):
            os.makedirs(self.log_directory)
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        try:
            memory_info = self.process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            return self.process.cpu_percent()
        except:
            return 0.0
    
    def start_operation(self, operation_type: str):
        """Start tracking an operation"""
        self.metrics.operation_type = operation_type
        self.metrics.start_time = time.time()
        
        # Update memory tracking
        current_memory = self._get_memory_usage_mb()
        self.metrics.memory_usage_mb = current_memory - self.initial_memory
        self.peak_memory = max(self.peak_memory, current_memory)
        self.metrics.peak_memory_mb = self.peak_memory - self.initial_memory
        
        print(f"Starting operation: {operation_type}")
        print(f"   Initial memory: {current_memory:.1f} MB")
    
    def end_operation(self, operation_type: str = ""):
        """End tracking current operation"""
        self.metrics.operation_type = operation_type
        self.metrics.end_time = time.time()
        self.metrics.execution_time = self.metrics.end_time - self.metrics.start_time
        self.metrics.cpu_usage_percent = self._get_cpu_usage()
        
        # Update final memory
        final_memory = self._get_memory_usage_mb()
        self.metrics.memory_usage_mb = final_memory - self.initial_memory
        
        print(f"Operation completed: {self.metrics.operation_type}")
        print(f"   Execution time: {self.metrics.execution_time:.4f} seconds")
        print(f"   Final memory: {final_memory:.1f} MB")
        print(f"   Memory change: +{self.metrics.memory_usage_mb:.1f} MB")
        
        # Log operation
        operation_log = {
            'timestamp': datetime.now().isoformat(),
            'operation': self.metrics.operation_type,
            'metrics': self.metrics.to_dict()
        }
        self.operation_logs.append(operation_log)
    
    def log_vector_addition(self, vector_id: int, dimension: int):
        """Log vector addition"""
        self.metrics.vectors_added += 1
        self.metrics.vectors_processed += 1
        
        # Update average dimension
        if self.metrics.average_vector_dimension == 0:
            self.metrics.average_vector_dimension = dimension
        else:
            total_vectors = self.metrics.vectors_processed
            self.metrics.average_vector_dimension = (
                (self.metrics.average_vector_dimension * (total_vectors - 1) + dimension) / total_vectors
            )
        
        print(f"   Added vector {vector_id} (dim: {dimension})")
    
    def log_search_operation(self, query_dimension: int, results_count: int, search_time: float, similarities: List[float]):
        """Log search operation"""
        self.metrics.search_queries += 1
        self.metrics.vectors_searched += results_count
        
        # Update search metrics
        if self.metrics.average_search_time == 0:
            self.metrics.average_search_time = search_time
        else:
            query_count = self.metrics.search_queries
            self.metrics.average_search_time = (
                (self.metrics.average_search_time * (query_count - 1) + search_time) / query_count
            )
        
        # Update results metrics
        if self.metrics.average_results_per_search == 0:
            self.metrics.average_results_per_search = results_count
        else:
            query_count = self.metrics.search_queries
            self.metrics.average_results_per_search = (
                (self.metrics.average_results_per_search * (query_count - 1) + results_count) / query_count
            )
        
        # Update similarity score metrics
        if similarities:
            max_sim = max(similarities)
            min_sim = min(similarities)
            
            self.metrics.highest_similarity_score = max(self.metrics.highest_similarity_score, max_sim)
            self.metrics.lowest_similarity_score = min(self.metrics.lowest_similarity_score, min_sim)
        
        print(f"   Search completed: {results_count} results in {search_time:.4f}s")
        if similarities:
            print(f"      Similarity range: {min(similarities):.4f} - {max(similarities):.4f}")
    
    def log_similarity_calculation(self, calculation_count: int, calculation_time: float):
        """Log similarity calculations"""
        self.metrics.total_similarity_calculations += calculation_count
        
        if self.metrics.average_similarity_calculation_time == 0:
            self.metrics.average_similarity_calculation_time = calculation_time
        else:
            total_calcs = self.metrics.total_similarity_calculations
            self.metrics.average_similarity_calculation_time = (
                (self.metrics.average_similarity_calculation_time * (total_calcs - calculation_count) + 
                 calculation_time) / total_calcs
            )
    
    def log_vector_deletion(self, vector_id: int, success: bool):
        """Log vector deletion"""
        if success:
            self.metrics.vectors_deleted += 1
            print(f"   Deleted vector {vector_id}")
        else:
            print(f"   Failed to delete vector {vector_id}")
    
    def log_database_operation(self, operation: str, success: bool):
        """Log database operations"""
        self.metrics.database_queries += 1
        
        if success:
            self.metrics.database_connections += 1
        else:
            self.metrics.database_errors += 1
            print(f"   Database operation failed: {operation}")
    
    @contextmanager
    def capture_output(self, filename: str = None):
        """Context manager to capture output to both console and file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"vector_db_analysis_{timestamp}.txt"
        
        log_path = os.path.join(self.log_directory, filename)
        original_stdout = sys.stdout
        
        try:
            with open(log_path, 'w', encoding='utf-8') as log_file:
                tee_writer = TeeWriter(original_stdout, log_file)
                sys.stdout = tee_writer
                
                self._write_log_header(log_path)
                yield log_path
                
        finally:
            sys.stdout = original_stdout
            print(f"\nVector DB analysis saved to: {log_path}")
    
    def _write_log_header(self, log_path: str):
        """Write header information to log"""
        print("=" * 100)
        print("VECTOR DATABASE ANALYSIS - COMPREHENSIVE OUTPUT LOG")
        print("=" * 100)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log File: {log_path}")
        print("=" * 100)
        print("Custom Vector Database with Cosine Similarity")
        print("PostgreSQL Backend with Dynamic Content Generation")
        print("=" * 100)
    
    def generate_comprehensive_report(self, filename: str = None):
        """Generate comprehensive analysis report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename}_report_{timestamp}.txt"
        
        with self.capture_output(filename) as log_path:
            self._print_executive_summary()
            self._print_performance_metrics()
            self._print_operation_analysis()
            self._print_database_analysis()
            self._print_similarity_analysis()
            self._print_technical_recommendations()
            
        return log_path
    
    def _print_executive_summary(self):
        """Print executive summary"""
        print(f"\nEXECUTIVE SUMMARY")
        print("=" * 80)
        print("This report provides comprehensive analysis of vector database operations")
        print("including performance metrics, similarity calculations, and database efficiency.")
        print("The analysis demonstrates custom cosine similarity implementation and")
        print("PostgreSQL-backed vector storage capabilities.")
        print()
    
    def _print_performance_metrics(self):
        """Print detailed performance metrics"""
        print(f"\nPERFORMANCE METRICS ANALYSIS")
        print("=" * 80)
        
        print(f"\nOVERALL EXECUTION METRICS")
        print("-" * 50)
        print(f"Total Execution Time: {self.metrics.execution_time:.4f} seconds")
        print(f"Memory Usage: {self.metrics.memory_usage_mb:.1f} MB")
        print(f"Peak Memory: {self.metrics.peak_memory_mb:.1f} MB")
        print(f"CPU Usage: {self.metrics.cpu_usage_percent:.1f}%")
        
        print(f"\nVECTOR PROCESSING METRICS")
        print("-" * 50)
        print(f"Total Vectors Processed: {self.metrics.vectors_processed:,}")
        print(f"Vectors Added: {self.metrics.vectors_added:,}")
        print(f"Vectors Searched: {self.metrics.vectors_searched:,}")
        print(f"Vectors Deleted: {self.metrics.vectors_deleted:,}")
        print(f"Average Vector Dimension: {self.metrics.average_vector_dimension:.1f}")
        
        # Calculate throughput
        if self.metrics.execution_time > 0:
            vector_throughput = self.metrics.vectors_processed / self.metrics.execution_time
            print(f"Vector Processing Throughput: {vector_throughput:.1f} vectors/second")
        
        print(f"\nSEARCH PERFORMANCE METRICS")
        print("-" * 50)
        print(f"Total Search Queries: {self.metrics.search_queries:,}")
        print(f"Average Search Time: {self.metrics.average_search_time:.4f} seconds")
        print(f"Average Results per Search: {self.metrics.average_results_per_search:.1f}")
        
        if self.metrics.search_queries > 0:
            search_throughput = self.metrics.search_queries / self.metrics.execution_time
            print(f"Search Query Throughput: {search_throughput:.1f} queries/second")
        
        print(f"\nSIMILARITY CALCULATION METRICS")
        print("-" * 50)
        print(f"Total Similarity Calculations: {self.metrics.total_similarity_calculations:,}")
        print(f"Average Calculation Time: {self.metrics.average_similarity_calculation_time:.6f} seconds")
        print(f"Highest Similarity Score: {self.metrics.highest_similarity_score:.4f}")
        print(f"Lowest Similarity Score: {self.metrics.lowest_similarity_score:.4f}")
        
        if self.metrics.total_similarity_calculations > 0 and self.metrics.execution_time > 0:
            calc_throughput = self.metrics.total_similarity_calculations / self.metrics.execution_time
            print(f"Similarity Calculation Throughput: {calc_throughput:.1f} calculations/second")
    
    def _print_operation_analysis(self):
        """Print operation analysis"""
        print(f"\nOPERATION ANALYSIS")
        print("=" * 80)
        
        if not self.operation_logs:
            print("No operations recorded")
            return
        
        print(f"Total Operations Logged: {len(self.operation_logs)}")
        print(f"\nOPERATION TIMELINE:")
        print("-" * 50)
        
        for i, op_log in enumerate(self.operation_logs, 1):
            timestamp = op_log['timestamp']
            operation = op_log['operation']
            metrics = op_log['metrics']
            
            print(f"{i:2}. {timestamp[:19]} | {operation}")
            print(f"    Execution: {metrics['execution_time']:.4f}s | "
                  f"Memory: {metrics['memory_usage_mb']:.1f}MB | "
                  f"Vectors: {metrics['vectors_processed']}")
    
    def _print_database_analysis(self):
        """Print database analysis"""
        print(f"\nDATABASE PERFORMANCE ANALYSIS")
        print("=" * 80)
        
        print(f"Database Connections: {self.metrics.database_connections}")
        print(f"Database Queries: {self.metrics.database_queries}")
        print(f"Database Errors: {self.metrics.database_errors}")
        
        if self.metrics.database_queries > 0:
            error_rate = (self.metrics.database_errors / self.metrics.database_queries) * 100
            success_rate = 100 - error_rate
            print(f"Database Success Rate: {success_rate:.1f}%")
            print(f"Database Error Rate: {error_rate:.1f}%")
        
        if self.metrics.execution_time > 0:
            query_rate = self.metrics.database_queries / self.metrics.execution_time
            print(f"Database Query Rate: {query_rate:.1f} queries/second")
    
    def _print_similarity_analysis(self):
        """Print similarity analysis"""
        print(f"\nSIMILARITY ANALYSIS")
        print("=" * 80)
        
        print(f"Similarity Algorithm: Custom Cosine Similarity Implementation")
        print(f"Total Calculations: {self.metrics.total_similarity_calculations:,}")
        print(f"Average Calculation Time: {self.metrics.average_similarity_calculation_time:.6f} seconds")
        
        print(f"\nSIMILARITY SCORE DISTRIBUTION")
        print("-" * 50)
        print(f"Highest Score: {self.metrics.highest_similarity_score:.4f}")
        print(f"Lowest Score: {self.metrics.lowest_similarity_score:.4f}")
        
        score_range = self.metrics.highest_similarity_score - self.metrics.lowest_similarity_score
        print(f"Score Range: {score_range:.4f}")
        
        if self.metrics.total_similarity_calculations > 0:
            avg_per_search = self.metrics.total_similarity_calculations / max(1, self.metrics.search_queries)
            print(f"Average Calculations per Search: {avg_per_search:.1f}")
    
    def _print_technical_recommendations(self):
        """Print technical recommendations"""
        print(f"\nTECHNICAL RECOMMENDATIONS")
        print("=" * 80)
        
        print(f"\nPERFORMANCE OPTIMIZATION")
        print("-" * 50)
        
        # Memory recommendations
        if self.metrics.peak_memory_mb > 500:
            print("High memory usage detected:")
            print("   • Consider implementing vector chunking")
            print("   • Use streaming for large datasets")
            print("   • Implement memory cleanup routines")
        else:
            print("Memory usage within acceptable limits")
        
        # Search performance recommendations
        if self.metrics.average_search_time > 1.0:
            print("Slow search performance detected:")
            print("   • Consider adding database indexes")
            print("   • Implement approximate similarity search")
            print("   • Cache frequently accessed vectors")
        else:
            print("Search performance is good")

        # Similarity calculation recommendations
        if self.metrics.average_similarity_calculation_time > 0.001:
            print("Similarity calculations could be optimized:")
            print("   • Consider vectorized operations")
            print("   • Implement parallel processing")
            print("   • Use optimized mathematical libraries")
        else:
            print("Similarity calculations are efficient")

class TeeWriter:
    """Writer that outputs to multiple streams simultaneously"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str):
        """Write text to all streams"""
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        """Flush all streams"""
        for stream in self.streams:
            stream.flush()