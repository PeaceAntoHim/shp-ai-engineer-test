from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Enhanced application settings."""

    # File Processing
    chunk_size: int = 8192  # Optimized chunk size
    small_file_threshold: int = 150000  # Increased threshold
    max_memory_usage_mb: float = 512.0

    # Analysis Parameters
    insight_top_categories: int = 15
    max_errors_to_collect: int = 50

    # Performance Monitoring
    memory_check_interval: int = 1000  # Check memory every N rows
    gc_threshold_mb: float = 300.0     # Force GC at this memory usage

    # Data Quality Thresholds
    min_completeness_score: float = 0.8
    min_validity_score: float = 0.95

    # File Paths
    small_csv_path: str = "dataset/customers-100000.csv"
    large_csv_path: str = "dataset/customers-2000000.csv"

    class Config:
        env_prefix = "CSV_PROCESSOR_"


settings = Settings()