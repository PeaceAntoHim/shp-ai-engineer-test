from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional, List, Dict, Any
from enum import Enum
import json


class ProcessingStrategy(Enum):
    """Processing strategy types."""
    SMALL_FILE_FULL_LOAD = "small_file_full_load"
    LARGE_FILE_CHUNKED_STREAMING = "large_file_chunked_streaming"


@dataclass
class Customer:
    """Customer domain model with validation."""
    customer_id: str
    first_name: str
    last_name: str
    email: str
    company: str
    city: str
    country: str
    phone_1: str
    phone_2: Optional[str] = None
    subscription_date: Optional[date] = None
    website: Optional[str] = None

    def __post_init__(self):
        """Validate customer data after initialization."""
        if not self.customer_id or not self.customer_id.strip():
            raise ValueError("Customer ID cannot be empty")
        if not self.email or '@' not in self.email:
            raise ValueError(f"Invalid email format: {self.email}")


@dataclass
class DataQualityMetrics:
    """Data quality assessment metrics."""
    total_rows: int
    valid_rows: int
    invalid_rows: int
    duplicate_count: int
    unique_customers: int
    completeness_score: float  # Percentage of non-null values
    validity_score: float      # Percentage of valid records
    consistency_score: float   # Data format consistency
    missing_values: Dict[str, int] = field(default_factory=dict)
    invalid_emails: int = 0
    invalid_dates: int = 0
    invalid_phones: int = 0


@dataclass
class BusinessInsights:
    """Business intelligence metrics."""
    geographic_distribution: Dict[str, int] = field(default_factory=dict)
    company_distribution: Dict[str, int] = field(default_factory=dict)
    industry_segments: Dict[str, int] = field(default_factory=dict)
    subscription_trends: Dict[str, Any] = field(default_factory=dict)
    customer_value_segments: Dict[str, int] = field(default_factory=dict)
    market_penetration: Dict[str, float] = field(default_factory=dict)


@dataclass
class TechnicalMetrics:
    """Technical performance metrics."""
    processing_strategy: ProcessingStrategy
    execution_time_seconds: float
    memory_usage_mb: float
    peak_memory_mb: float
    cpu_usage_percent: float
    throughput_rows_per_second: float
    chunks_processed: int = 0
    garbage_collections: int = 0
    memory_efficient: bool = False


@dataclass
class CommunicationAnalysis:
    """Communication patterns and preferences."""
    email_domains: Dict[str, int] = field(default_factory=dict)
    phone_patterns: Dict[str, int] = field(default_factory=dict)
    website_domains: Dict[str, int] = field(default_factory=dict)
    communication_preferences: Dict[str, int] = field(default_factory=dict)
    contact_completeness: Dict[str, float] = field(default_factory=dict)


@dataclass
class TemporalAnalysis:
    """Time-based analysis and trends."""
    date_range: Dict[str, str] = field(default_factory=dict)
    subscription_by_year: Dict[int, int] = field(default_factory=dict)
    subscription_by_month: Dict[str, int] = field(default_factory=dict)
    seasonal_patterns: Dict[str, int] = field(default_factory=dict)
    growth_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ComprehensiveInsights:
    """Complete data insights package."""
    data_quality: DataQualityMetrics
    business_insights: BusinessInsights
    communication_analysis: CommunicationAnalysis
    temporal_analysis: TemporalAnalysis
    columns_info: Dict[str, Any] = field(default_factory=dict)
    statistical_summary: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Convert insights to JSON format."""
        return json.dumps(self, default=str, indent=2)


@dataclass
class CSVProcessingResult:
    """Enhanced processing result with separated metrics."""
    # Basic Processing Info
    total_rows: int
    processed_rows: int
    failed_rows: int
    errors: List[str] = field(default_factory=list)

    # Performance Metrics
    technical_metrics: TechnicalMetrics = None

    # Data Insights
    insights: ComprehensiveInsights = None

    # Processing Context
    file_path: str = ""
    processed_at: datetime = field(default_factory=datetime.now)

    def get_success_rate(self) -> float:
        """Calculate processing success rate."""
        return (self.processed_rows / self.total_rows * 100) if self.total_rows > 0 else 0.0

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for comparison."""
        return {
            "strategy": self.technical_metrics.processing_strategy.value,
            "execution_time": self.technical_metrics.execution_time_seconds,
            "memory_usage": self.technical_metrics.memory_usage_mb,
            "throughput": self.technical_metrics.throughput_rows_per_second,
            "success_rate": self.get_success_rate()
        }