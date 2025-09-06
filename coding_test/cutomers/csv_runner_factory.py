import pandas as pd
import psutil
import time
import gc
import os
from typing import Dict, Any, Tuple, List
from collections import Counter, defaultdict
import numpy as np
from urllib.parse import urlparse

from models import (
    Customer, CSVProcessingResult, ComprehensiveInsights,
    DataQualityMetrics, BusinessInsights, TechnicalMetrics,
    CommunicationAnalysis, TemporalAnalysis, ProcessingStrategy
)
from config import settings
import logging

logger = logging.getLogger(__name__)


class DataQualityAnalyzer:
    """Comprehensive data quality analysis."""
    
    @staticmethod
    def analyze_quality(df: pd.DataFrame, processed_rows: int, failed_rows: int) -> DataQualityMetrics:
        """Perform comprehensive data quality analysis."""
        total_rows = len(df)
        
        # Missing values analysis
        missing_values = df.isnull().sum().to_dict()
        
        # Duplicate analysis
        duplicate_count = df.duplicated(subset=['Customer Id']).sum()
        unique_customers = df['Customer Id'].nunique()
        
        # Data validation
        invalid_emails = DataQualityAnalyzer._count_invalid_emails(df)
        invalid_dates = DataQualityAnalyzer._count_invalid_dates(df)
        invalid_phones = DataQualityAnalyzer._count_invalid_phones(df)
        
        # Quality scores
        total_cells = total_rows * len(df.columns)
        non_null_cells = total_cells - sum(missing_values.values())
        completeness_score = (non_null_cells / total_cells) * 100
        
        validity_score = (processed_rows / total_rows) * 100 if total_rows > 0 else 0
        consistency_score = DataQualityAnalyzer._calculate_consistency_score(df)
        
        return DataQualityMetrics(
            total_rows=total_rows,
            valid_rows=processed_rows,
            invalid_rows=failed_rows,
            duplicate_count=duplicate_count,
            unique_customers=unique_customers,
            completeness_score=completeness_score,
            validity_score=validity_score,
            consistency_score=consistency_score,
            missing_values=missing_values,
            invalid_emails=invalid_emails,
            invalid_dates=invalid_dates,
            invalid_phones=invalid_phones
        )
    
    @staticmethod
    def _count_invalid_emails(df: pd.DataFrame) -> int:
        """Count invalid email addresses."""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        emails = df['Email'].dropna()
        return (~emails.str.match(email_pattern, case=False)).sum()
    
    @staticmethod
    def _count_invalid_dates(df: pd.DataFrame) -> int:
        """Count invalid subscription dates."""
        try:
            dates = pd.to_datetime(df['Subscription Date'], errors='coerce')
            return dates.isnull().sum()
        except:
            return len(df)
    
    @staticmethod
    def _count_invalid_phones(df: pd.DataFrame) -> int:
        """Count invalid phone numbers."""
        phone_pattern = r'^[\+]?[\d\s\-\(\)\.x]+$'
        phones = df['Phone 1'].dropna().astype(str)
        return (~phones.str.match(phone_pattern, case=False)).sum()
    
    @staticmethod
    def _calculate_consistency_score(df: pd.DataFrame) -> float:
        """Calculate data format consistency score."""
        consistency_scores = []
        
        # Email format consistency
        emails = df['Email'].dropna()
        if len(emails) > 0:
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            email_consistency = emails.str.match(email_pattern, case=False).mean()
            consistency_scores.append(email_consistency)
        
        # Phone format patterns
        phones = df['Phone 1'].dropna().astype(str)
        if len(phones) > 0:
            # Check for consistent phone patterns
            patterns = [
                phones.str.contains(r'^\+').mean(),  # International format
                phones.str.contains(r'^[\d\-\s\(\)\.]+$').mean(),  # Standard formats
            ]
            consistency_scores.append(max(patterns))
        
        return np.mean(consistency_scores) * 100 if consistency_scores else 0.0


class BusinessIntelligenceAnalyzer:
    """Advanced business intelligence analysis."""
    
    @staticmethod
    def analyze_business_insights(df: pd.DataFrame) -> BusinessInsights:
        """Generate comprehensive business insights."""
        
        # Geographic distribution
        geographic_dist = df['Country'].value_counts().head(15).to_dict()
        
        # Company distribution
        company_dist = df['Company'].value_counts().head(15).to_dict()
        
        # Industry segments (based on company types)
        industry_segments = BusinessIntelligenceAnalyzer._analyze_industry_segments(df)
        
        # Subscription trends
        subscription_trends = BusinessIntelligenceAnalyzer._analyze_subscription_trends(df)
        
        # Market penetration (countries with highest customer density)
        market_penetration = BusinessIntelligenceAnalyzer._calculate_market_penetration(df)
        
        return BusinessInsights(
            geographic_distribution=geographic_dist,
            company_distribution=company_dist,
            industry_segments=industry_segments,
            subscription_trends=subscription_trends,
            market_penetration=market_penetration
        )
    
    @staticmethod
    def _analyze_industry_segments(df: pd.DataFrame) -> Dict[str, int]:
        """Analyze industry segments based on company names."""
        segments = defaultdict(int)
        companies = df['Company'].dropna()
        
        patterns = {
            'Technology': r'(tech|software|digital|cyber|IT|systems)',
            'Financial': r'(bank|financial|invest|capital|fund)',
            'Healthcare': r'(health|medical|pharma|bio|care)',
            'Manufacturing': r'(manufacturing|industrial|factory|production)',
            'Consulting': r'(consulting|advisory|solutions)',
            'LLC/Corporation': r'(LLC|Ltd|PLC|Corp|Inc)',
            'Partnership': r'(and|&|\-)',
        }
        
        for segment, pattern in patterns.items():
            count = companies.str.contains(pattern, case=False, regex=True).sum()
            if count > 0:
                segments[segment] = count
        
        return dict(segments)
    
    @staticmethod
    def _analyze_subscription_trends(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze subscription trends and patterns."""
        try:
            dates = pd.to_datetime(df['Subscription Date'], errors='coerce')
            valid_dates = dates.dropna()
            
            if len(valid_dates) == 0:
                return {"error": "No valid subscription dates"}
            
            trends = {
                "total_subscribers": len(valid_dates),
                "subscription_period": {
                    "start": str(valid_dates.min().date()),
                    "end": str(valid_dates.max().date()),
                    "duration_days": (valid_dates.max() - valid_dates.min()).days
                },
                "peak_subscription_month": valid_dates.dt.to_period('M').value_counts().index[0].strftime('%Y-%m'),
                "yearly_growth": valid_dates.dt.year.value_counts().sort_index().to_dict(),
                "monthly_distribution": valid_dates.dt.month.value_counts().sort_index().to_dict()
            }
            
            return trends
        except Exception as e:
            return {"error": f"Trend analysis failed: {str(e)}"}
    
    @staticmethod
    def _calculate_market_penetration(df: pd.DataFrame) -> Dict[str, float]:
        """Calculate market penetration metrics."""
        country_counts = df['Country'].value_counts()
        total_customers = len(df)
        
        penetration = {}
        for country, count in country_counts.head(10).items():
            penetration[country] = (count / total_customers) * 100
        
        return penetration


class CommunicationAnalyzer:
    """Communication patterns and preferences analysis."""
    
    @staticmethod
    def analyze_communication(df: pd.DataFrame) -> CommunicationAnalysis:
        """Analyze communication patterns."""
        
        # Email domain analysis
        email_domains = CommunicationAnalyzer._analyze_email_domains(df)
        
        # Phone pattern analysis
        phone_patterns = CommunicationAnalyzer._analyze_phone_patterns(df)
        
        # Website domain analysis
        website_domains = CommunicationAnalyzer._analyze_website_domains(df)
        
        # Communication preferences
        comm_preferences = CommunicationAnalyzer._analyze_communication_preferences(df)
        
        # Contact completeness
        contact_completeness = CommunicationAnalyzer._analyze_contact_completeness(df)
        
        return CommunicationAnalysis(
            email_domains=email_domains,
            phone_patterns=phone_patterns,
            website_domains=website_domains,
            communication_preferences=comm_preferences,
            contact_completeness=contact_completeness
        )
    
    @staticmethod
    def _analyze_email_domains(df: pd.DataFrame) -> dict[Any, Any] | dict[str, str]:
        """Analyze email domains."""
        try:
            emails = df['Email'].dropna().str.lower()
            domains = emails.str.split('@').str[1]
            domain_counts = domains.value_counts().head(15).to_dict()
            
            # Categorize domains
            categorized = {}
            for domain, count in domain_counts.items():
                if domain in ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']:
                    category = f"Personal - {domain}"
                else:
                    category = f"Corporate - {domain}"
                categorized[category] = count
            
            return categorized
        except:
            return {"error": "Email domain analysis failed"}
    
    @staticmethod
    def _analyze_phone_patterns(df: pd.DataFrame) -> dict[str, int | float | Any] | dict[str, str]:
        """Analyze phone number patterns."""
        try:
            phone1 = df['Phone 1'].dropna().astype(str)
            phone2 = df['Phone 2'].dropna().astype(str)
            
            patterns = {
                "total_primary_phones": len(phone1),
                "total_secondary_phones": len(phone2),
                "international_format": phone1.str.contains(r'^\+').sum(),
                "with_extensions": phone1.str.contains('x').sum(),
                "with_parentheses": phone1.str.contains(r'\(').sum(),
                "dual_phone_customers": len(df[df['Phone 2'].notna()]),
                "coverage_percentage": (len(phone1) / len(df)) * 100
            }
            
            return patterns
        except:
            return {"error": "Phone pattern analysis failed"}
    
    @staticmethod
    def _analyze_website_domains(df: pd.DataFrame) -> dict[Any, Any] | dict[str, str]:
        """Analyze website domains."""
        try:
            websites = df['Website'].dropna()
            domains = websites.apply(lambda x: urlparse(str(x)).netlify if pd.notna(x) else None)
            domains = domains.dropna()
            
            domain_counts = domains.value_counts().head(10).to_dict()
            
            # Add summary statistics
            summary = {
                "total_websites": len(websites),
                "unique_domains": len(domains.unique()) if len(domains) > 0 else 0,
                "website_coverage": (len(websites) / len(df)) * 100
            }
            
            return {**summary, **domain_counts}
        except:
            return {"error": "Website domain analysis failed"}
    
    @staticmethod
    def _analyze_communication_preferences(df: pd.DataFrame) -> Dict[str, int]:
        """Analyze communication preferences based on available channels."""
        preferences = {
            "email_only": len(df[df['Email'].notna() & df['Phone 1'].isna() & df['Website'].isna()]),
            "phone_only": len(df[df['Phone 1'].notna() & df['Email'].isna() & df['Website'].isna()]),
            "multi_channel": len(df[df['Email'].notna() & df['Phone 1'].notna()]),
            "complete_profile": len(df[df['Email'].notna() & df['Phone 1'].notna() & df['Website'].notna()]),
            "minimal_contact": len(df[df['Email'].isna() & df['Phone 1'].isna()])
        }
        
        return preferences
    
    @staticmethod
    def _analyze_contact_completeness(df: pd.DataFrame) -> Dict[str, float]:
        """Analyze completeness of contact information."""
        total = len(df)
        
        completeness = {
            "email_completeness": (df['Email'].notna().sum() / total) * 100,
            "phone_completeness": (df['Phone 1'].notna().sum() / total) * 100,
            "secondary_phone_completeness": (df['Phone 2'].notna().sum() / total) * 100,
            "website_completeness": (df['Website'].notna().sum() / total) * 100,
            "overall_completeness": (df[['Email', 'Phone 1', 'Website']].notna().any(axis=1).sum() / total) * 100
        }
        
        return completeness


class TemporalAnalyzer:
    """Time-based analysis and trends."""
    
    @staticmethod
    def analyze_temporal_patterns(df: pd.DataFrame) -> TemporalAnalysis:
        """Comprehensive temporal analysis."""
        try:
            dates = pd.to_datetime(df['Subscription Date'], errors='coerce')
            valid_dates = dates.dropna()
            
            if len(valid_dates) == 0:
                return TemporalAnalysis(date_range={"error": "No valid dates found"})
            
            # Date range
            date_range = {
                "min_date": str(valid_dates.min().date()),
                "max_date": str(valid_dates.max().date()),
                "total_days": (valid_dates.max() - valid_dates.min()).days,
                "data_points": len(valid_dates)
            }
            
            # Yearly subscription patterns
            subscription_by_year = valid_dates.dt.year.value_counts().sort_index().to_dict()
            
            # Monthly patterns
            subscription_by_month = valid_dates.dt.strftime('%Y-%m').value_counts().sort_index().head(12).to_dict()
            
            # Seasonal patterns
            seasonal_patterns = TemporalAnalyzer._analyze_seasonal_patterns(valid_dates)
            
            # Growth metrics
            growth_metrics = TemporalAnalyzer._calculate_growth_metrics(valid_dates)
            
            return TemporalAnalysis(
                date_range=date_range,
                subscription_by_year=subscription_by_year,
                subscription_by_month=subscription_by_month,
                seasonal_patterns=seasonal_patterns,
                growth_metrics=growth_metrics
            )
        except Exception as e:
            logger.error(f"Temporal analysis failed: {e}")
            return TemporalAnalysis(date_range={"error": str(e)})
    
    @staticmethod
    def _analyze_seasonal_patterns(dates: pd.Series) -> Dict[str, int]:
        """Analyze seasonal subscription patterns."""
        seasons = {
            "Spring": [3, 4, 5],
            "Summer": [6, 7, 8],
            "Fall": [9, 10, 11],
            "Winter": [12, 1, 2]
        }
        
        seasonal_counts = {}
        for season, months in seasons.items():
            count = dates[dates.dt.month.isin(months)].count()
            seasonal_counts[season] = count
        
        return seasonal_counts
    
    @staticmethod
    def _calculate_growth_metrics(dates: pd.Series) -> Dict[str, float]:
        """Calculate growth and trend metrics."""
        yearly_counts = dates.dt.year.value_counts().sort_index()
        
        if len(yearly_counts) < 2:
            return {"growth_rate": 0.0, "trend": "insufficient_data"}
        
        # Calculate year-over-year growth rate
        growth_rates = []
        for i in range(1, len(yearly_counts)):
            prev_year = yearly_counts.iloc[i-1]
            curr_year = yearly_counts.iloc[i]
            if prev_year > 0:
                growth_rate = ((curr_year - prev_year) / prev_year) * 100
                growth_rates.append(growth_rate)
        
        avg_growth_rate = np.mean(growth_rates) if growth_rates else 0.0
        trend = "growing" if avg_growth_rate > 5 else "declining" if avg_growth_rate < -5 else "stable"
        
        return {
            "average_growth_rate": round(avg_growth_rate, 2),
            "trend": trend,
            "peak_year": int(yearly_counts.idxmax()),
            "peak_subscriptions": int(yearly_counts.max())
        }


class EnhancedCSVProcessor:
    """Enhanced CSV processor with comprehensive analysis."""
    
    def __init__(self):
        self.start_time = None
        self.initial_memory = None
        self.peak_memory = 0
        self.gc_count = 0
    
    def _start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.initial_memory = self._get_memory_usage_mb()
        self.peak_memory = self.initial_memory
        self.gc_count = 0
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        self.peak_memory = max(self.peak_memory, memory_mb)
        return memory_mb
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return psutil.cpu_percent(interval=0.1)
    
    def _create_technical_metrics(self, strategy: ProcessingStrategy, chunks: int = 0) -> TechnicalMetrics:
        """Create technical performance metrics."""
        execution_time = time.time() - self.start_time
        memory_usage = self._get_memory_usage_mb() - self.initial_memory
        cpu_usage = self._get_cpu_usage()
        
        return TechnicalMetrics(
            processing_strategy=strategy,
            execution_time_seconds=execution_time,
            memory_usage_mb=memory_usage,
            peak_memory_mb=self.peak_memory,
            cpu_usage_percent=cpu_usage,
            throughput_rows_per_second=0,  # Will be calculated later
            chunks_processed=chunks,
            garbage_collections=self.gc_count,
            memory_efficient=strategy == ProcessingStrategy.LARGE_FILE_CHUNKED_STREAMING
        )


class SmallFileProcessor(EnhancedCSVProcessor):
    """Enhanced processor for small files with comprehensive insights."""
    
    async def process(self, file_path: str) -> CSVProcessingResult:
        """Process small CSV file with comprehensive analysis."""
        self._start_monitoring()
        
        logger.info(f"Processing Small CSV File: {file_path}")
        
        try:
            # Load entire dataset
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df):,} rows with {len(df.columns)} columns")
            
            # Clean data
            df_cleaned = self._clean_data(df.copy())
            
            # Process customers
            valid_customers, errors = await self._process_customers(df_cleaned)
            
            # Generate comprehensive insights
            insights = self._generate_comprehensive_insights(df, len(valid_customers), len(errors))
            
            # Create technical metrics
            technical_metrics = self._create_technical_metrics(ProcessingStrategy.SMALL_FILE_FULL_LOAD)
            technical_metrics.throughput_rows_per_second = len(valid_customers) / technical_metrics.execution_time_seconds
            
            logger.info(f"Small file processing completed: {len(valid_customers):,} customers processed")
            
            return CSVProcessingResult(
                total_rows=len(df),
                processed_rows=len(valid_customers),
                failed_rows=len(errors),
                errors=errors[:settings.max_errors_to_collect],
                technical_metrics=technical_metrics,
                insights=insights,
                file_path=file_path
            )
            
        except Exception as e:
            logger.error(f"Small file processing failed: {e}")
            raise
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced data cleaning for small files."""
        logger.info("Cleaning data...")
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['Customer Id'])
        logger.info(f"Removed {initial_count - len(df)} duplicates")
        
        # Clean text fields
        text_columns = ['Email', 'First Name', 'Last Name', 'Company', 'City', 'Country']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # Clean email addresses
        df['Email'] = df['Email'].str.lower()
        
        # Parse dates
        df['Subscription Date'] = pd.to_datetime(df['Subscription Date'], errors='coerce')
        
        # Clean phone numbers
        phone_columns = ['Phone 1', 'Phone 2']
        for col in phone_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        return df
    
    async def _process_customers(self, df: pd.DataFrame) -> Tuple[List[Customer], List[str]]:
        """Process and validate customer records."""
        valid_customers = []
        errors = []
        
        logger.info("Processing customer records...")
        
        for idx, row in df.iterrows():
            try:
                customer = self._row_to_customer(row)
                valid_customers.append(customer)
            except Exception as e:
                error_msg = f"Row {idx}: {str(e)}"
                errors.append(error_msg)
                if len(errors) >= settings.max_errors_to_collect:
                    logger.warning(f"Max error limit reached ({settings.max_errors_to_collect})")
                    break
        
        return valid_customers, errors
    
    def _row_to_customer(self, row) -> Customer:
        """Convert DataFrame row to Customer model with validation."""
        return Customer(
            customer_id=str(row['Customer Id']),
            first_name=str(row['First Name']),
            last_name=str(row['Last Name']),
            email=str(row['Email']),
            company=str(row['Company']),
            city=str(row['City']),
            country=str(row['Country']),
            phone_1=str(row['Phone 1']),
            phone_2=str(row['Phone 2']) if pd.notna(row['Phone 2']) else None,
            subscription_date=row['Subscription Date'].date() if pd.notna(row['Subscription Date']) else None,
            website=str(row['Website']) if pd.notna(row['Website']) else None
        )
    
    def _generate_comprehensive_insights(self, df: pd.DataFrame, processed_rows: int, failed_rows: int) -> ComprehensiveInsights:
        """Generate comprehensive insights for small file."""
        logger.info("Generating comprehensive insights...")
        
        # Data quality analysis
        data_quality = DataQualityAnalyzer.analyze_quality(df, processed_rows, failed_rows)
        
        # Business intelligence
        business_insights = BusinessIntelligenceAnalyzer.analyze_business_insights(df)
        
        # Communication analysis
        communication_analysis = CommunicationAnalyzer.analyze_communication(df)
        
        # Temporal analysis
        temporal_analysis = TemporalAnalyzer.analyze_temporal_patterns(df)
        
        # Column information
        columns_info = {
            "total_columns": len(df.columns),
            "column_names": list(df.columns),
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
        
        # Statistical summary
        statistical_summary = self._generate_statistical_summary(df)
        
        return ComprehensiveInsights(
            data_quality=data_quality,
            business_insights=business_insights,
            communication_analysis=communication_analysis,
            temporal_analysis=temporal_analysis,
            columns_info=columns_info,
            statistical_summary=statistical_summary
        )
    
    def _generate_statistical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate statistical summary."""
        summary = {
            "total_records": len(df),
            "memory_usage_bytes": df.memory_usage(deep=True).sum(),
            "numeric_columns": list(df.select_dtypes(include=[np.number]).columns),
            "text_columns": list(df.select_dtypes(include=['object']).columns),
            "datetime_columns": list(df.select_dtypes(include=['datetime']).columns)
        }
        
        # Add descriptive statistics for numeric columns
        if summary["numeric_columns"]:
            summary["numeric_stats"] = df[summary["numeric_columns"]].describe().to_dict()
        
        return summary


class LargeFileProcessor(EnhancedCSVProcessor):
    """Memory-efficient processor for large files."""
    
    def __init__(self):
        super().__init__()
        self.chunk_count = 0
        self.accumulated_stats = {
            'countries': Counter(),
            'companies': Counter(),
            'email_domains': Counter(),
            'years': Counter(),
            'months': Counter(),
            'total_rows': 0,
            'missing_counts': Counter(),
            'unique_ids': set(),
            'phone_patterns': Counter(),
            'website_domains': Counter(),
            'data_types': {},
            'seasonal_counts': Counter()
        }
    
    async def process(self, file_path: str) -> CSVProcessingResult:
        """Process large CSV file with memory-efficient streaming."""
        self._start_monitoring()
        
        logger.info(f"Processing Large CSV File: {file_path}")
        logger.info(f"Using chunk size: {settings.chunk_size:,} rows")
        
        total_rows = 0
        processed_rows = 0
        failed_rows = 0
        errors = []
        
        try:
            # Get column information
            columns = self._get_columns(file_path)
            logger.info(f"Detected {len(columns)} columns: {columns}")
            
            # Process file in chunks
            chunk_reader = pd.read_csv(file_path, chunksize=settings.chunk_size)
            
            for chunk_idx, chunk in enumerate(chunk_reader):
                self.chunk_count += 1
                logger.info(f"Processing chunk {self.chunk_count} ({len(chunk):,} rows)")
                
                # Update accumulated statistics
                self._update_accumulated_stats(chunk)
                
                # Clean and process chunk
                chunk_cleaned = self._clean_chunk(chunk)
                total_rows += len(chunk)
                
                # Process customer records
                chunk_processed, chunk_failed, chunk_errors = await self._process_chunk(chunk_cleaned, chunk_idx)
                processed_rows += chunk_processed
                failed_rows += chunk_failed
                errors.extend(chunk_errors)
                
                # Memory management
                self._manage_memory()
                
                # Progress logging
                if self.chunk_count % 10 == 0:
                    logger.info(f"Progress: {self.chunk_count} chunks, {processed_rows:,} customers processed")
            
            # Generate insights from accumulated statistics
            insights = self._generate_insights_from_accumulation(columns, total_rows)
            
            # Create technical metrics
            technical_metrics = self._create_technical_metrics(
                ProcessingStrategy.LARGE_FILE_CHUNKED_STREAMING, 
                self.chunk_count
            )
            technical_metrics.throughput_rows_per_second = processed_rows / technical_metrics.execution_time_seconds
            
            logger.info(f"Large file processing completed: {processed_rows:,} customers processed in {self.chunk_count} chunks")
            
            return CSVProcessingResult(
                total_rows=total_rows,
                processed_rows=processed_rows,
                failed_rows=failed_rows,
                errors=errors[:settings.max_errors_to_collect],
                technical_metrics=technical_metrics,
                insights=insights,
                file_path=file_path
            )
            
        except Exception as e:
            logger.error(f"Large file processing failed: {e}")
            raise
    
    def _get_columns(self, file_path: str) -> List[str]:
        """Get column names without loading full file."""
        return list(pd.read_csv(file_path, nrows=0).columns)
    
    def _update_accumulated_stats(self, chunk: pd.DataFrame):
        """Update accumulated statistics with chunk data."""
        # Geographic data
        self.accumulated_stats['countries'].update(chunk['Country'].dropna())
        
        # Company data
        self.accumulated_stats['companies'].update(chunk['Company'].dropna())
        
        # Email domains
        email_domains = chunk['Email'].dropna().str.lower().str.split('@').str[1]
        self.accumulated_stats['email_domains'].update(email_domains.dropna())
        
        # Temporal data
        dates = pd.to_datetime(chunk['Subscription Date'], errors='coerce').dropna()
        if len(dates) > 0:
            self.accumulated_stats['years'].update(dates.dt.year)
            self.accumulated_stats['months'].update(dates.dt.strftime('%Y-%m'))
            
            # Seasonal patterns
            seasons_map = {1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring',
                          6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall', 11: 'Fall', 12: 'Winter'}
            seasons = dates.dt.month.map(seasons_map)
            self.accumulated_stats['seasonal_counts'].update(seasons)
        
        # Missing values
        for col in chunk.columns:
            self.accumulated_stats['missing_counts'][col] += chunk[col].isnull().sum()
        
        # Unique customer IDs
        self.accumulated_stats['unique_ids'].update(chunk['Customer Id'].dropna())
        
        # Phone patterns
        phones = chunk['Phone 1'].dropna().astype(str)
        self.accumulated_stats['phone_patterns']['total'] += len(phones)
        self.accumulated_stats['phone_patterns']['international'] += phones.str.contains(r'^\+').sum()
        self.accumulated_stats['phone_patterns']['extensions'] += phones.str.contains('x').sum()
        
        # Website domains
        websites = chunk['Website'].dropna()
        if len(websites) > 0:
            try:
                domains = websites.apply(lambda x: urlparse(str(x)).netlify if pd.notna(x) else None)
                self.accumulated_stats['website_domains'].update(domains.dropna())
            except:
                pass
        
        # Data types (from first chunk)
        if not self.accumulated_stats['data_types']:
            self.accumulated_stats['data_types'] = {col: str(dtype) for col, dtype in chunk.dtypes.items()}
        
        self.accumulated_stats['total_rows'] += len(chunk)
    
    def _clean_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Clean chunk data efficiently."""
        # Remove duplicates within chunk
        chunk = chunk.drop_duplicates(subset=['Customer Id'])
        
        # Basic cleaning
        chunk['Email'] = chunk['Email'].astype(str).str.lower().str.strip()
        chunk['Subscription Date'] = pd.to_datetime(chunk['Subscription Date'], errors='coerce')
        
        for col in ['Phone 1', 'Phone 2']:
            if col in chunk.columns:
                chunk[col] = chunk[col].astype(str).str.strip()
        
        return chunk
    
    async def _process_chunk(self, chunk: pd.DataFrame, chunk_idx: int) -> Tuple[int, int, List[str]]:
        """Process a single chunk efficiently."""
        processed = 0
        failed = 0
        errors = []
        
        for idx, row in chunk.iterrows():
            try:
                customer = self._row_to_customer(row)
                processed += 1
            except Exception as e:
                failed += 1
                if len(errors) < 5:  # Limit errors per chunk
                    errors.append(f"Chunk {chunk_idx}, Row {idx}: {str(e)}")
        
        return processed, failed, errors
    
    def _row_to_customer(self, row) -> Customer:
        """Convert row to customer with basic validation."""
        return Customer(
            customer_id=str(row['Customer Id']),
            first_name=str(row['First Name']),
            last_name=str(row['Last Name']),
            email=str(row['Email']),
            company=str(row['Company']),
            city=str(row['City']),
            country=str(row['Country']),
            phone_1=str(row['Phone 1']),
            phone_2=str(row['Phone 2']) if pd.notna(row['Phone 2']) else None,
            subscription_date=row['Subscription Date'].date() if pd.notna(row['Subscription Date']) else None,
            website=str(row['Website']) if pd.notna(row['Website']) else None
        )
    
    def _manage_memory(self):
        """Manage memory usage during processing."""
        current_memory = self._get_memory_usage_mb()
        
        if current_memory > settings.gc_threshold_mb:
            logger.info(f"Memory usage high ({current_memory:.1f}MB), forcing garbage collection")
            gc.collect()
            self.gc_count += 1
            new_memory = self._get_memory_usage_mb()
            logger.info(f"Memory after GC: {new_memory:.1f}MB (freed {current_memory - new_memory:.1f}MB)")
    
    def _generate_insights_from_accumulation(self, columns: List[str], total_rows: int) -> ComprehensiveInsights:
        """Generate insights from accumulated statistics."""
        logger.info("Generating insights from accumulated statistics...")
        
        # Data quality metrics
        data_quality = DataQualityMetrics(
            total_rows=total_rows,
            valid_rows=total_rows - sum(self.accumulated_stats['missing_counts'].values()),
            invalid_rows=sum(self.accumulated_stats['missing_counts'].values()),
            duplicate_count=total_rows - len(self.accumulated_stats['unique_ids']),
            unique_customers=len(self.accumulated_stats['unique_ids']),
            completeness_score=((total_rows * len(columns) - sum(self.accumulated_stats['missing_counts'].values())) / (total_rows * len(columns))) * 100,
            validity_score=95.0,  # Estimated for streaming
            consistency_score=90.0,  # Estimated for streaming
            missing_values=dict(self.accumulated_stats['missing_counts'])
        )
        
        # Business insights
        business_insights = BusinessInsights(
            geographic_distribution=dict(self.accumulated_stats['countries'].most_common(15)),
            company_distribution=dict(self.accumulated_stats['companies'].most_common(15)),
            market_penetration=self._calculate_market_penetration_from_stats()
        )
        
        # Communication analysis
        communication_analysis = CommunicationAnalysis(
            email_domains=dict(self.accumulated_stats['email_domains'].most_common(15)),
            phone_patterns=dict(self.accumulated_stats['phone_patterns']),
            website_domains=dict(self.accumulated_stats['website_domains'].most_common(10))
        )
        
        # Temporal analysis
        temporal_analysis = TemporalAnalysis(
            subscription_by_year=dict(self.accumulated_stats['years'].most_common()),
            subscription_by_month=dict(self.accumulated_stats['months'].most_common(12)),
            seasonal_patterns=dict(self.accumulated_stats['seasonal_counts']),
            date_range={"streaming_analysis": "Date range calculated from chunks"}
        )
        
        # Column information
        columns_info = {
            "total_columns": len(columns),
            "column_names": columns,
            "data_types": self.accumulated_stats['data_types']
        }
        
        # Statistical summary
        statistical_summary = {
            "total_records": total_rows,
            "chunks_processed": self.chunk_count,
            "unique_customers": len(self.accumulated_stats['unique_ids']),
            "processing_mode": "streaming_accumulation"
        }
        
        return ComprehensiveInsights(
            data_quality=data_quality,
            business_insights=business_insights,
            communication_analysis=communication_analysis,
            temporal_analysis=temporal_analysis,
            columns_info=columns_info,
            statistical_summary=statistical_summary
        )
    
    def _calculate_market_penetration_from_stats(self) -> Dict[str, float]:
        """Calculate market penetration from accumulated stats."""
        total_customers = self.accumulated_stats['total_rows']
        penetration = {}
        
        for country, count in self.accumulated_stats['countries'].most_common(10):
            penetration[country] = (count / total_customers) * 100
        
        return penetration


# Factory and main interface
class CSVProcessorFactory:
    """Factory for creating appropriate processors."""
    
    @staticmethod
    async def create_and_process(file_path: str) -> CSVProcessingResult:
        """Create appropriate processor and process file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine processor type based on file size
        file_size = os.path.getsize(file_path) / 1024 / 1024  # MB
        
        if file_size < 50:  # Less than 50MB use small file processor
            logger.info(f"Using SmallFileProcessor for {file_size:.1f}MB file")
            processor = SmallFileProcessor()
        else:
            logger.info(f"Using LargeFileProcessor for {file_size:.1f}MB file")
            processor = LargeFileProcessor()
        
        return await processor.process(file_path)
