"""
EdTech Token Economy Database Module

This module handles database connections, queries, and data extraction
for the EdTech token economy platform.

Author: EdTech Token Economy Team
Date: October 2025
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    db_path: str = "edtech_token_economy.db"
    timeout: int = 30


class EdTechDatabaseManager:
    """Main class for EdTech database operations and data extraction"""

    def __init__(self, config: DatabaseConfig = None):
        """
        Initialize database manager
        
        Args:
            config: DatabaseConfig object with connection settings
        """
        self.config = config or DatabaseConfig()
        self.connection = None

    def connect(self) -> bool:
        """
        Establish database connection
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.connection = sqlite3.connect(
                self.config.db_path,
                timeout=self.config.timeout
            )
            self.connection.row_factory = sqlite3.Row
            logger.info(f"Connected to database: {self.config.db_path}")
            return True
        except sqlite3.Error as e:
            logger.error(f"Database connection failed: {e}")
            return False

    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")

    def execute_query(self, query: str, params: Tuple = None) -> pd.DataFrame:
        """
        Execute a custom SQL query and return results as DataFrame
        
        Args:
            query: SQL query string
            params: Optional query parameters
            
        Returns:
            DataFrame with query results
        """
        try:
            if params:
                df = pd.read_sql_query(query, self.connection, params=params)
            else:
                df = pd.read_sql_query(query, self.connection)
            
            logger.info(f"Query executed successfully. Returned {len(df)} rows")
            return df
        
        except sqlite3.Error as e:
            logger.error(f"Query execution failed: {e}")
            return pd.DataFrame()

    def get_enrollments_with_details(self, start_date: str = None, 
                                     end_date: str = None) -> pd.DataFrame:
        """
        Get enrollment data joined with course, learner, and teacher information
        
        Args:
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
            
        Returns:
            DataFrame with joined enrollment data
        """
        base_query = """
            SELECT
                e.enrollment_id,
                e.learner_id,
                e.course_id,
                e.teacher_id,
                e.enrollment_date,
                e.tokens_spent,
                e.tokens_to_teacher,
                e.tokens_to_platform,
                e.completed,
                e.rating_given,
                e.discount_received_pct,
                e.device_type,
                e.referral_source,
                c.course_title,
                c.category,
                c.subcategory,
                c.difficulty_level,
                c.duration_hours,
                c.token_price,
                c.original_token_price,
                c.avg_rating as course_avg_rating,
                c.total_enrollments as course_total_enrollments,
                l.user_type,
                l.education_level,
                l.age_group,
                l.location,
                l.completion_rate as learner_completion_rate,
                l.price_sensitivity_score,
                l.token_balance,
                t.teacher_quality_score,
                t.quality_tier,
                t.avg_course_rating as teacher_avg_rating,
                t.specialization
            FROM enrollments e
            LEFT JOIN courses c ON e.course_id = c.course_id
            LEFT JOIN learners l ON e.learner_id = l.learner_id
            LEFT JOIN teachers t ON e.teacher_id = t.teacher_id
        """
        
        if start_date and end_date:
            query = f"{base_query} WHERE date(e.enrollment_date) BETWEEN ? AND ?"
            params = (start_date, end_date)
        elif start_date:
            query = f"{base_query} WHERE date(e.enrollment_date) >= ?"
            params = (start_date,)
        elif end_date:
            query = f"{base_query} WHERE date(e.enrollment_date) <= ?"
            params = (end_date,)
        else:
            query = base_query
            params = None
        
        return self.execute_query(query, params)

    def get_price_elasticity_data(self) -> pd.DataFrame:
        """
        Prepare data for token price elasticity analysis
        
        Returns:
            DataFrame with price and enrollment data for elasticity modeling
        """
        query = """
            SELECT
                c.course_id,
                c.course_title,
                c.category,
                c.subcategory,
                c.difficulty_level,
                c.duration_hours,
                c.token_price,
                c.original_token_price,
                c.current_discount_pct,
                c.total_enrollments,
                c.avg_rating,
                c.review_count,
                c.completion_rate,
                c.teacher_earnings_per_enrollment,
                c.platform_fee_pct,
                c.competitive_courses_count,
                c.certificate_offered,
                c.video_count,
                c.assignment_count,
                t.teacher_id,
                t.teacher_quality_score,
                t.quality_tier,
                t.avg_course_rating as teacher_avg_rating,
                t.total_students_taught,
                t.total_courses_created,
                t.specialization,
                COUNT(e.enrollment_id) as actual_enrollments,
                AVG(e.tokens_spent) as avg_tokens_spent,
                SUM(e.tokens_spent) as total_revenue,
                AVG(CASE WHEN e.completed = 1 THEN 1 ELSE 0 END) as actual_completion_rate
            FROM courses c
            LEFT JOIN teachers t ON c.teacher_id = t.teacher_id
            LEFT JOIN enrollments e ON c.course_id = e.course_id
            WHERE e.enrollment_id IS NOT NULL
            GROUP BY c.course_id, c.course_title, c.category, c.subcategory,
                     c.difficulty_level, c.duration_hours, c.token_price,
                     c.original_token_price, c.current_discount_pct,
                     c.total_enrollments, c.avg_rating, c.review_count,
                     c.completion_rate, c.teacher_earnings_per_enrollment,
                     c.platform_fee_pct, c.competitive_courses_count,
                     c.certificate_offered, c.video_count, c.assignment_count,
                     t.teacher_id, t.teacher_quality_score, t.quality_tier,
                     t.teacher_avg_rating, t.total_students_taught,
                     t.total_courses_created, t.specialization
        """
        
        df = self.execute_query(query)
        
        # Create additional features for elasticity modeling
        if not df.empty:
            # Price relative to category average
            df['price_vs_category_avg'] = df.groupby('category')['token_price'].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            )
            
            # Quality score
            df['quality_score'] = (
                df['avg_rating'] / 5 * 0.4 +
                df['completion_rate'] * 0.3 +
                df['teacher_quality_score'] / 100 * 0.3
            )
            
            # Demand intensity
            df['demand_intensity'] = df['total_enrollments'] / (df['duration_hours'] + 1)
            
        return df

    def get_learner_summary(self) -> pd.DataFrame:
        """
        Get learner behavior summary statistics
        
        Returns:
            DataFrame with learner summary metrics
        """
        query = """
            SELECT
                l.*,
                COUNT(e.enrollment_id) as enrollments_count,
                SUM(CASE WHEN e.completed = 1 THEN 1 ELSE 0 END) as completed_count,
                AVG(e.tokens_spent) as avg_spending_per_enrollment,
                MAX(e.enrollment_date) as last_enrollment_date,
                MIN(e.enrollment_date) as first_enrollment_date
            FROM learners l
            LEFT JOIN enrollments e ON l.learner_id = e.learner_id
            GROUP BY l.learner_id
            ORDER BY l.total_tokens_spent DESC
        """
        
        return self.execute_query(query)

    def get_teacher_performance(self) -> pd.DataFrame:
        """
        Get teacher performance metrics
        
        Returns:
            DataFrame with teacher performance data
        """
        query = """
            SELECT
                t.*,
                COUNT(DISTINCT c.course_id) as active_courses,
                COUNT(e.enrollment_id) as total_actual_enrollments,
                SUM(e.tokens_to_teacher) as total_actual_earnings,
                AVG(e.rating_given) as avg_student_rating,
                AVG(CASE WHEN e.completed = 1 THEN 1 ELSE 0 END) as actual_completion_rate,
                SUM(e.tokens_spent) as total_revenue_generated
            FROM teachers t
            LEFT JOIN courses c ON t.teacher_id = c.teacher_id
            LEFT JOIN enrollments e ON c.course_id = e.course_id
            GROUP BY t.teacher_id
            ORDER BY total_actual_earnings DESC
        """
        
        return self.execute_query(query)

    def get_course_performance(self) -> pd.DataFrame:
        """
        Get course performance metrics
        
        Returns:
            DataFrame with course performance data
        """
        query = """
            SELECT
                c.*,
                t.teacher_quality_score,
                t.quality_tier,
                COUNT(e.enrollment_id) as actual_enrollments,
                SUM(e.tokens_spent) as actual_revenue,
                AVG(e.rating_given) as actual_avg_rating,
                SUM(CASE WHEN e.completed = 1 THEN 1 ELSE 0 END) as completed_enrollments,
                SUM(CASE WHEN e.refunded = 1 THEN 1 ELSE 0 END) as refunded_enrollments,
                (SUM(CASE WHEN e.completed = 1 THEN 1 ELSE 0 END) * 1.0 / 
                 COUNT(e.enrollment_id)) as actual_completion_rate
            FROM courses c
            LEFT JOIN teachers t ON c.teacher_id = t.teacher_id
            LEFT JOIN enrollments e ON c.course_id = e.course_id
            WHERE e.enrollment_id IS NOT NULL
            GROUP BY c.course_id
            ORDER BY actual_revenue DESC
        """
        
        return self.execute_query(query)

    def get_category_analysis(self) -> pd.DataFrame:
        """
        Analyze performance by course category
        
        Returns:
            DataFrame with category analysis
        """
        query = """
            SELECT
                c.category,
                COUNT(DISTINCT c.course_id) as total_courses,
                COUNT(e.enrollment_id) as total_enrollments,
                SUM(e.tokens_spent) as total_revenue,
                AVG(c.token_price) as avg_token_price,
                AVG(e.rating_given) as avg_rating,
                AVG(CASE WHEN e.completed = 1 THEN 1 ELSE 0 END) as avg_completion_rate,
                COUNT(DISTINCT t.teacher_id) as total_teachers,
                COUNT(DISTINCT e.learner_id) as total_unique_learners
            FROM courses c
            LEFT JOIN enrollments e ON c.course_id = e.course_id
            LEFT JOIN teachers t ON c.teacher_id = t.teacher_id
            WHERE e.enrollment_id IS NOT NULL
            GROUP BY c.category
            ORDER BY total_revenue DESC
        """
        
        return self.execute_query(query)

    def get_token_economy_metrics(self, days: int = 30) -> pd.DataFrame:
        """
        Get token economy health metrics
        
        Args:
            days: Number of days to analyze
            
        Returns:
            DataFrame with token economy metrics
        """
        cutoff_date = (datetime.now() - timedelta(days=days)).date().isoformat()
        
        query = """
            SELECT
                date(e.enrollment_date) as date,
                COUNT(e.enrollment_id) as daily_enrollments,
                SUM(e.tokens_spent) as daily_tokens_burned,
                AVG(e.tokens_spent) as avg_tokens_per_enrollment,
                SUM(e.tokens_to_teacher) as daily_tokens_to_teachers,
                SUM(e.tokens_to_platform) as daily_platform_revenue,
                COUNT(DISTINCT e.learner_id) as daily_active_learners,
                COUNT(DISTINCT e.teacher_id) as daily_active_teachers,
                COUNT(DISTINCT e.course_id) as daily_active_courses
            FROM enrollments e
            WHERE date(e.enrollment_date) >= ?
            GROUP BY date(e.enrollment_date)
            ORDER BY date(e.enrollment_date)
        """
        
        return self.execute_query(query, (cutoff_date,))

    def get_learner_propensity_data(self) -> pd.DataFrame:
        """
        Prepare data for learner enrollment propensity modeling
        
        Returns:
            DataFrame for propensity modeling
        """
        query = """
            SELECT
                l.*,
                COUNT(e.enrollment_id) as historical_enrollments,
                MAX(date(e.enrollment_date)) as last_enrollment_date,
                AVG(e.tokens_spent) as avg_historical_spending,
                SUM(CASE WHEN e.completed = 1 THEN 1 ELSE 0 END) as historical_completions,
                AVG(e.rating_given) as avg_rating_given,
                julianday('now') - julianday(MAX(e.enrollment_date)) as days_since_last_enrollment,
                CASE WHEN julianday('now') - julianday(MAX(e.enrollment_date)) <= 30 
                     THEN 1 ELSE 0 END as enrolled_last_30_days
            FROM learners l
            LEFT JOIN enrollments e ON l.learner_id = e.learner_id
            GROUP BY l.learner_id
        """
        
        return self.execute_query(query)

    def export_all_tables(self, output_dir: str = "data/processed") -> Dict[str, str]:
        """
        Export all tables to CSV files
        
        Args:
            output_dir: Directory to save CSV files
            
        Returns:
            Dictionary mapping table names to file paths
        """
        import os
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        tables = ['learners', 'teachers', 'courses', 'enrollments', 
                 'token_transactions', 'platform_metrics']
        exported_files = {}
        
        for table in tables:
            try:
                df = pd.read_sql(f"SELECT * FROM {table}", self.connection)
                file_path = os.path.join(output_dir, f"{table}.csv")
                df.to_csv(file_path, index=False)
                exported_files[table] = file_path
                logger.info(f"Exported {table} to {file_path}")
            except Exception as e:
                logger.error(f"Failed to export {table}: {e}")
        
        return exported_files


# Context manager for database connections
class EdTechDatabaseConnection:
    """Context manager for EdTech database connections"""
    
    def __init__(self, config: DatabaseConfig = None):
        self.config = config or DatabaseConfig()
        self.manager = EdTechDatabaseManager(self.config)
        self._connected = False
    
    def __enter__(self):
        self._connected = self.manager.connect()
        if not self._connected:
            raise ConnectionError("Failed to connect to database")
        return self.manager
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.manager.disconnect()


if __name__ == "__main__":
    print("Testing EdTech Database Module...")
    
    db_manager = EdTechDatabaseManager()
    
    if db_manager.connect():
        print("✓ Database connection successful")
        
        # Test queries
        elasticity_data = db_manager.get_price_elasticity_data()
        print(f"✓ Price elasticity data: {len(elasticity_data)} courses")
        
        learner_summary = db_manager.get_learner_summary()
        print(f"✓ Learner summary: {len(learner_summary)} learners")
        
        category_analysis = db_manager.get_category_analysis()
        print(f"✓ Category analysis: {len(category_analysis)} categories")
        
        db_manager.disconnect()
        print("✓ Database connection closed")
    else:
        print("✗ Database connection failed")
    
    print("EdTech Database Module test completed!")


