#!/usr/bin/env python3
"""
Data Quality Assessment Script for EdTech Token Economy Database

This script performs comprehensive data quality checks including:
- Data completeness and missing values
- Data consistency and validity
- Statistical distributions and outliers
- Business logic validation
- Data relationships and referential integrity
- Data freshness and temporal patterns

Author: EdTech Token Economy Team
Version: 1.0.0
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class DataQualityAssessment:
    """Comprehensive data quality assessment for EdTech database"""
    
    def __init__(self, db_path: str = "edtech_token_economy.db"):
        """Initialize the assessment with database path"""
        self.db_path = db_path
        self.conn = None
        self.results = {}
        self.issues = []
        
    def connect(self):
        """Connect to the database"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            print(f"[OK] Connected to database: {self.db_path}")
        except Exception as e:
            print(f"[ERROR] Failed to connect to database: {e}")
            raise
    
    def disconnect(self):
        """Disconnect from the database"""
        if self.conn:
            self.conn.close()
            print("[OK] Database connection closed")
    
    def get_table_info(self) -> Dict[str, Any]:
        """Get basic information about all tables"""
        print("\n" + "="*60)
        print("TABLE INFORMATION")
        print("="*60)
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        table_info = {}
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            
            table_info[table] = {
                'row_count': count,
                'columns': [col[1] for col in columns],
                'column_types': {col[1]: col[2] for col in columns}
            }
            
            print(f"\n{table.upper()}:")
            print(f"  Rows: {count:,}")
            print(f"  Columns: {len(columns)}")
            print(f"  Column names: {', '.join([col[1] for col in columns])}")
        
        self.results['table_info'] = table_info
        return table_info
    
    def check_data_completeness(self) -> Dict[str, Any]:
        """Check for missing values and data completeness"""
        print("\n" + "="*60)
        print("DATA COMPLETENESS ANALYSIS")
        print("="*60)
        
        completeness_results = {}
        
        for table in self.results['table_info'].keys():
            print(f"\n{table.upper()} - Missing Values:")
            df = pd.read_sql_query(f"SELECT * FROM {table}", self.conn)
            
            missing_data = df.isnull().sum()
            missing_pct = (missing_data / len(df)) * 100
            
            table_completeness = {}
            for col in df.columns:
                missing_count = missing_data[col]
                missing_percentage = missing_pct[col]
                
                table_completeness[col] = {
                    'missing_count': int(missing_count),
                    'missing_percentage': round(missing_percentage, 2),
                    'completeness_score': round(100 - missing_percentage, 2)
                }
                
                status = "[OK]" if missing_percentage < 5 else "[WARN]" if missing_percentage < 20 else "[ERROR]"
                print(f"  {status} {col}: {missing_count:,} missing ({missing_percentage:.1f}%)")
                
                if missing_percentage > 20:
                    self.issues.append(f"High missing values in {table}.{col}: {missing_percentage:.1f}%")
            
            completeness_results[table] = table_completeness
        
        self.results['completeness'] = completeness_results
        return completeness_results
    
    def validate_business_rules(self) -> Dict[str, Any]:
        """Validate business logic and constraints"""
        print("\n" + "="*60)
        print("BUSINESS RULE VALIDATION")
        print("="*60)
        
        validation_results = {}
        
        # Courses validation
        print("\nCOURSES VALIDATION:")
        courses_df = pd.read_sql_query("SELECT * FROM courses", self.conn)
        
        course_issues = []
        
        # Price validation
        negative_prices = courses_df[courses_df['token_price'] < 0]
        if not negative_prices.empty:
            course_issues.append(f"Negative prices: {len(negative_prices)} courses")
            print(f"  [ERROR] Negative prices: {len(negative_prices)} courses")
        
        # Rating validation
        invalid_ratings = courses_df[(courses_df['avg_rating'] < 0) | (courses_df['avg_rating'] > 5)]
        if not invalid_ratings.empty:
            course_issues.append(f"Invalid ratings: {len(invalid_ratings)} courses")
            print(f"  [ERROR] Invalid ratings: {len(invalid_ratings)} courses")
        
        # Duration validation
        invalid_duration = courses_df[courses_df['duration_hours'] <= 0]
        if not invalid_duration.empty:
            course_issues.append(f"Invalid duration: {len(invalid_duration)} courses")
            print(f"  [ERROR] Invalid duration: {len(invalid_duration)} courses")
        
        # Enrollments validation
        negative_enrollments = courses_df[courses_df['total_enrollments'] < 0]
        if not negative_enrollments.empty:
            course_issues.append(f"Negative enrollments: {len(negative_enrollments)} courses")
            print(f"  [ERROR] Negative enrollments: {len(negative_enrollments)} courses")
        
        validation_results['courses'] = {
            'total_courses': len(courses_df),
            'issues': course_issues,
            'valid_courses': len(courses_df) - len(negative_prices) - len(invalid_ratings) - len(invalid_duration) - len(negative_enrollments)
        }
        
        # Teachers validation
        print("\nTEACHERS VALIDATION:")
        teachers_df = pd.read_sql_query("SELECT * FROM teachers", self.conn)
        
        teacher_issues = []
        
        # Quality score validation
        invalid_quality = teachers_df[(teachers_df['teacher_quality_score'] < 0) | (teachers_df['teacher_quality_score'] > 100)]
        if not invalid_quality.empty:
            teacher_issues.append(f"Invalid quality scores: {len(invalid_quality)} teachers")
            print(f"  [ERROR] Invalid quality scores: {len(invalid_quality)} teachers")
        
        # Course count validation
        negative_courses = teachers_df[teachers_df['total_courses_created'] < 0]
        if not negative_courses.empty:
            teacher_issues.append(f"Negative course counts: {len(negative_courses)} teachers")
            print(f"  [ERROR] Negative course counts: {len(negative_courses)} teachers")
        
        validation_results['teachers'] = {
            'total_teachers': len(teachers_df),
            'issues': teacher_issues,
            'valid_teachers': len(teachers_df) - len(invalid_quality) - len(negative_courses)
        }
        
        # Enrollments validation
        print("\nENROLLMENTS VALIDATION:")
        enrollments_df = pd.read_sql_query("SELECT * FROM enrollments", self.conn)
        
        enrollment_issues = []
        
        # Token spending validation
        negative_tokens = enrollments_df[enrollments_df['tokens_spent'] < 0]
        if not negative_tokens.empty:
            enrollment_issues.append(f"Negative token spending: {len(negative_tokens)} enrollments")
            print(f"  [ERROR] Negative token spending: {len(negative_tokens)} enrollments")
        
        # Completion validation
        invalid_completion = enrollments_df[~enrollments_df['completed'].isin([0, 1])]
        if not invalid_completion.empty:
            enrollment_issues.append(f"Invalid completion values: {len(invalid_completion)} enrollments")
            print(f"  [ERROR] Invalid completion values: {len(invalid_completion)} enrollments")
        
        validation_results['enrollments'] = {
            'total_enrollments': len(enrollments_df),
            'issues': enrollment_issues,
            'valid_enrollments': len(enrollments_df) - len(negative_tokens) - len(invalid_completion)
        }
        
        self.results['business_rules'] = validation_results
        return validation_results
    
    def analyze_statistical_distributions(self) -> Dict[str, Any]:
        """Analyze statistical distributions and identify outliers"""
        print("\n" + "="*60)
        print("STATISTICAL DISTRIBUTION ANALYSIS")
        print("="*60)
        
        distribution_results = {}
        
        # Courses analysis
        print("\nCOURSES STATISTICAL ANALYSIS:")
        courses_df = pd.read_sql_query("SELECT * FROM courses", self.conn)
        
        numeric_cols = ['token_price', 'total_enrollments', 'avg_rating', 'duration_hours', 'review_count']
        course_stats = {}
        
        for col in numeric_cols:
            if col in courses_df.columns:
                stats = courses_df[col].describe()
                course_stats[col] = {
                    'mean': round(stats['mean'], 2),
                    'median': round(stats['50%'], 2),
                    'std': round(stats['std'], 2),
                    'min': round(stats['min'], 2),
                    'max': round(stats['max'], 2),
                    'q25': round(stats['25%'], 2),
                    'q75': round(stats['75%'], 2)
                }
                
                # Outlier detection using IQR
                Q1 = stats['25%']
                Q3 = stats['75%']
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = courses_df[(courses_df[col] < lower_bound) | (courses_df[col] > upper_bound)]
                outlier_pct = (len(outliers) / len(courses_df)) * 100
                
                print(f"  {col}:")
                print(f"    Mean: {stats['mean']:.2f}, Median: {stats['50%']:.2f}")
                print(f"    Range: {stats['min']:.2f} - {stats['max']:.2f}")
                print(f"    Outliers: {len(outliers)} ({outlier_pct:.1f}%)")
                
                if outlier_pct > 10:
                    self.issues.append(f"High outlier percentage in courses.{col}: {outlier_pct:.1f}%")
        
        distribution_results['courses'] = course_stats
        
        # Price-Enrollment correlation analysis
        print("\nPRICE-ENROLLMENT CORRELATION ANALYSIS:")
        price_enrollment_corr = courses_df['token_price'].corr(courses_df['total_enrollments'])
        print(f"  Price-Enrollment Correlation: {price_enrollment_corr:.3f}")
        
        if abs(price_enrollment_corr) < 0.1:
            self.issues.append(f"Very weak price-enrollment correlation: {price_enrollment_corr:.3f}")
        
        distribution_results['correlations'] = {
            'price_enrollment': round(price_enrollment_corr, 3)
        }
        
        self.results['distributions'] = distribution_results
        return distribution_results
    
    def check_referential_integrity(self) -> Dict[str, Any]:
        """Check referential integrity between tables"""
        print("\n" + "="*60)
        print("REFERENTIAL INTEGRITY CHECK")
        print("="*60)
        
        integrity_results = {}
        
        # Check course-teacher relationships
        print("\nCOURSE-TEACHER INTEGRITY:")
        courses_df = pd.read_sql_query("SELECT course_id, teacher_id FROM courses", self.conn)
        teachers_df = pd.read_sql_query("SELECT teacher_id FROM teachers", self.conn)
        
        orphaned_courses = courses_df[~courses_df['teacher_id'].isin(teachers_df['teacher_id'])]
        if not orphaned_courses.empty:
            print(f"  [ERROR] Orphaned courses: {len(orphaned_courses)} courses reference non-existent teachers")
            self.issues.append(f"Orphaned courses: {len(orphaned_courses)} courses")
        else:
            print(f"  [OK] All courses have valid teacher references")
        
        integrity_results['course_teacher'] = {
            'orphaned_courses': len(orphaned_courses),
            'valid_references': len(courses_df) - len(orphaned_courses)
        }
        
        # Check enrollment-course relationships
        print("\nENROLLMENT-COURSE INTEGRITY:")
        enrollments_df = pd.read_sql_query("SELECT enrollment_id, course_id FROM enrollments", self.conn)
        
        orphaned_enrollments = enrollments_df[~enrollments_df['course_id'].isin(courses_df['course_id'])]
        if not orphaned_enrollments.empty:
            print(f"  [ERROR] Orphaned enrollments: {len(orphaned_enrollments)} enrollments reference non-existent courses")
            self.issues.append(f"Orphaned enrollments: {len(orphaned_enrollments)} enrollments")
        else:
            print(f"  [OK] All enrollments have valid course references")
        
        integrity_results['enrollment_course'] = {
            'orphaned_enrollments': len(orphaned_enrollments),
            'valid_references': len(enrollments_df) - len(orphaned_enrollments)
        }
        
        self.results['referential_integrity'] = integrity_results
        return integrity_results
    
    def analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze temporal patterns and data freshness"""
        print("\n" + "="*60)
        print("TEMPORAL PATTERN ANALYSIS")
        print("="*60)
        
        temporal_results = {}
        
        # Enrollment date analysis
        print("\nENROLLMENT TEMPORAL PATTERNS:")
        enrollments_df = pd.read_sql_query("SELECT enrollment_date FROM enrollments", self.conn)
        
        if 'enrollment_date' in enrollments_df.columns:
            enrollments_df['enrollment_date'] = pd.to_datetime(enrollments_df['enrollment_date'])
            
            date_range = {
                'earliest': enrollments_df['enrollment_date'].min(),
                'latest': enrollments_df['enrollment_date'].max(),
                'span_days': (enrollments_df['enrollment_date'].max() - enrollments_df['enrollment_date'].min()).days
            }
            
            print(f"  Date range: {date_range['earliest']} to {date_range['latest']}")
            print(f"  Span: {date_range['span_days']} days")
            
            # Monthly enrollment patterns
            monthly_enrollments = enrollments_df.groupby(enrollments_df['enrollment_date'].dt.to_period('M')).size()
            print(f"  Monthly enrollment range: {monthly_enrollments.min()} - {monthly_enrollments.max()}")
            
            temporal_results['enrollments'] = {
                'date_range': date_range,
                'monthly_stats': {
                    'min': int(monthly_enrollments.min()),
                    'max': int(monthly_enrollments.max()),
                    'mean': round(monthly_enrollments.mean(), 1)
                }
            }
        
        self.results['temporal_patterns'] = temporal_results
        return temporal_results
    
    def generate_data_quality_score(self) -> Dict[str, Any]:
        """Generate overall data quality score"""
        print("\n" + "="*60)
        print("DATA QUALITY SCORE")
        print("="*60)
        
        scores = {}
        
        # Completeness score
        completeness_score = 0
        total_columns = 0
        for table, data in self.results['completeness'].items():
            for col, metrics in data.items():
                completeness_score += metrics['completeness_score']
                total_columns += 1
        
        avg_completeness = completeness_score / total_columns if total_columns > 0 else 0
        scores['completeness'] = round(avg_completeness, 1)
        
        # Validity score (based on business rules)
        total_records = 0
        valid_records = 0
        for table, data in self.results['business_rules'].items():
            total_records += data['total_' + table]
            valid_records += data['valid_' + table]
        
        validity_score = (valid_records / total_records * 100) if total_records > 0 else 0
        scores['validity'] = round(validity_score, 1)
        
        # Integrity score
        total_references = 0
        valid_references = 0
        for table, data in self.results['referential_integrity'].items():
            # Map table names to their orphaned key names
            orphaned_key_map = {
                'course_teacher': 'orphaned_courses',
                'enrollment_course': 'orphaned_enrollments'
            }
            orphaned_key = orphaned_key_map.get(table, 'orphaned_' + table.split('_')[0])
            total_references += data[orphaned_key] + data['valid_references']
            valid_references += data['valid_references']
        
        integrity_score = (valid_references / total_references * 100) if total_references > 0 else 100
        scores['integrity'] = round(integrity_score, 1)
        
        # Overall score
        overall_score = (scores['completeness'] + scores['validity'] + scores['integrity']) / 3
        scores['overall'] = round(overall_score, 1)
        
        print(f"\nData Quality Scores:")
        print(f"  Completeness: {scores['completeness']}/100")
        print(f"  Validity: {scores['validity']}/100")
        print(f"  Integrity: {scores['integrity']}/100")
        print(f"  Overall: {scores['overall']}/100")
        
        # Quality grade
        if overall_score >= 90:
            grade = "A (Excellent)"
        elif overall_score >= 80:
            grade = "B (Good)"
        elif overall_score >= 70:
            grade = "C (Fair)"
        elif overall_score >= 60:
            grade = "D (Poor)"
        else:
            grade = "F (Very Poor)"
        
        print(f"  Grade: {grade}")
        
        self.results['quality_scores'] = scores
        return scores
    
    def generate_report(self) -> str:
        """Generate comprehensive data quality report"""
        report = []
        report.append("EDTECH TOKEN ECONOMY - DATA QUALITY ASSESSMENT REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Database: {self.db_path}")
        report.append("")
        
        # Summary
        if 'quality_scores' in self.results:
            scores = self.results['quality_scores']
            report.append("EXECUTIVE SUMMARY")
            report.append("-" * 40)
            report.append(f"Overall Data Quality Score: {scores['overall']}/100")
            report.append(f"Completeness: {scores['completeness']}/100")
            report.append(f"Validity: {scores['validity']}/100")
            report.append(f"Integrity: {scores['integrity']}/100")
            report.append("")
        
        # Issues
        if self.issues:
            report.append("CRITICAL ISSUES IDENTIFIED")
            report.append("-" * 40)
            for i, issue in enumerate(self.issues, 1):
                report.append(f"{i}. {issue}")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        if scores['overall'] < 70:
            report.append("1. Data quality is below acceptable standards")
            report.append("2. Consider regenerating data with improved logic")
            report.append("3. Implement data validation rules")
        if len(self.issues) > 5:
            report.append("4. High number of issues detected - comprehensive review needed")
        
        return "\n".join(report)
    
    def run_full_assessment(self) -> Dict[str, Any]:
        """Run complete data quality assessment"""
        print("EDTECH TOKEN ECONOMY - DATA QUALITY ASSESSMENT")
        print("=" * 80)
        
        try:
            self.connect()
            
            # Run all assessments
            self.get_table_info()
            self.check_data_completeness()
            self.validate_business_rules()
            self.analyze_statistical_distributions()
            self.check_referential_integrity()
            self.analyze_temporal_patterns()
            self.generate_data_quality_score()
            
            # Generate report
            report = self.generate_report()
            print("\n" + report)
            
            return self.results
            
        except Exception as e:
            print(f"[ERROR] Assessment failed: {e}")
            raise
        finally:
            self.disconnect()


def main():
    """Main function for command-line execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='EdTech Token Economy Data Quality Assessment')
    parser.add_argument('--db-path', default='edtech_token_economy.db', 
                       help='Path to the database file')
    parser.add_argument('--output', help='Output file for detailed report')
    
    args = parser.parse_args()
    
    # Run assessment
    assessor = DataQualityAssessment(args.db_path)
    results = assessor.run_full_assessment()
    
    # Save detailed report if requested
    if args.output:
        report = assessor.generate_report()
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"\n[OK] Detailed report saved to: {args.output}")
    
    return results


if __name__ == "__main__":
    main()
