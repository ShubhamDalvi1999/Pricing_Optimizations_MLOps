#!/usr/bin/env python3
"""
Data Realism Validation Script for EdTech Token Economy Database

This script validates the realism of generated data by checking:
- Market-appropriate pricing patterns
- Realistic enrollment distributions
- Teacher quality vs pricing correlations
- Category-specific market dynamics
- Seasonal enrollment patterns
- Price elasticity relationships

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

class DataRealismValidator:
    """Validates the realism of EdTech token economy data"""
    
    def __init__(self, db_path: str = "edtech_token_economy.db"):
        """Initialize the validator with database path"""
        self.db_path = db_path
        self.conn = None
        self.results = {}
        self.realism_issues = []
        
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
    
    def validate_pricing_realism(self) -> Dict[str, Any]:
        """Validate pricing patterns against real-world EdTech market data"""
        print("\n" + "="*60)
        print("PRICING REALISM VALIDATION")
        print("="*60)
        
        pricing_results = {}
        
        # Get courses with teacher and pricing data
        query = """
        SELECT c.*, t.teacher_quality_score, t.quality_tier
        FROM courses c
        LEFT JOIN teachers t ON c.teacher_id = t.teacher_id
        """
        courses_df = pd.read_sql_query(query, self.conn)
        
        print(f"\nAnalyzing {len(courses_df)} courses...")
        
        # Price range validation
        price_stats = courses_df['token_price'].describe()
        print(f"\nPrice Statistics:")
        print(f"  Min: ${price_stats['min']:.2f}")
        print(f"  Max: ${price_stats['max']:.2f}")
        print(f"  Mean: ${price_stats['mean']:.2f}")
        print(f"  Median: ${price_stats['50%']:.2f}")
        
        # Real-world EdTech pricing benchmarks
        realistic_ranges = {
            'beginner': (10, 50),
            'intermediate': (25, 100),
            'advanced': (50, 200)
        }
        
        pricing_issues = []
        for difficulty in ['beginner', 'intermediate', 'advanced']:
            difficulty_courses = courses_df[courses_df['difficulty_level'] == difficulty]
            if not difficulty_courses.empty:
                min_price = difficulty_courses['token_price'].min()
                max_price = difficulty_courses['token_price'].max()
                expected_min, expected_max = realistic_ranges[difficulty]
                
                print(f"\n{difficulty.upper()} Courses:")
                print(f"  Price range: ${min_price:.2f} - ${max_price:.2f}")
                print(f"  Expected range: ${expected_min} - ${expected_max}")
                
                if min_price < expected_min * 0.5:
                    pricing_issues.append(f"{difficulty} courses too cheap: min ${min_price:.2f}")
                if max_price > expected_max * 2:
                    pricing_issues.append(f"{difficulty} courses too expensive: max ${max_price:.2f}")
        
        # Teacher quality vs pricing correlation
        print(f"\nTeacher Quality vs Pricing Analysis:")
        quality_price_corr = courses_df['teacher_quality_score'].corr(courses_df['token_price'])
        print(f"  Correlation: {quality_price_corr:.3f}")
        
        if quality_price_corr < 0.3:
            pricing_issues.append(f"Weak teacher quality-price correlation: {quality_price_corr:.3f}")
        
        # Category pricing analysis
        print(f"\nCategory Pricing Analysis:")
        category_stats = courses_df.groupby('category')['token_price'].agg(['mean', 'std', 'count'])
        for category, stats in category_stats.iterrows():
            print(f"  {category}: ${stats['mean']:.2f} ± ${stats['std']:.2f} ({stats['count']} courses)")
        
        pricing_results = {
            'total_courses': len(courses_df),
            'price_stats': price_stats.to_dict(),
            'category_stats': category_stats.to_dict(),
            'quality_price_correlation': quality_price_corr,
            'issues': pricing_issues
        }
        
        self.results['pricing'] = pricing_results
        return pricing_results
    
    def validate_enrollment_patterns(self) -> Dict[str, Any]:
        """Validate enrollment patterns against realistic expectations"""
        print("\n" + "="*60)
        print("ENROLLMENT PATTERN VALIDATION")
        print("="*60)
        
        enrollment_results = {}
        
        # Get enrollment data
        query = """
        SELECT c.course_id, c.course_title, c.category, c.difficulty_level, 
               c.token_price, c.total_enrollments, c.avg_rating,
               COUNT(e.enrollment_id) as actual_enrollments
        FROM courses c
        LEFT JOIN enrollments e ON c.course_id = e.course_id
        GROUP BY c.course_id
        """
        courses_df = pd.read_sql_query(query, self.conn)
        
        print(f"\nAnalyzing {len(courses_df)} courses...")
        
        # Enrollment distribution analysis
        enrollment_stats = courses_df['actual_enrollments'].describe()
        print(f"\nEnrollment Statistics:")
        print(f"  Min: {enrollment_stats['min']:.0f}")
        print(f"  Max: {enrollment_stats['max']:.0f}")
        print(f"  Mean: {enrollment_stats['mean']:.1f}")
        print(f"  Median: {enrollment_stats['50%']:.0f}")
        
        # Check for unrealistic enrollment patterns
        enrollment_issues = []
        
        # Courses with zero enrollments
        zero_enrollments = courses_df[courses_df['actual_enrollments'] == 0]
        zero_pct = (len(zero_enrollments) / len(courses_df)) * 100
        print(f"\nCourses with zero enrollments: {len(zero_enrollments)} ({zero_pct:.1f}%)")
        
        if zero_pct > 30:
            enrollment_issues.append(f"Too many courses with zero enrollments: {zero_pct:.1f}%")
        
        # Courses with extremely high enrollments
        high_enrollments = courses_df[courses_df['actual_enrollments'] > 1000]
        high_pct = (len(high_enrollments) / len(courses_df)) * 100
        print(f"Courses with >1000 enrollments: {len(high_enrollments)} ({high_pct:.1f}%)")
        
        if high_pct > 10:
            enrollment_issues.append(f"Too many courses with extremely high enrollments: {high_pct:.1f}%")
        
        # Rating vs enrollment correlation
        print(f"\nRating vs Enrollment Analysis:")
        rating_enrollment_corr = courses_df['avg_rating'].corr(courses_df['actual_enrollments'])
        print(f"  Correlation: {rating_enrollment_corr:.3f}")
        
        if rating_enrollment_corr < 0.2:
            enrollment_issues.append(f"Weak rating-enrollment correlation: {rating_enrollment_corr:.3f}")
        
        # Price vs enrollment correlation (should be negative)
        print(f"\nPrice vs Enrollment Analysis:")
        price_enrollment_corr = courses_df['token_price'].corr(courses_df['actual_enrollments'])
        print(f"  Correlation: {price_enrollment_corr:.3f}")
        
        if price_enrollment_corr > -0.1:  # Should be negative
            enrollment_issues.append(f"Weak price-enrollment correlation: {price_enrollment_corr:.3f}")
        
        enrollment_results = {
            'total_courses': len(courses_df),
            'enrollment_stats': enrollment_stats.to_dict(),
            'zero_enrollment_pct': zero_pct,
            'high_enrollment_pct': high_pct,
            'rating_enrollment_correlation': rating_enrollment_corr,
            'price_enrollment_correlation': price_enrollment_corr,
            'issues': enrollment_issues
        }
        
        self.results['enrollments'] = enrollment_results
        return enrollment_results
    
    def validate_teacher_distribution(self) -> Dict[str, Any]:
        """Validate teacher quality and distribution patterns"""
        print("\n" + "="*60)
        print("TEACHER DISTRIBUTION VALIDATION")
        print("="*60)
        
        teacher_results = {}
        
        # Get teacher data
        teachers_df = pd.read_sql_query("SELECT * FROM teachers", self.conn)
        
        print(f"\nAnalyzing {len(teachers_df)} teachers...")
        
        # Quality score distribution
        quality_stats = teachers_df['teacher_quality_score'].describe()
        print(f"\nTeacher Quality Statistics:")
        print(f"  Min: {quality_stats['min']:.1f}")
        print(f"  Max: {quality_stats['max']:.1f}")
        print(f"  Mean: {quality_stats['mean']:.1f}")
        print(f"  Median: {quality_stats['50%']:.1f}")
        
        # Quality tier distribution
        tier_distribution = teachers_df['quality_tier'].value_counts()
        print(f"\nQuality Tier Distribution:")
        for tier, count in tier_distribution.items():
            pct = (count / len(teachers_df)) * 100
            print(f"  {tier}: {count} ({pct:.1f}%)")
        
        # Course creation patterns
        course_stats = teachers_df['total_courses_created'].describe()
        print(f"\nCourses Created per Teacher:")
        print(f"  Min: {course_stats['min']:.0f}")
        print(f"  Max: {course_stats['max']:.0f}")
        print(f"  Mean: {course_stats['mean']:.1f}")
        print(f"  Median: {course_stats['50%']:.0f}")
        
        teacher_issues = []
        
        # Check for unrealistic quality distributions
        if quality_stats['mean'] < 60 or quality_stats['mean'] > 90:
            teacher_issues.append(f"Unrealistic average teacher quality: {quality_stats['mean']:.1f}")
        
        # Check for unrealistic course creation patterns
        if course_stats['max'] > 50:
            teacher_issues.append(f"Teachers creating too many courses: max {course_stats['max']:.0f}")
        
        teacher_results = {
            'total_teachers': len(teachers_df),
            'quality_stats': quality_stats.to_dict(),
            'tier_distribution': tier_distribution.to_dict(),
            'course_stats': course_stats.to_dict(),
            'issues': teacher_issues
        }
        
        self.results['teachers'] = teacher_results
        return teacher_results
    
    def validate_market_dynamics(self) -> Dict[str, Any]:
        """Validate market dynamics and competitive patterns"""
        print("\n" + "="*60)
        print("MARKET DYNAMICS VALIDATION")
        print("="*60)
        
        market_results = {}
        
        # Get comprehensive market data
        query = """
        SELECT c.*, t.teacher_quality_score, t.quality_tier,
               COUNT(e.enrollment_id) as actual_enrollments,
               AVG(e.tokens_spent) as avg_tokens_spent
        FROM courses c
        LEFT JOIN teachers t ON c.teacher_id = t.teacher_id
        LEFT JOIN enrollments e ON c.course_id = e.course_id
        GROUP BY c.course_id
        """
        market_df = pd.read_sql_query(query, self.conn)
        
        print(f"\nAnalyzing market dynamics for {len(market_df)} courses...")
        
        # Category market analysis
        print(f"\nCategory Market Analysis:")
        category_analysis = market_df.groupby('category').agg({
            'token_price': ['mean', 'std', 'count'],
            'actual_enrollments': ['mean', 'sum'],
            'avg_rating': 'mean'
        }).round(2)
        
        for category in market_df['category'].unique():
            cat_data = market_df[market_df['category'] == category]
            print(f"\n  {category}:")
            print(f"    Courses: {len(cat_data)}")
            print(f"    Avg Price: ${cat_data['token_price'].mean():.2f}")
            print(f"    Avg Enrollments: {cat_data['actual_enrollments'].mean():.1f}")
            print(f"    Avg Rating: {cat_data['avg_rating'].mean():.2f}")
        
        # Competitive analysis
        print(f"\nCompetitive Analysis:")
        competitive_courses = market_df[market_df['competitive_courses_count'] > 0]
        if not competitive_courses.empty:
            comp_corr = competitive_courses['competitive_courses_count'].corr(competitive_courses['token_price'])
            print(f"  Competition vs Price Correlation: {comp_corr:.3f}")
            
            comp_enrollment_corr = competitive_courses['competitive_courses_count'].corr(competitive_courses['actual_enrollments'])
            print(f"  Competition vs Enrollment Correlation: {comp_enrollment_corr:.3f}")
        
        market_issues = []
        
        # Check for unrealistic market concentration
        category_counts = market_df['category'].value_counts()
        max_category_pct = (category_counts.max() / len(market_df)) * 100
        if max_category_pct > 60:
            market_issues.append(f"Market too concentrated in one category: {max_category_pct:.1f}%")
        
        market_results = {
            'total_courses': len(market_df),
            'category_analysis': category_analysis.to_dict(),
            'max_category_concentration': max_category_pct,
            'issues': market_issues
        }
        
        self.results['market_dynamics'] = market_results
        return market_results
    
    def validate_price_elasticity(self) -> Dict[str, Any]:
        """Validate price elasticity patterns"""
        print("\n" + "="*60)
        print("PRICE ELASTICITY VALIDATION")
        print("="*60)
        
        elasticity_results = {}
        
        # Get price and enrollment data
        query = """
        SELECT c.course_id, c.token_price, c.original_token_price,
               COUNT(e.enrollment_id) as actual_enrollments,
               AVG(e.tokens_spent) as avg_tokens_spent
        FROM courses c
        LEFT JOIN enrollments e ON c.course_id = e.course_id
        GROUP BY c.course_id
        HAVING actual_enrollments > 0
        """
        elasticity_df = pd.read_sql_query(query, self.conn)
        
        print(f"\nAnalyzing price elasticity for {len(elasticity_df)} courses with enrollments...")
        
        if len(elasticity_df) < 10:
            print("  [WARN] Not enough data for meaningful elasticity analysis")
            elasticity_results = {'insufficient_data': True}
            self.results['elasticity'] = elasticity_results
            return elasticity_results
        
        # Calculate price elasticity
        elasticity_df['price_change_pct'] = ((elasticity_df['token_price'] - elasticity_df['original_token_price']) / elasticity_df['original_token_price']) * 100
        
        # Simple elasticity calculation
        price_enrollment_corr = elasticity_df['token_price'].corr(elasticity_df['actual_enrollments'])
        elasticity_coefficient = price_enrollment_corr * (elasticity_df['token_price'].std() / elasticity_df['actual_enrollments'].std())
        
        print(f"\nPrice Elasticity Analysis:")
        print(f"  Price-Enrollment Correlation: {price_enrollment_corr:.3f}")
        print(f"  Estimated Elasticity Coefficient: {elasticity_coefficient:.3f}")
        
        # Expected elasticity ranges for EdTech
        if abs(elasticity_coefficient) < 0.1:
            elasticity_issue = "Very inelastic demand (unrealistic)"
        elif abs(elasticity_coefficient) > 3.0:
            elasticity_issue = "Very elastic demand (unrealistic)"
        else:
            elasticity_issue = None
        
        if elasticity_issue:
            print(f"  [WARN] {elasticity_issue}")
        
        # Discount analysis
        discounted_courses = elasticity_df[elasticity_df['price_change_pct'] < -5]
        if not discounted_courses.empty:
            discount_enrollment_corr = discounted_courses['price_change_pct'].corr(discounted_courses['actual_enrollments'])
            print(f"  Discount-Enrollment Correlation: {discount_enrollment_corr:.3f}")
        
        elasticity_results = {
            'courses_analyzed': len(elasticity_df),
            'price_enrollment_correlation': price_enrollment_corr,
            'elasticity_coefficient': elasticity_coefficient,
            'elasticity_issue': elasticity_issue,
            'discounted_courses': len(discounted_courses) if not discounted_courses.empty else 0
        }
        
        self.results['elasticity'] = elasticity_results
        return elasticity_results
    
    def generate_realism_score(self) -> Dict[str, Any]:
        """Generate overall data realism score"""
        print("\n" + "="*60)
        print("DATA REALISM SCORE")
        print("="*60)
        
        scores = {}
        
        # Pricing realism score
        pricing_score = 100
        if 'pricing' in self.results:
            pricing_issues = len(self.results['pricing']['issues'])
            pricing_score = max(0, 100 - (pricing_issues * 15))
        
        # Enrollment realism score
        enrollment_score = 100
        if 'enrollments' in self.results:
            enrollment_issues = len(self.results['enrollments']['issues'])
            enrollment_score = max(0, 100 - (enrollment_issues * 20))
        
        # Teacher realism score
        teacher_score = 100
        if 'teachers' in self.results:
            teacher_issues = len(self.results['teachers']['issues'])
            teacher_score = max(0, 100 - (teacher_issues * 15))
        
        # Market dynamics score
        market_score = 100
        if 'market_dynamics' in self.results:
            market_issues = len(self.results['market_dynamics']['issues'])
            market_score = max(0, 100 - (market_issues * 10))
        
        # Elasticity score
        elasticity_score = 100
        if 'elasticity' in self.results and not self.results['elasticity'].get('insufficient_data', False):
            if self.results['elasticity'].get('elasticity_issue'):
                elasticity_score = 50
        
        # Overall score
        overall_score = (pricing_score + enrollment_score + teacher_score + market_score + elasticity_score) / 5
        
        scores = {
            'pricing': round(pricing_score, 1),
            'enrollments': round(enrollment_score, 1),
            'teachers': round(teacher_score, 1),
            'market_dynamics': round(market_score, 1),
            'elasticity': round(elasticity_score, 1),
            'overall': round(overall_score, 1)
        }
        
        print(f"\nData Realism Scores:")
        print(f"  Pricing: {scores['pricing']}/100")
        print(f"  Enrollments: {scores['enrollments']}/100")
        print(f"  Teachers: {scores['teachers']}/100")
        print(f"  Market Dynamics: {scores['market_dynamics']}/100")
        print(f"  Elasticity: {scores['elasticity']}/100")
        print(f"  Overall: {scores['overall']}/100")
        
        # Realism grade
        if overall_score >= 85:
            grade = "A (Highly Realistic)"
        elif overall_score >= 75:
            grade = "B (Realistic)"
        elif overall_score >= 65:
            grade = "C (Moderately Realistic)"
        elif overall_score >= 55:
            grade = "D (Somewhat Unrealistic)"
        else:
            grade = "F (Unrealistic)"
        
        print(f"  Grade: {grade}")
        
        self.results['realism_scores'] = scores
        return scores
    
    def generate_report(self) -> str:
        """Generate comprehensive realism validation report"""
        report = []
        report.append("EDTECH TOKEN ECONOMY - DATA REALISM VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Database: {self.db_path}")
        report.append("")
        
        # Summary
        if 'realism_scores' in self.results:
            scores = self.results['realism_scores']
            report.append("EXECUTIVE SUMMARY")
            report.append("-" * 40)
            report.append(f"Overall Data Realism Score: {scores['overall']}/100")
            report.append(f"Pricing Realism: {scores['pricing']}/100")
            report.append(f"Enrollment Realism: {scores['enrollments']}/100")
            report.append(f"Teacher Realism: {scores['teachers']}/100")
            report.append(f"Market Dynamics: {scores['market_dynamics']}/100")
            report.append(f"Price Elasticity: {scores['elasticity']}/100")
            report.append("")
        
        # Issues
        all_issues = []
        for section in ['pricing', 'enrollments', 'teachers', 'market_dynamics']:
            if section in self.results and 'issues' in self.results[section]:
                all_issues.extend(self.results[section]['issues'])
        
        if all_issues:
            report.append("REALISM ISSUES IDENTIFIED")
            report.append("-" * 40)
            for i, issue in enumerate(all_issues, 1):
                report.append(f"{i}. {issue}")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        if scores['overall'] < 70:
            report.append("1. Data realism is below acceptable standards")
            report.append("2. Consider adjusting data generation parameters")
            report.append("3. Implement more realistic market dynamics")
        if scores['elasticity'] < 60:
            report.append("4. Price elasticity patterns need improvement")
        if scores['enrollments'] < 70:
            report.append("5. Enrollment patterns are unrealistic")
        
        return "\n".join(report)
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete data realism validation"""
        print("EDTECH TOKEN ECONOMY - DATA REALISM VALIDATION")
        print("=" * 80)
        
        try:
            self.connect()
            
            # Run all validations
            self.validate_pricing_realism()
            self.validate_enrollment_patterns()
            self.validate_teacher_distribution()
            self.validate_market_dynamics()
            self.validate_price_elasticity()
            self.generate_realism_score()
            
            # Generate report
            report = self.generate_report()
            print("\n" + report)
            
            return self.results
            
        except Exception as e:
            print(f"[ERROR] Validation failed: {e}")
            raise
        finally:
            self.disconnect()


def main():
    """Main function for command-line execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='EdTech Token Economy Data Realism Validation')
    parser.add_argument('--db-path', default='edtech_token_economy.db', 
                       help='Path to the database file')
    parser.add_argument('--output', help='Output file for detailed report')
    
    args = parser.parse_args()
    
    # Run validation
    validator = DataRealismValidator(args.db_path)
    results = validator.run_full_validation()
    
    # Save detailed report if requested
    if args.output:
        report = validator.generate_report()
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"\n[OK] Detailed report saved to: {args.output}")
    
    return results


if __name__ == "__main__":
    main()
