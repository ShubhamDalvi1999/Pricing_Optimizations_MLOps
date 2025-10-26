#!/usr/bin/env python3
"""
Master Data Assessment Script for EdTech Token Economy

This script runs comprehensive data quality, realism, and model performance assessments:
- Data Quality Assessment
- Data Realism Validation  
- Model Performance Analysis
- Integrated reporting and recommendations

Author: EdTech Token Economy Team
Version: 1.0.0
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import argparse

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))

from data_quality_assessment import DataQualityAssessment
from data_realism_validation import DataRealismValidator
from model_performance_analysis import ModelPerformanceAnalyzer

class MasterDataAssessment:
    """Master class for comprehensive data assessment"""
    
    def __init__(self, db_path: str = "edtech_token_economy.db", models_dir: str = "models"):
        """Initialize the master assessment"""
        self.db_path = db_path
        self.models_dir = models_dir
        self.results = {}
        self.overall_issues = []
        
    def run_data_quality_assessment(self):
        """Run data quality assessment"""
        print("\n" + "="*80)
        print("RUNNING DATA QUALITY ASSESSMENT")
        print("="*80)
        
        try:
            assessor = DataQualityAssessment(self.db_path)
            quality_results = assessor.run_full_assessment()
            self.results['data_quality'] = quality_results
            self.overall_issues.extend(assessor.issues)
            return True
        except Exception as e:
            print(f"[ERROR] Data quality assessment failed: {e}")
            self.overall_issues.append(f"Data quality assessment failed: {e}")
            return False
    
    def run_data_realism_validation(self):
        """Run data realism validation"""
        print("\n" + "="*80)
        print("RUNNING DATA REALISM VALIDATION")
        print("="*80)
        
        try:
            validator = DataRealismValidator(self.db_path)
            realism_results = validator.run_full_validation()
            self.results['data_realism'] = realism_results
            return True
        except Exception as e:
            print(f"[ERROR] Data realism validation failed: {e}")
            self.overall_issues.append(f"Data realism validation failed: {e}")
            return False
    
    def run_model_performance_analysis(self):
        """Run model performance analysis"""
        print("\n" + "="*80)
        print("RUNNING MODEL PERFORMANCE ANALYSIS")
        print("="*80)
        
        try:
            analyzer = ModelPerformanceAnalyzer(self.models_dir)
            performance_results = analyzer.run_full_analysis()
            self.results['model_performance'] = performance_results
            self.overall_issues.extend(analyzer.performance_issues)
            return True
        except Exception as e:
            print(f"[ERROR] Model performance analysis failed: {e}")
            self.overall_issues.append(f"Model performance analysis failed: {e}")
            return False
    
    def generate_integrated_report(self) -> str:
        """Generate integrated assessment report"""
        report = []
        report.append("EDTECH TOKEN ECONOMY - COMPREHENSIVE DATA ASSESSMENT REPORT")
        report.append("=" * 100)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Database: {self.db_path}")
        report.append(f"Models Directory: {self.models_dir}")
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 50)
        
        # Data Quality Summary
        if 'data_quality' in self.results and 'quality_scores' in self.results['data_quality']:
            quality_scores = self.results['data_quality']['quality_scores']
            report.append(f"Data Quality Score: {quality_scores['overall']}/100")
            report.append(f"  - Completeness: {quality_scores['completeness']}/100")
            report.append(f"  - Validity: {quality_scores['validity']}/100")
            report.append(f"  - Integrity: {quality_scores['integrity']}/100")
        else:
            report.append("Data Quality Score: Not Available")
        
        # Data Realism Summary
        if 'data_realism' in self.results and 'realism_scores' in self.results['data_realism']:
            realism_scores = self.results['data_realism']['realism_scores']
            report.append(f"Data Realism Score: {realism_scores['overall']}/100")
            report.append(f"  - Pricing: {realism_scores['pricing']}/100")
            report.append(f"  - Enrollments: {realism_scores['enrollments']}/100")
            report.append(f"  - Teachers: {realism_scores['teachers']}/100")
            report.append(f"  - Market Dynamics: {realism_scores['market_dynamics']}/100")
            report.append(f"  - Elasticity: {realism_scores['elasticity']}/100")
        else:
            report.append("Data Realism Score: Not Available")
        
        # Model Performance Summary
        if 'model_performance' in self.results and 'performance_scores' in self.results['model_performance']:
            performance_scores = self.results['model_performance']['performance_scores']
            report.append(f"Model Performance Score: {performance_scores['overall']}/100")
            report.append(f"  - Performance: {performance_scores['performance']}/100")
            report.append(f"  - Stability: {performance_scores['stability']}/100")
            report.append(f"  - Features: {performance_scores['features']}/100")
        else:
            report.append("Model Performance Score: Not Available")
        
        report.append("")
        
        # Overall Assessment
        report.append("OVERALL ASSESSMENT")
        report.append("-" * 50)
        
        # Calculate overall score
        overall_scores = []
        if 'data_quality' in self.results and 'quality_scores' in self.results['data_quality']:
            overall_scores.append(self.results['data_quality']['quality_scores']['overall'])
        if 'data_realism' in self.results and 'realism_scores' in self.results['data_realism']:
            overall_scores.append(self.results['data_realism']['realism_scores']['overall'])
        if 'model_performance' in self.results and 'performance_scores' in self.results['model_performance']:
            overall_scores.append(self.results['model_performance']['performance_scores']['overall'])
        
        if overall_scores:
            overall_score = sum(overall_scores) / len(overall_scores)
            report.append(f"Overall System Score: {overall_score:.1f}/100")
            
            if overall_score >= 85:
                grade = "A (Excellent)"
                status = "READY FOR PRODUCTION"
            elif overall_score >= 75:
                grade = "B (Good)"
                status = "READY WITH MINOR IMPROVEMENTS"
            elif overall_score >= 65:
                grade = "C (Fair)"
                status = "NEEDS IMPROVEMENTS"
            elif overall_score >= 55:
                grade = "D (Poor)"
                status = "SIGNIFICANT ISSUES"
            else:
                grade = "F (Very Poor)"
                status = "NOT READY FOR PRODUCTION"
            
            report.append(f"Grade: {grade}")
            report.append(f"Status: {status}")
        else:
            report.append("Overall System Score: Not Available")
        
        report.append("")
        
        # Critical Issues
        if self.overall_issues:
            report.append("CRITICAL ISSUES")
            report.append("-" * 50)
            for i, issue in enumerate(self.overall_issues, 1):
                report.append(f"{i}. {issue}")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 50)
        
        recommendations = []
        
        # Data Quality Recommendations
        if 'data_quality' in self.results and 'quality_scores' in self.results['data_quality']:
            quality_score = self.results['data_quality']['quality_scores']['overall']
            if quality_score < 70:
                recommendations.append("1. Improve data quality - address missing values and inconsistencies")
        
        # Data Realism Recommendations
        if 'data_realism' in self.results and 'realism_scores' in self.results['data_realism']:
            realism_score = self.results['data_realism']['realism_scores']['overall']
            if realism_score < 70:
                recommendations.append("2. Improve data realism - adjust generation parameters for more realistic patterns")
        
        # Model Performance Recommendations
        if 'model_performance' in self.results and 'performance_scores' in self.results['model_performance']:
            performance_score = self.results['model_performance']['performance_scores']['overall']
            if performance_score < 70:
                recommendations.append("3. Improve model performance - retrain with better data and features")
        
        # General Recommendations
        recommendations.extend([
            "4. Implement continuous monitoring of data quality and model performance",
            "5. Set up automated alerts for data quality degradation",
            "6. Establish data governance policies and procedures",
            "7. Create regular data quality dashboards",
            "8. Implement A/B testing for model improvements"
        ])
        
        for i, rec in enumerate(recommendations, 1):
            report.append(f"{i}. {rec}")
        
        report.append("")
        
        # Next Steps
        report.append("NEXT STEPS")
        report.append("-" * 50)
        
        if overall_scores and overall_score < 70:
            report.append("1. Address critical issues identified in this report")
            report.append("2. Regenerate data with improved parameters")
            report.append("3. Retrain models with better data quality")
            report.append("4. Re-run this assessment after improvements")
        else:
            report.append("1. System is ready for production deployment")
            report.append("2. Implement monitoring and alerting systems")
            report.append("3. Set up regular assessment schedules")
            report.append("4. Plan for continuous improvement")
        
        return "\n".join(report)
    
    def run_comprehensive_assessment(self, skip_quality=False, skip_realism=False, skip_performance=False):
        """Run comprehensive assessment"""
        print("EDTECH TOKEN ECONOMY - COMPREHENSIVE DATA ASSESSMENT")
        print("=" * 100)
        print(f"Database: {self.db_path}")
        print(f"Models Directory: {self.models_dir}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        success_count = 0
        total_assessments = 3
        
        # Run assessments
        if not skip_quality:
            if self.run_data_quality_assessment():
                success_count += 1
        else:
            print("\nSkipping data quality assessment")
        
        if not skip_realism:
            if self.run_data_realism_validation():
                success_count += 1
        else:
            print("\nSkipping data realism validation")
        
        if not skip_performance:
            if self.run_model_performance_analysis():
                success_count += 1
        else:
            print("\nSkipping model performance analysis")
        
        # Generate integrated report
        print("\n" + "="*100)
        print("GENERATING INTEGRATED REPORT")
        print("="*100)
        
        report = self.generate_integrated_report()
        print(report)
        
        print(f"\nAssessment completed: {success_count}/{total_assessments} assessments successful")
        
        return self.results


def main():
    """Main function for command-line execution"""
    parser = argparse.ArgumentParser(description='EdTech Token Economy Comprehensive Data Assessment')
    parser.add_argument('--db-path', default='edtech_token_economy.db', 
                       help='Path to the database file')
    parser.add_argument('--models-dir', default='models', 
                       help='Directory containing model files')
    parser.add_argument('--output', help='Output file for detailed report')
    parser.add_argument('--skip-quality', action='store_true', 
                       help='Skip data quality assessment')
    parser.add_argument('--skip-realism', action='store_true', 
                       help='Skip data realism validation')
    parser.add_argument('--skip-performance', action='store_true', 
                       help='Skip model performance analysis')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.db_path):
        print(f"[ERROR] Database file not found: {args.db_path}")
        return 1
    
    if not os.path.exists(args.models_dir):
        print(f"[ERROR] Models directory not found: {args.models_dir}")
        return 1
    
    # Run assessment
    assessor = MasterDataAssessment(args.db_path, args.models_dir)
    results = assessor.run_comprehensive_assessment(
        skip_quality=args.skip_quality,
        skip_realism=args.skip_realism,
        skip_performance=args.skip_performance
    )
    
    # Save detailed report if requested
    if args.output:
        report = assessor.generate_integrated_report()
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"\n[OK] Detailed report saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
