#!/usr/bin/env python3
"""
Quick Data Assessment Runner

Simple script to run data quality and realism assessments quickly.

Usage:
    python run_assessment.py                    # Run all assessments
    python run_assessment.py --quick           # Run quick assessment only
    python run_assessment.py --quality-only     # Run only data quality
    python run_assessment.py --realism-only    # Run only data realism
    python run_assessment.py --models-only     # Run only model analysis
"""

import sys
import os
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))

def run_quick_assessment():
    """Run a quick assessment focusing on critical issues"""
    print("Running Quick Data Assessment...")
    
    try:
        from comprehensive_assessment import MasterDataAssessment
        
        assessor = MasterDataAssessment()
        
        # Run only data quality and realism (skip model performance for speed)
        assessor.run_data_quality_assessment()
        assessor.run_data_realism_validation()
        
        # Generate quick report
        report = assessor.generate_integrated_report()
        print("\n" + "="*80)
        print("QUICK ASSESSMENT RESULTS")
        print("="*80)
        print(report)
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Quick assessment failed: {e}")
        return False

def run_full_assessment():
    """Run complete assessment including model performance"""
    print("Running Full Data Assessment...")
    
    try:
        from comprehensive_assessment import MasterDataAssessment
        
        assessor = MasterDataAssessment()
        results = assessor.run_comprehensive_assessment()
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Full assessment failed: {e}")
        return False

def run_quality_only():
    """Run only data quality assessment"""
    print("Running Data Quality Assessment Only...")
    
    try:
        from data_quality_assessment import DataQualityAssessment
        
        assessor = DataQualityAssessment()
        results = assessor.run_full_assessment()
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Data quality assessment failed: {e}")
        return False

def run_realism_only():
    """Run only data realism validation"""
    print("Running Data Realism Validation Only...")
    
    try:
        from data_realism_validation import DataRealismValidator
        
        validator = DataRealismValidator()
        results = validator.run_full_validation()
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Data realism validation failed: {e}")
        return False

def run_models_only():
    """Run only model performance analysis"""
    print("Running Model Performance Analysis Only...")
    
    try:
        from model_performance_analysis import ModelPerformanceAnalyzer
        
        analyzer = ModelPerformanceAnalyzer()
        results = analyzer.run_full_analysis()
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Model performance analysis failed: {e}")
        return False

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick Data Assessment Runner')
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick assessment (skip model analysis)')
    parser.add_argument('--quality-only', action='store_true', 
                       help='Run only data quality assessment')
    parser.add_argument('--realism-only', action='store_true', 
                       help='Run only data realism validation')
    parser.add_argument('--models-only', action='store_true', 
                       help='Run only model performance analysis')
    
    args = parser.parse_args()
    
    # Check if database exists
    if not os.path.exists('edtech_token_economy.db'):
        print("[ERROR] Database file 'edtech_token_economy.db' not found")
        print("  Please run the pipeline first to generate data")
        return 1
    
    # Run appropriate assessment
    success = False
    
    if args.quality_only:
        success = run_quality_only()
    elif args.realism_only:
        success = run_realism_only()
    elif args.models_only:
        success = run_models_only()
    elif args.quick:
        success = run_quick_assessment()
    else:
        success = run_full_assessment()
    
    if success:
        print("\n[OK] Assessment completed successfully")
        return 0
    else:
        print("\n[ERROR] Assessment failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
