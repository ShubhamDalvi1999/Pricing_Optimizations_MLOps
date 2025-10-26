#!/usr/bin/env python3
"""
EdTech Token Economy Pipeline Orchestrator

Main entry point for running the complete ML pipeline.

Author: EdTech Token Economy Team
Date: October 2025
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline.orchestrator import EdTechPipelineOrchestrator


def main():
    """Main function for command-line execution"""
    print("\n" + "="*80)
    print("🎓 EDTECH TOKEN ECONOMY ML PIPELINE")
    print("="*80 + "\n")
    
    # Create orchestrator
    orchestrator = EdTechPipelineOrchestrator(db_path="edtech_token_economy.db")
    
    # Run complete pipeline
    print("🚀 Starting ML Pipeline...")
    results = orchestrator.run_complete_pipeline(
        generate_data=True,
        n_learners=10000,
        n_teachers=500,
        n_courses=2000,
        n_enrollments=50000
    )
    
    # Print summary
    if results['status'] == 'completed':
        print("\n" + "="*80)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\n⏱️  Total Duration: {results['total_duration_seconds']:.2f} seconds")
        
        # Data Generation Info
        if 'data_generation' in results['stages']:
            dg = results['stages']['data_generation']
            print(f"\n📊 Data Generated:")
            print(f"  - Learners: {dg.get('learners_generated', 0):,}")
            print(f"  - Teachers: {dg.get('teachers_generated', 0):,}")
            print(f"  - Courses: {dg.get('courses_generated', 0):,}")
            print(f"  - Enrollments: {dg.get('enrollments_generated', 0):,}")
        
        # Model Training Info
        if 'model_training' in results['stages']:
            mt = results['stages']['model_training']
            print(f"\n🤖 Models Trained: {mt.get('models_trained', 0)}")
            if mt.get('best_model'):
                print(f"  - Best Model: {mt['best_model']}")
                print(f"  - Best R²: {mt.get('best_model_r2', 0):.3f}")
        
        print("\n📂 Output Locations:")
        print("  - Database: edtech_token_economy.db")
        print("  - Raw Data: data/raw/")
        print("  - Processed Data: data/processed/")
        print("  - Models: models/")
        print("  - Reports: reports/")
        
        print("\n🚀 Next Steps:")
        print("  1. Start API server:")
        print("     cd api && uvicorn main:app --reload")
        print("\n  2. Access API docs:")
        print("     http://localhost:8000/docs")
        print("\n  3. Review reports:")
        print("     cat reports/pipeline_summary.txt")
        print("\n  4. Test API endpoints:")
        print("     curl http://localhost:8000/health")
        print("\n" + "="*80 + "\n")
    
    else:
        print("\n" + "="*80)
        print("❌ PIPELINE FAILED!")
        print("="*80)
        print(f"\nError: {results.get('error', 'Unknown error')}")
        print("\nPlease check the logs for more details.")
        print("="*80 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()


