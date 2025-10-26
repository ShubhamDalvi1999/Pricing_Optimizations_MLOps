"""
EdTech Token Economy Pipeline Orchestrator

This module orchestrates the complete ML pipeline for the EdTech platform:
1. Data Generation
2. Data Understanding
3. Data Preparation  
4. Model Training (Token Price Elasticity)
5. Model Evaluation
6. Model Deployment

Author: EdTech Token Economy Team
Date: October 2025
"""

import sys
import os
import logging
from datetime import datetime
from typing import Dict, Any
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import modules
from src.data.edtech_sources import EdTechTokenEconomyGenerator
from src.data.edtech_database import EdTechDatabaseManager, DatabaseConfig
from src.ml.token_elasticity_modeling import TokenPriceElasticityModeler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('edtech_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EdTechPipelineOrchestrator:
    """Main orchestrator for EdTech Token Economy ML pipeline"""
    
    def __init__(self, db_path: str = "edtech_token_economy.db"):
        """
        Initialize pipeline orchestrator
        
        Args:
            db_path: Path to database file
        """
        self.db_path = db_path
        self.results = {}
        self.start_time = None
        
    def run_complete_pipeline(self, 
                             generate_data: bool = True,
                             n_learners: int = 10000,
                             n_teachers: int = 500,
                             n_courses: int = 2000,
                             n_enrollments: int = 50000) -> Dict[str, Any]:
        """
        Run the complete EdTech ML pipeline
        
        Args:
            generate_data: Whether to generate new data
            n_learners: Number of learners to generate
            n_teachers: Number of teachers to generate
            n_courses: Number of courses to generate
            n_enrollments: Number of enrollments to generate
            
        Returns:
            Dictionary with pipeline results
        """
        self.start_time = datetime.now()
        logger.info("="*80)
        logger.info("🎓 EDTECH TOKEN ECONOMY ML PIPELINE STARTED")
        logger.info("="*80)
        
        try:
            # Stage 1: Data Generation
            if generate_data:
                logger.info("\n[STAGE 1] Data Generation")
                self._stage_data_generation(n_learners, n_teachers, n_courses, n_enrollments)
            else:
                logger.info("\n[STAGE 1] Skipping data generation (using existing data)")
            
            # Stage 2: Data Understanding
            logger.info("\n[STAGE 2] Data Understanding")
            self._stage_data_understanding()
            
            # Stage 3: Data Preparation
            logger.info("\n[STAGE 3] Data Preparation")
            self._stage_data_preparation()
            
            # Stage 4: Model Training
            logger.info("\n[STAGE 4] Token Price Elasticity Model Training")
            self._stage_model_training()
            
            # Stage 5: Model Evaluation
            logger.info("\n[STAGE 5] Model Evaluation")
            self._stage_model_evaluation()
            
            # Stage 6: Generate Reports
            logger.info("\n[STAGE 6] Report Generation")
            self._stage_report_generation()
            
            # Pipeline completed
            total_duration = (datetime.now() - self.start_time).total_seconds()
            
            pipeline_results = {
                'status': 'completed',
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_duration_seconds': total_duration,
                'stages': self.results
            }
            
            logger.info("\n" + "="*80)
            logger.info("✅ EDTECH TOKEN ECONOMY ML PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"⏱️  Total Duration: {total_duration:.2f} seconds")
            logger.info("="*80)
            
            return pipeline_results
        
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return {
                'status': 'failed',
                'error': str(e),
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': datetime.now().isoformat()
            }
    
    def _stage_data_generation(self, n_learners: int, n_teachers: int,
                               n_courses: int, n_enrollments: int):
        """Stage 1: Generate EdTech token economy data"""
        stage_start = datetime.now()
        
        generator = EdTechTokenEconomyGenerator(db_path=self.db_path)
        generator.generate_all_data(
            n_learners=n_learners,
            n_teachers=n_teachers,
            n_courses=n_courses,
            n_enrollments=n_enrollments
        )
        generator.export_to_csv('data/raw')
        generator.close()
        
        self.results['data_generation'] = {
            'status': 'completed',
            'duration': (datetime.now() - stage_start).total_seconds(),
            'learners_generated': n_learners,
            'teachers_generated': n_teachers,
            'courses_generated': n_courses,
            'enrollments_generated': n_enrollments
        }
        
        logger.info(f"✓ Data generation completed in {self.results['data_generation']['duration']:.2f}s")
    
    def _stage_data_understanding(self):
        """Stage 2: Understand the data"""
        stage_start = datetime.now()
        
        config = DatabaseConfig(db_path=self.db_path)
        db_manager = EdTechDatabaseManager(config)
        db_manager.connect()
        
        # Get summary statistics
        learner_summary = db_manager.get_learner_summary()
        teacher_performance = db_manager.get_teacher_performance()
        course_performance = db_manager.get_course_performance()
        category_analysis = db_manager.get_category_analysis()
        
        db_manager.disconnect()
        
        self.results['data_understanding'] = {
            'status': 'completed',
            'duration': (datetime.now() - stage_start).total_seconds(),
            'total_learners': len(learner_summary),
            'total_teachers': len(teacher_performance),
            'total_courses': len(course_performance),
            'total_categories': len(category_analysis)
        }
        
        logger.info(f"✓ Data understanding completed in {self.results['data_understanding']['duration']:.2f}s")
        logger.info(f"  - Learners: {len(learner_summary):,}")
        logger.info(f"  - Teachers: {len(teacher_performance):,}")
        logger.info(f"  - Courses: {len(course_performance):,}")
    
    def _stage_data_preparation(self):
        """Stage 3: Prepare data for modeling"""
        stage_start = datetime.now()
        
        config = DatabaseConfig(db_path=self.db_path)
        db_manager = EdTechDatabaseManager(config)
        db_manager.connect()
        
        # Get price elasticity data
        elasticity_data = db_manager.get_price_elasticity_data()
        
        # Export processed data
        os.makedirs('data/processed', exist_ok=True)
        elasticity_data.to_csv('data/processed/token_price_elasticity_data.csv', index=False)
        
        # Export other analytical datasets
        db_manager.export_all_tables('data/processed')
        
        db_manager.disconnect()
        
        self.results['data_preparation'] = {
            'status': 'completed',
            'duration': (datetime.now() - stage_start).total_seconds(),
            'elasticity_data_shape': elasticity_data.shape,
            'elasticity_data_rows': len(elasticity_data),
            'elasticity_data_features': len(elasticity_data.columns)
        }
        
        logger.info(f"✓ Data preparation completed in {self.results['data_preparation']['duration']:.2f}s")
        logger.info(f"  - Elasticity data: {elasticity_data.shape}")
    
    def _stage_model_training(self):
        """Stage 4: Train token price elasticity models"""
        stage_start = datetime.now()
        
        # Load prepared data
        import pandas as pd
        elasticity_data = pd.read_csv('data/processed/token_price_elasticity_data.csv')
        
        # Train models
        modeler = TokenPriceElasticityModeler(target_column='total_enrollments')
        model_results = modeler.compare_models(elasticity_data)
        
        # Save models
        os.makedirs('models', exist_ok=True)
        import pickle
        
        for model_name, result in model_results.items():
            model_file = f'models/elasticity_{model_name}.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump({
                    'model': result.model,
                    'metrics': result.metrics,
                    'feature_importance': result.feature_importance,
                    'parameters': result.parameters
                }, f)
            logger.info(f"  - Saved {model_name} to {model_file}")
        
        # Store results
        model_metrics = {
            name: {
                'test_r2': result.metrics['test_r2'],
                'test_rmse': result.metrics['test_rmse'],
                'price_elasticity': result.metrics['price_elasticity'],
                'training_time': result.training_time
            }
            for name, result in model_results.items()
        }
        
        self.results['model_training'] = {
            'status': 'completed',
            'duration': (datetime.now() - stage_start).total_seconds(),
            'models_trained': len(model_results),
            'model_metrics': model_metrics,
            'best_model': modeler.best_model.model_name if modeler.best_model else None,
            'best_model_r2': modeler.best_model.metrics['test_r2'] if modeler.best_model else None
        }
        
        logger.info(f"✓ Model training completed in {self.results['model_training']['duration']:.2f}s")
        logger.info(f"  - Models trained: {len(model_results)}")
        if modeler.best_model:
            logger.info(f"  - Best model: {modeler.best_model.model_name} (R² = {modeler.best_model.metrics['test_r2']:.3f})")
    
    def _stage_model_evaluation(self):
        """Stage 5: Evaluate models"""
        stage_start = datetime.now()
        
        # Model evaluation metrics already captured in training stage
        # This stage can be extended for additional evaluation
        
        self.results['model_evaluation'] = {
            'status': 'completed',
            'duration': (datetime.now() - stage_start).total_seconds(),
            'evaluation_notes': 'Models evaluated based on R², RMSE, and price elasticity coefficient'
        }
        
        logger.info(f"✓ Model evaluation completed in {self.results['model_evaluation']['duration']:.2f}s")
    
    def _stage_report_generation(self):
        """Stage 6: Generate reports"""
        stage_start = datetime.now()
        
        os.makedirs('reports', exist_ok=True)
        
        # Save pipeline results
        results_file = 'reports/pipeline_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Generate summary report
        summary_file = 'reports/pipeline_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("EDTECH TOKEN ECONOMY ML PIPELINE SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Pipeline Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for stage_name, stage_results in self.results.items():
                f.write(f"{stage_name.upper().replace('_', ' ')}:\n")
                f.write("-"*50 + "\n")
                for key, value in stage_results.items():
                    if key != 'model_metrics':
                        f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            f.write("="*80 + "\n")
        
        self.results['report_generation'] = {
            'status': 'completed',
            'duration': (datetime.now() - stage_start).total_seconds(),
            'reports_generated': [results_file, summary_file]
        }
        
        logger.info(f"✓ Report generation completed in {self.results['report_generation']['duration']:.2f}s")
        logger.info(f"  - Reports saved to reports/")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🎓 EdTech Token Economy ML Pipeline")
    print("="*80 + "\n")
    
    # Create orchestrator
    orchestrator = EdTechPipelineOrchestrator()
    
    # Run pipeline
    results = orchestrator.run_complete_pipeline(
        generate_data=True,
        n_learners=10000,
        n_teachers=500,
        n_courses=2000,
        n_enrollments=50000
    )
    
    if results['status'] == 'completed':
        print("\n✅ Pipeline completed successfully!")
        print(f"⏱️  Total duration: {results['total_duration_seconds']:.2f} seconds")
        print("\n📊 Next Steps:")
        print("1. Start API server: cd api && uvicorn main:app --reload")
        print("2. Review reports in reports/ directory")
        print("3. Check trained models in models/ directory")
    else:
        print("\n❌ Pipeline failed!")
        print(f"Error: {results.get('error', 'Unknown error')}")


