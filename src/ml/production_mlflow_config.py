"""
Production MLflow Configuration for EdTech Token Economy

This module provides production-ready MLflow setup with:
- Database backend for reliability
- Version compatibility checks
- Health monitoring
- Migration utilities

Author: EdTech Token Economy Pipeline
Date: October 2025
"""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import logging
import os
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any
import warnings

logger = logging.getLogger(__name__)

class ProductionMLFlowConfig:
    """Production-ready MLflow configuration with database backend"""
    
    def __init__(self, 
                 experiment_name: str = "EdTech_Token_Economy_Production",
                 use_database: bool = True,
                 db_path: str = "./mlflow_production.db"):
        """
        Initialize production MLflow configuration
        
        Args:
            experiment_name: Name of the MLFlow experiment
            use_database: Whether to use database backend (recommended for production)
            db_path: Path to SQLite database file
        """
        self.experiment_name = experiment_name
        self.use_database = use_database
        self.db_path = db_path
        
        # Check MLflow version compatibility
        self._check_mlflow_version()
        
        # Setup tracking URI
        if use_database:
            self.tracking_uri = f"sqlite:///{db_path}"
            self._setup_database()
        else:
            self.tracking_uri = "file:./mlruns"
            
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Initialize experiment
        self._setup_experiment()
        
    def _check_mlflow_version(self):
        """Check MLflow version compatibility"""
        mlflow_version = mlflow.__version__
        major_version = int(mlflow_version.split('.')[0])
        
        if major_version >= 3:
            warnings.warn(
                f"MLflow {mlflow_version} detected. Version 3.x has breaking changes. "
                "Consider using MLflow 2.8.1 for production stability.",
                UserWarning
            )
            
        logger.info(f"MLflow version: {mlflow_version}")
        
    def _setup_database(self):
        """Setup SQLite database for MLflow"""
        try:
            # Create database directory if it doesn't exist
            db_dir = Path(self.db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            
            # Test database connection
            conn = sqlite3.connect(self.db_path)
            conn.close()
            
            logger.info(f"Database backend configured: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to setup database backend: {e}")
            raise
            
    def _setup_experiment(self):
        """Setup MLflow experiment with production tags"""
        try:
            # Check if experiment exists
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            
            if experiment:
                self.experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {self.experiment_name}")
            else:
                # Create new experiment with production tags
                production_tags = {
                    "project": "EdTech Token Economy Platform",
                    "pipeline_version": "1.0",
                    "environment": "production",
                    "domain": "education_technology",
                    "mlflow_version": mlflow.__version__,
                    "backend_type": "database" if self.use_database else "file"
                }
                
                self.experiment_id = mlflow.create_experiment(
                    self.experiment_name,
                    tags=production_tags
                )
                logger.info(f"Created new production experiment: {self.experiment_name}")
                
            # Set as active experiment
            mlflow.set_experiment(self.experiment_name)
            
        except Exception as e:
            logger.error(f"Failed to setup experiment: {e}")
            raise
            
    def health_check(self) -> Dict[str, Any]:
        """Perform MLflow health check"""
        health_status = {
            "status": "healthy",
            "mlflow_version": mlflow.__version__,
            "tracking_uri": self.tracking_uri,
            "experiment_id": self.experiment_id,
            "database_backend": self.use_database,
            "errors": []
        }
        
        try:
            # Test MLflow client
            client = MlflowClient()
            experiments = client.search_experiments()
            
            # Test experiment access
            experiment = client.get_experiment(self.experiment_id)
            
            logger.info("MLflow health check passed")
            
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["errors"].append(str(e))
            logger.error(f"MLflow health check failed: {e}")
            
        return health_status
        
    def migrate_from_file_store(self, old_mlruns_path: str):
        """Migrate runs from file store to database backend"""
        if not self.use_database:
            raise ValueError("Migration only supported with database backend")
            
        logger.info(f"Starting migration from {old_mlruns_path}")
        
        try:
            # This would implement the migration logic
            # For now, just log the intention
            logger.info("Migration functionality would be implemented here")
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise

def setup_production_mlflow(experiment_name: str = "EdTech_Token_Economy_Production") -> ProductionMLFlowConfig:
    """
    Setup production-ready MLflow configuration
    
    Args:
        experiment_name: Name of the MLFlow experiment
        
    Returns:
        Configured ProductionMLFlowConfig
    """
    config = ProductionMLFlowConfig(experiment_name=experiment_name)
    
    # Perform health check
    health = config.health_check()
    if health["status"] != "healthy":
        raise RuntimeError(f"MLflow setup failed: {health['errors']}")
        
    logger.info("Production MLflow setup completed successfully")
    return config

# Example usage
if __name__ == "__main__":
    print("Testing Production MLflow Configuration...")
    
    try:
        config = setup_production_mlflow("EdTech_Production_Test")
        
        print(f"[SUCCESS] Production MLflow configured")
        print(f"[SUCCESS] Tracking URI: {config.tracking_uri}")
        print(f"[SUCCESS] Experiment ID: {config.experiment_id}")
        
        # Health check
        health = config.health_check()
        print(f"[SUCCESS] Health Status: {health['status']}")
        
        print("\n[SUCCESS] Production MLflow test completed!")
        
    except Exception as e:
        print(f"[ERROR] Production MLflow setup failed: {e}")
