#!/usr/bin/env python3
"""
MLflow Model Registry Management Script

This script demonstrates how to use the MLflow Model Registry functionality
for managing model versions, stages, and deployments.

Author: EdTech Token Economy Team
Date: October 2025
"""

import sys
import os

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.mlflow_setup import setup_mlflow_for_edtech_pipeline
import mlflow
from mlflow.tracking import MlflowClient

def list_registered_models():
    """List all registered models"""
    print("\n" + "="*60)
    print("REGISTERED MODELS")
    print("="*60)
    
    try:
        client = MlflowClient()
        models = client.search_registered_models()
        
        if not models:
            print("No registered models found.")
            return
        
        for model in models:
            print(f"\nModel: {model.name}")
            print(f"  Description: {model.description}")
            print(f"  Creation Time: {model.creation_timestamp}")
            print(f"  Last Updated: {model.last_updated_timestamp}")
            
            # Get latest version
            latest_version = client.get_latest_versions(model.name, stages=["None"])
            if latest_version:
                latest = latest_version[0]
                print(f"  Latest Version: {latest.version}")
                print(f"  Stage: {latest.current_stage}")
                print(f"  R² Score: {latest.tags.get('test_r2', 'N/A')}")
                print(f"  Model Type: {latest.tags.get('model_type', 'N/A')}")
    
    except Exception as e:
        print(f"Error listing models: {e}")

def list_model_versions(model_name="EdTech_Token_Elasticity_Model"):
    """List all versions of a specific model"""
    print(f"\n" + "="*60)
    print(f"MODEL VERSIONS: {model_name}")
    print("="*60)
    
    try:
        tracker = setup_mlflow_for_edtech_pipeline()
        versions = tracker.get_model_versions(model_name)
        
        if not versions:
            print(f"No versions found for model: {model_name}")
            return
        
        for version in versions:
            print(f"\nVersion: {version['version']}")
            print(f"  Stage: {version['stage']}")
            print(f"  Created: {version['creation_timestamp']}")
            print(f"  Description: {version['description']}")
            print(f"  Tags: {version['tags']}")
    
    except Exception as e:
        print(f"Error listing versions: {e}")

def load_production_model(model_name="EdTech_Token_Elasticity_Model"):
    """Load and test the production model"""
    print(f"\n" + "="*60)
    print(f"LOADING PRODUCTION MODEL: {model_name}")
    print("="*60)
    
    try:
        tracker = setup_mlflow_for_edtech_pipeline()
        model = tracker.load_production_model(model_name)
        
        if model:
            print("✓ Production model loaded successfully!")
            print(f"  Model type: {type(model).__name__}")
            
            # Test prediction with sample data
            import numpy as np
            sample_features = np.array([[100.0, 4.5, 20, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
            prediction = model.predict(sample_features)
            print(f"  Sample prediction: {prediction[0]:.2f} enrollments")
        else:
            print("✗ Failed to load production model")
    
    except Exception as e:
        print(f"Error loading model: {e}")

def promote_model_to_production(model_name="EdTech_Token_Elasticity_Model", version=None):
    """Promote a model version to production"""
    print(f"\n" + "="*60)
    print(f"PROMOTING MODEL TO PRODUCTION")
    print("="*60)
    
    try:
        client = MlflowClient()
        
        if not version:
            # Get latest version
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                print("No versions found to promote")
                return
            version = versions[0].version
        
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True
        )
        
        print(f"✓ Model version {version} promoted to Production stage")
        print(f"  Previous production versions archived")
    
    except Exception as e:
        print(f"Error promoting model: {e}")

def main():
    """Main function"""
    print("\n" + "="*80)
    print("MLFLOW MODEL REGISTRY MANAGEMENT")
    print("="*80)
    
    # Set tracking URI to server
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # List registered models
    list_registered_models()
    
    # List model versions
    list_model_versions()
    
    # Load production model
    load_production_model()
    
    print("\n" + "="*80)
    print("MODEL REGISTRY MANAGEMENT COMPLETED")
    print("="*80)
    print("\nNext Steps:")
    print("1. View Model Registry in MLflow UI: http://localhost:5000")
    print("2. Compare model versions and performance")
    print("3. Promote best models to Production stage")
    print("4. Use production models in API endpoints")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
