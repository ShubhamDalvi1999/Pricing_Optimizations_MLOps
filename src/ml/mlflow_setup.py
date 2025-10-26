"""
MLFlow Setup and Model Tracking - EdTech Token Economy

This module handles MLFlow integration for experiment tracking,
model versioning, and performance monitoring in the EdTech Token Economy platform.

Author: EdTech Token Economy Pipeline
Date: October 2025
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime
import os
import json
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EdTechMLFlowTracker:
    """Main class for MLFlow experiment tracking and model management in EdTech platform"""

    def __init__(self, experiment_name: str = "EdTech_Token_Economy",
                 tracking_uri: str = "file:./mlruns"):
        """
        Initialize MLFlow tracker for EdTech

        Args:
            experiment_name: Name of the MLFlow experiment
            tracking_uri: MLFlow tracking URI
        """
        self.experiment_name = experiment_name
        # Convert Windows path to file:// URI format for MLFlow compatibility
        if tracking_uri and not tracking_uri.startswith(('file:', 'http:', 'https:', 'sqlite:', 'mysql:', 'postgresql:')):
            import pathlib
            abs_path = pathlib.Path(tracking_uri).resolve()
            tracking_uri = f"file:///{str(abs_path).replace(os.sep, '/')}"
        self.tracking_uri = tracking_uri
        self.client = None
        self.experiment_id = None

        # Set MLFlow tracking URI
        mlflow.set_tracking_uri(tracking_uri)

        # Initialize experiment
        self._setup_experiment()

    def _setup_experiment(self):
        """Set up MLFlow experiment"""
        try:
            # Check if experiment exists
            experiment = mlflow.get_experiment_by_name(self.experiment_name)

            if experiment:
                self.experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {self.experiment_name}")
            else:
                # Create new experiment with tags and description
                experiment_tags = {
                    "project": "EdTech Token Economy Platform",
                    "pipeline_version": "1.0",
                    "environment": "development",
                    "domain": "education_technology"
                }
                self.experiment_id = mlflow.create_experiment(
                    self.experiment_name,
                    tags=experiment_tags
                )
                logger.info(f"Created new experiment: {self.experiment_name}")

            # Set as active experiment
            mlflow.set_experiment(self.experiment_name)

            # Initialize client
            try:
                self.client = MlflowClient()
                
                # Set experiment description
                if self.client and self.experiment_id:
                    experiment_description = self._get_experiment_description()
                    try:
                        self.client.set_experiment_tag(
                            self.experiment_id,
                            "mlflow.note.content",
                            experiment_description
                        )
                    except Exception as desc_error:
                        logger.debug(f"Could not set experiment description: {desc_error}")
                        
            except Exception as client_error:
                logger.warning(f"MLFlow client initialization warning: {client_error}")
                self.client = None

        except Exception as e:
            logger.error(f"Failed to setup MLFlow experiment: {e}")
            self.experiment_id = None
            self.client = None
    
    def _get_experiment_description(self) -> str:
        """Get description for the experiment"""
        return """
# EdTech Token Economy ML Pipeline

This experiment tracks all model training runs for the EdTech Token Economy platform.

## Model Types:
- **Token Price Elasticity Models**: Linear, GAM, Polynomial, Random Forest, Gradient Boosting, XGBoost
- **Enrollment Propensity Models**: Logistic Regression, Random Forest, XGBoost, LightGBM
- **Course Completion Models**: Classification and regression models
- **Teacher Quality Models**: Scoring and ranking models
- **Dynamic Pricing Models**: Revenue optimization models
- **Churn Prediction Models**: Student retention models

## Data Sources:
- Learner profiles (demographics, preferences, history)
- Teacher profiles (experience, ratings, specializations)
- Course data (pricing, content, difficulty, reviews)
- Enrollment data (transactions, completions, ratings)
- Platform metrics (engagement, revenue, growth)

## Pipeline Stages:
1. Data Generation & Preparation
2. Feature Engineering (token pricing, enrollment patterns, quality scores)
3. Model Training & Evaluation
4. Business Analysis (ROI, LTV, token economy metrics)
5. Model Deployment via API

## Metrics Tracked:
- **Regression**: R², MAE, RMSE, MAPE, Price Elasticity
- **Classification**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Business**: Token Revenue, Enrollment Rate, Teacher Earnings, Platform Commission, LTV

## Tags:
Use tags to filter runs by model_type, model_name, category, and training_type.
        """.strip()
    
    def _generate_run_description(self, model_type: str, model_name: str, 
                                  dataset_info: Dict[str, Any], 
                                  model_results: Any) -> str:
        """Generate a rich description for the MLflow run"""
        
        # Model type descriptions
        model_descriptions = {
            "linear": "Linear Regression model for token price elasticity analysis.",
            "polynomial_deg2": "Polynomial Regression (degree 2) capturing non-linear pricing relationships.",
            "polynomial_deg3": "Polynomial Regression (degree 3) modeling complex pricing dynamics.",
            "random_forest": "Random Forest ensemble for robust enrollment prediction.",
            "gradient_boosting": "Gradient Boosting for sequential error minimization in elasticity modeling.",
            "xgboost": "XGBoost for high-performance price elasticity and propensity modeling.",
            "gam": "Generalized Additive Model using smooth functions for each feature.",
            "logistic_regression": "Logistic Regression for enrollment propensity classification.",
            "lightgbm": "LightGBM for efficient large-scale enrollment prediction.",
        }
        
        model_desc = model_descriptions.get(model_name, f"{model_name} machine learning model")
        
        # Build description based on model type
        if model_type == "token_elasticity":
            description = f"""# Token Price Elasticity Model: {model_name.replace('_', ' ').title()}

## Model Description
{model_desc}

## Purpose
Predicting token price elasticity to optimize course pricing and maximize platform revenue.

## Dataset Information
"""
            if dataset_info:
                description += f"""- **Training Courses**: {dataset_info.get('rows', 'N/A')}
- **Features**: {dataset_info.get('features', 'N/A')}
- **Target Variable**: `{dataset_info.get('target', 'total_enrollments')}`
"""
                if isinstance(dataset_info.get('features'), list):
                    description += f"- **Feature Count**: {len(dataset_info['features'])}\n"
                    description += f"- **Key Features**: {', '.join(dataset_info['features'][:5])}\n"
            
            if hasattr(model_results, 'metrics') and model_results.metrics:
                metrics = model_results.metrics
                description += "\n## Model Performance\n"
                
                for metric_name in ['train_r2', 'test_r2', 'mae', 'rmse']:
                    if metric_name in metrics:
                        value = metrics[metric_name]
                        value_str = f"{value:.4f}" if isinstance(value, (int, float)) else 'N/A'
                        description += f"- **{metric_name.upper()}**: {value_str}\n"
                
                if 'price_elasticity' in metrics:
                    elasticity = metrics['price_elasticity']
                    if isinstance(elasticity, (int, float)):
                        description += f"""
## Token Price Elasticity Analysis
- **Elasticity Coefficient**: {elasticity:.4f}
- **Interpretation**: {"Elastic (>1)" if abs(elasticity) > 1 else "Inelastic (<1)"}
- **Impact**: {"Token price increase decreases enrollments" if elasticity < 0 else "Token price increase increases enrollments"}
"""
        
        elif model_type == "enrollment_propensity":
            description = f"""# Enrollment Propensity Model: {model_name.replace('_', ' ').title()}

## Model Description
{model_desc}

## Purpose
Predicting learner enrollment probability to optimize marketing and personalization strategies.

## Dataset Information
"""
            if dataset_info:
                description += f"""- **Training Samples**: {dataset_info.get('rows', 'N/A')}
- **Features**: {dataset_info.get('features', 'N/A')}
- **Target Variable**: Enrollment probability (0-1)
"""
            
            if hasattr(model_results, 'metrics') and model_results.metrics:
                metrics = model_results.metrics
                description += "\n## Model Performance\n"
                
                for metric_name in ['test_accuracy', 'f1_score', 'precision', 'recall', 'auc']:
                    if metric_name in metrics:
                        value = metrics[metric_name]
                        value_str = f"{value:.4f}" if isinstance(value, (int, float)) else 'N/A'
                        description += f"- **{metric_name.replace('_', ' ').title()}**: {value_str}\n"
        
        else:
            description = f"""# {model_type.replace('_', ' ').title()}: {model_name.replace('_', ' ').title()}

## Model Description
{model_desc}

## Dataset Information
"""
            if dataset_info:
                description += f"""- **Training Samples**: {dataset_info.get('rows', 'N/A')}
- **Features**: {dataset_info.get('features', 'N/A')}
"""
        
        description += f"""
## Training Information
- **Training Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Platform**: EdTech Token Economy v1.0
"""
        
        return description
    
    def _log_evaluation_artifacts(self, model_results: Any, model_type: str, 
                                  model_name: str, dataset_info: Dict[str, Any] = None):
        """Log comprehensive evaluation artifacts for the model"""
        import tempfile
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 8)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            
            # 1. Log predictions sample
            if hasattr(model_results, 'predictions') and len(model_results.predictions) > 0:
                predictions_path = os.path.join(tmpdir, "predictions_sample.csv")
                
                pred_df = pd.DataFrame({
                    'prediction': model_results.predictions[:100]
                })
                
                if hasattr(model_results, 'y_test') and model_results.y_test is not None:
                    pred_df['actual'] = model_results.y_test[:100]
                    pred_df['error'] = pred_df['actual'] - pred_df['prediction']
                    pred_df['absolute_error'] = abs(pred_df['error'])
                
                pred_df.to_csv(predictions_path, index=False)
                mlflow.log_artifact(predictions_path, "evaluation")
            
            # 2. Regression-specific evaluations
            if model_type in ['token_elasticity', 'regression'] and hasattr(model_results, 'y_test'):
                try:
                    y_test = model_results.y_test
                    y_pred = model_results.predictions
                    
                    # Actual vs Predicted plot
                    plt.figure(figsize=(10, 6))
                    plt.scatter(y_test, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
                    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                            'r--', lw=2, label='Perfect Prediction')
                    plt.xlabel('Actual Enrollments')
                    plt.ylabel('Predicted Enrollments')
                    plt.title(f'Token Elasticity: Actual vs Predicted - {model_name.replace("_", " ").title()}')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    pred_plot_path = os.path.join(tmpdir, "actual_vs_predicted.png")
                    plt.savefig(pred_plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    mlflow.log_artifact(pred_plot_path, "evaluation")
                    
                    # Residuals plot
                    residuals = y_test - y_pred
                    plt.figure(figsize=(10, 6))
                    plt.scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
                    plt.axhline(y=0, color='r', linestyle='--', lw=2)
                    plt.xlabel('Predicted Values')
                    plt.ylabel('Residuals')
                    plt.title(f'Residual Plot - {model_name.replace("_", " ").title()}')
                    plt.grid(True, alpha=0.3)
                    residual_path = os.path.join(tmpdir, "residuals_plot.png")
                    plt.savefig(residual_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    mlflow.log_artifact(residual_path, "evaluation")
                    
                except Exception as e:
                    logger.warning(f"Could not create regression evaluation plots: {e}")
            
            # 3. Feature Importance Plot
            if hasattr(model_results, 'feature_importance') and model_results.feature_importance:
                try:
                    sorted_features = sorted(model_results.feature_importance.items(),
                                           key=lambda x: abs(x[1]), reverse=True)[:15]
                    
                    features, importances = zip(*sorted_features)
                    
                    plt.figure(figsize=(10, 8))
                    plt.barh(range(len(features)), importances)
                    plt.yticks(range(len(features)), features)
                    plt.xlabel('Importance')
                    plt.title(f'Top 15 Feature Importance - {model_name.replace("_", " ").title()}')
                    plt.tight_layout()
                    plt.grid(True, alpha=0.3, axis='x')
                    fi_path = os.path.join(tmpdir, "feature_importance.png")
                    plt.savefig(fi_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    mlflow.log_artifact(fi_path, "evaluation")
                    
                    # Save as CSV
                    fi_df = pd.DataFrame(sorted_features, columns=['feature', 'importance'])
                    fi_csv_path = os.path.join(tmpdir, "feature_importance.csv")
                    fi_df.to_csv(fi_csv_path, index=False)
                    mlflow.log_artifact(fi_csv_path, "evaluation")
                    
                except Exception as e:
                    logger.warning(f"Could not create feature importance plot: {e}")
            
            # 4. Model Evaluation Summary
            summary_path = os.path.join(tmpdir, "evaluation_summary.txt")
            with open(summary_path, 'w') as f:
                f.write(f"EdTech Token Economy Model Evaluation\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Model Type: {model_type}\n")
                f.write(f"Model Name: {model_name}\n")
                f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                if dataset_info:
                    f.write("Dataset Information:\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"  Rows: {dataset_info.get('rows', 'N/A')}\n")
                    f.write(f"  Columns: {dataset_info.get('columns', 'N/A')}\n")
                    f.write(f"  Features: {dataset_info.get('features', 'N/A')}\n\n")
                
                if hasattr(model_results, 'metrics') and model_results.metrics:
                    f.write("Performance Metrics:\n")
                    f.write("-" * 80 + "\n")
                    for metric_name, metric_value in sorted(model_results.metrics.items()):
                        if isinstance(metric_value, (int, float)):
                            f.write(f"  {metric_name}: {metric_value:.6f}\n")
                        else:
                            f.write(f"  {metric_name}: {metric_value}\n")
                    f.write("\n")
                
                if hasattr(model_results, 'parameters') and model_results.parameters:
                    f.write("Model Parameters:\n")
                    f.write("-" * 80 + "\n")
                    for param_name, param_value in sorted(model_results.parameters.items()):
                        f.write(f"  {param_name}: {param_value}\n")
            
            mlflow.log_artifact(summary_path, "evaluation")
            
            logger.info(f"Evaluation artifacts logged successfully for {model_name}")
    
    def _sanitize_param_name(self, name: str) -> str:
        """Sanitize parameter/metric names for MLflow compatibility"""
        import re
        sanitized = re.sub(r'[^\w\-\.\s/]', '_', name)
        return sanitized[:250]
    
    def _extract_feature_names(self, model: Any) -> Optional[List[str]]:
        """Extract feature names from a trained model"""
        try:
            if hasattr(model, 'feature_names_in_'):
                return list(model.feature_names_in_)
            
            if hasattr(model, 'get_booster') and hasattr(model.get_booster(), 'feature_names'):
                return model.get_booster().feature_names
            
            if hasattr(model, 'steps'):
                last_step = model.steps[-1][1]
                if hasattr(last_step, 'feature_names_in_'):
                    return list(last_step.feature_names_in_)
            
            return None
            
        except Exception as e:
            logger.warning(f"Could not extract feature names from model: {e}")
            return None

    def log_model_training(self, model_results: Any,
                          model_type: str,
                          dataset_info: Dict[str, Any] = None,
                          hyperparameters: Dict[str, Any] = None,
                          tags: Dict[str, str] = None,
                          train_data: pd.DataFrame = None,
                          test_data: pd.DataFrame = None) -> str:
        """
        Log model training experiment to MLFlow

        Args:
            model_results: ModelResults object from modeling module
            model_type: Type of model (token_elasticity, enrollment_propensity, etc.)
            dataset_info: Information about training dataset
            hyperparameters: Model hyperparameters
            tags: Additional tags for the run
            train_data: Training dataset (optional)
            test_data: Testing dataset (optional)

        Returns:
            Run ID of the logged experiment
        """
        # Create descriptive run name
        if tags and 'model_name' in tags:
            model_name = tags['model_name']
        else:
            model_name = 'model'
        
        dataset_size = dataset_info.get('rows', 0) if dataset_info else 0
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if model_type == 'token_elasticity':
            run_name = f"Token_Elasticity_{model_name.replace('_', ' ').title().replace(' ', '_')}_{dataset_size}courses_{timestamp}"
        elif model_type == 'enrollment_propensity':
            run_name = f"Enrollment_Propensity_{model_name.replace('_', ' ').title().replace(' ', '_')}_{dataset_size}learners_{timestamp}"
        else:
            run_name = f"{model_type.title()}_{model_name.replace('_', ' ').title().replace(' ', '_')}_{timestamp}"
        
        with mlflow.start_run(run_name=run_name):

            # Log datasets if provided
            if train_data is not None:
                self.log_dataset(train_data, f"{model_type}_train", context="training")
            if test_data is not None:
                self.log_dataset(test_data, f"{model_type}_test", context="testing")

            # Log run description
            run_description = self._generate_run_description(
                model_type, model_name, dataset_info, model_results
            )
            mlflow.set_tag("mlflow.note.content", run_description)

            # Log model parameters
            if hasattr(model_results, 'parameters') and model_results.parameters:
                mlflow.log_params(model_results.parameters)

            # Log additional hyperparameters
            if hyperparameters:
                mlflow.log_params(hyperparameters)

            # Log model metrics
            if hasattr(model_results, 'metrics') and model_results.metrics:
                mlflow.log_metrics(model_results.metrics)

            # Log dataset information
            if dataset_info:
                mlflow.log_params({
                    'dataset_rows': dataset_info.get('rows', 0),
                    'dataset_columns': dataset_info.get('columns', 0),
                    'dataset_features': dataset_info.get('features', 0),
                    'dataset_target': dataset_info.get('target', 'unknown')
                })
                
                if 'features' in dataset_info and isinstance(dataset_info['features'], list):
                    mlflow.set_tag("dataset_feature_count", len(dataset_info['features']))
                    feature_preview = ', '.join(dataset_info['features'][:5])
                    if len(dataset_info['features']) > 5:
                        feature_preview += f", ... (+{len(dataset_info['features']) - 5} more)"
                    mlflow.set_tag("dataset_features_preview", feature_preview)

            # Log feature importance
            if hasattr(model_results, 'feature_importance') and model_results.feature_importance:
                sorted_features = sorted(model_results.feature_importance.items(),
                                       key=lambda x: abs(x[1]), reverse=True)[:10]

                for i, (feature, importance) in enumerate(sorted_features):
                    safe_feature_name = self._sanitize_param_name(feature)
                    mlflow.log_metric(f"feature_importance_{i+1}_{safe_feature_name}", importance)

            # Log primary metrics
            if hasattr(model_results, 'metrics'):
                metrics = model_results.metrics

                if model_type == 'token_elasticity':
                    primary_metric = metrics.get('test_r2', metrics.get('r2', 0))
                    mlflow.log_metric('primary_metric_r2', primary_metric)

                    if 'price_elasticity' in metrics:
                        mlflow.log_metric('price_elasticity_coefficient', metrics['price_elasticity'])

                elif model_type == 'enrollment_propensity':
                    primary_metric = metrics.get('f1_score', metrics.get('accuracy', 0))
                    mlflow.log_metric('primary_metric_f1', primary_metric)

                if hasattr(model_results, 'training_time'):
                    mlflow.log_metric('training_time_seconds', model_results.training_time)

            # Log the model
            try:
                model_signature = None
                input_example = None

                if hasattr(model_results, 'model') and hasattr(model_results.model, 'predict'):
                    try:
                        feature_names = self._extract_feature_names(model_results.model)
                        sample_input = self._create_sample_input(model_type, feature_names)
                        if sample_input is not None:
                            model_signature = infer_signature(sample_input, model_results.predictions[:min(10, len(model_results.predictions))])
                            input_example = sample_input

                    except Exception as e:
                        logger.warning(f"Could not infer model signature: {e}")

                mlflow.sklearn.log_model(
                    model_results.model,
                    "model",
                    signature=model_signature,
                    input_example=input_example
                )

                logger.info(f"Model logged successfully: {model_type}")

            except Exception as e:
                logger.error(f"Failed to log model: {e}")
            
            # Log evaluation artifacts
            try:
                self._log_evaluation_artifacts(
                    model_results, 
                    model_type, 
                    model_name,
                    dataset_info
                )
            except Exception as e:
                logger.warning(f"Failed to log evaluation artifacts: {e}")

            # Log additional tags
            if tags:
                mlflow.set_tags(tags)

            # Log model metadata
            mlflow.set_tag("model_type", model_type)
            mlflow.set_tag("training_timestamp", datetime.now().isoformat())
            mlflow.set_tag("model_developer", "EdTech Token Economy Pipeline")
            mlflow.set_tag("platform", "EdTech")

            run_id = mlflow.active_run().info.run_id
            logger.info(f"Training run logged with ID: {run_id}")

            return run_id

    def _create_sample_input(self, model_type: str, feature_names: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """Create sample input data for model signature"""
        try:
            if feature_names is not None and len(feature_names) > 0:
                sample_data = {}
                for feature in feature_names:
                    if 'token_price' in feature.lower() or 'price' in feature.lower():
                        sample_data[feature] = [100.0, 150.0, 80.0]
                    elif 'log_' in feature.lower():
                        sample_data[feature] = [4.6, 5.0, 4.4]
                    elif 'enrollment' in feature.lower():
                        sample_data[feature] = [50, 100, 25]
                    elif 'category_' in feature.lower() or feature.startswith('level_'):
                        sample_data[feature] = [0, 1, 0]
                    elif 'rating' in feature.lower() or 'score' in feature.lower():
                        sample_data[feature] = [4.5, 4.8, 4.2]
                    elif 'duration' in feature.lower():
                        sample_data[feature] = [10, 20, 5]
                    else:
                        sample_data[feature] = [0.5, 1.0, 0.3]
                
                return pd.DataFrame(sample_data)
            
            return None

        except Exception as e:
            logger.warning(f"Could not create sample input: {e}")
            return None

    def log_dataset(self, dataset: pd.DataFrame, name: str, context: str = "training") -> None:
        """Log dataset to MLflow with versioning"""
        try:
            from mlflow.data.pandas_dataset import from_pandas
            import hashlib
            
            dataset_hash = hashlib.md5(
                pd.util.hash_pandas_object(dataset, index=True).values
            ).hexdigest()[:8]
            
            mlflow_dataset = from_pandas(
                dataset, 
                source=f"database://{name}",
                name=f"{name}_{context}",
                digest=dataset_hash
            )
            
            mlflow.log_input(mlflow_dataset, context=context)
            
            mlflow.log_params({
                f'{context}_dataset_rows': len(dataset),
                f'{context}_dataset_cols': len(dataset.columns),
                f'{context}_dataset_hash': dataset_hash
            })
            
            logger.info(f"Dataset '{name}' logged successfully (hash: {dataset_hash})")
            
        except Exception as e:
            logger.warning(f"Could not log dataset: {e}")

    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get comprehensive experiment summary"""
        try:
            experiment = mlflow.get_experiment(self.experiment_id)

            if self.client is None:
                logger.warning("MLflow client not initialized, creating a new one")
                self.client = MlflowClient()
            
            runs = self.client.search_runs(experiment_ids=[self.experiment_id])

            summary = {
                'experiment_name': self.experiment_name,
                'experiment_id': self.experiment_id,
                'total_runs': len(runs),
                'tracking_uri': self.tracking_uri,
                'creation_time': pd.to_datetime(experiment.creation_time, unit='ms').isoformat() if experiment.creation_time else None,
                'last_update_time': pd.to_datetime(experiment.last_update_time, unit='ms').isoformat() if experiment.last_update_time else None
            }

            # Analyze runs by model type
            model_types = {}
            for run in runs:
                try:
                    model_type = run.data.tags.get('model_type', 'unknown')
                    if model_type not in model_types:
                        model_types[model_type] = 0
                    model_types[model_type] += 1
                except Exception as e:
                    logger.warning(f"Error processing run: {e}")
                    continue

            summary['model_types'] = model_types

            return summary

        except Exception as e:
            logger.error(f"Failed to get experiment summary: {e}")
            return {}

    def export_experiment_results(self, output_path: str = "mlruns/experiment_summary.json"):
        """Export experiment results to file"""
        try:
            summary = self.get_experiment_summary()

            def convert_timestamps(obj):
                if isinstance(obj, pd.Timestamp):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: convert_timestamps(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_timestamps(item) for item in obj]
                else:
                    return obj
            
            summary = convert_timestamps(summary)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)

            logger.info(f"Experiment summary exported to {output_path}")

        except Exception as e:
            logger.error(f"Failed to export experiment results: {e}")


# Utility functions for MLFlow integration
def setup_mlflow_for_edtech_pipeline(experiment_name: str = "EdTech_Token_Economy") -> EdTechMLFlowTracker:
    """
    Set up MLFlow for the EdTech pipeline

    Args:
        experiment_name: Name of the MLFlow experiment

    Returns:
        Configured EdTechMLFlowTracker
    """
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    root_mlruns_path = os.path.join(project_root, "mlruns")
    tracker = EdTechMLFlowTracker(experiment_name=experiment_name, tracking_uri=root_mlruns_path)

    # Log pipeline setup
    with mlflow.start_run(run_name="EdTech_Pipeline_Setup"):
        mlflow.set_tag("pipeline_component", "mlflow_setup")
        mlflow.set_tag("setup_timestamp", datetime.now().isoformat())
        mlflow.log_param("experiment_name", experiment_name)
        mlflow.log_param("tracking_uri", tracker.tracking_uri)

    logger.info(f"MLFlow setup completed for EdTech experiment: {experiment_name}")
    return tracker


# Example usage
if __name__ == "__main__":
    print("Testing EdTech MLFlow Setup Module...")

    try:
        tracker = setup_mlflow_for_edtech_pipeline("EdTech_Test")

        print(f"✅ MLFlow tracker initialized: {tracker.experiment_name}")
        print(f"✅ Experiment ID: {tracker.experiment_id}")
        print(f"✅ Tracking URI: {tracker.tracking_uri}")

        summary = tracker.get_experiment_summary()
        print(f"✅ Experiment summary generated: {summary.get('total_runs', 0)} runs")

        tracker.export_experiment_results()
        print("✅ Experiment results exported")

        print("\n✅ EdTech MLFlow Setup Module test completed successfully!")

    except Exception as e:
        print(f"❌ MLFlow setup failed: {e}")
        logger.error(f"MLFlow test error: {e}")

