#!/usr/bin/env python3
"""
Model Performance Analysis Script for EdTech Token Economy ML Models

This script analyzes the performance of trained ML models including:
- Model accuracy and metrics comparison
- Feature importance analysis
- Prediction quality assessment
- Model reliability and stability
- Cross-validation results
- Model selection recommendations

Author: EdTech Token Economy Team
Version: 1.0.0
"""

import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class ModelPerformanceAnalyzer:
    """Analyzes performance of trained ML models"""
    
    def __init__(self, models_dir: str = "models"):
        """Initialize the analyzer with models directory"""
        self.models_dir = Path(models_dir)
        self.models = {}
        self.results = {}
        self.performance_issues = []
        
    def load_all_models(self) -> Dict[str, Any]:
        """Load all available trained models"""
        print("\n" + "="*60)
        print("LOADING TRAINED MODELS")
        print("="*60)
        
        model_files = list(self.models_dir.glob("*.pkl"))
        
        if not model_files:
            print("[ERROR] No model files found in models directory")
            return {}
        
        loaded_models = {}
        
        for model_file in model_files:
            try:
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                    model_name = model_file.stem
                    loaded_models[model_name] = model_data
                    print(f"[OK] Loaded {model_name}")
                    
                    # Print basic info
                    if isinstance(model_data, dict) and 'metrics' in model_data:
                        metrics = model_data['metrics']
                        print(f"    R²: {metrics.get('test_r2', 'N/A'):.3f}")
                        print(f"    RMSE: {metrics.get('test_rmse', 'N/A'):.3f}")
                    else:
                        print(f"    Model type: {type(model_data)}")
                        
            except Exception as e:
                print(f"[ERROR] Failed to load {model_file.name}: {e}")
                self.performance_issues.append(f"Failed to load {model_file.name}: {e}")
        
        self.models = loaded_models
        print(f"\n[OK] Loaded {len(loaded_models)} models successfully")
        return loaded_models
    
    def analyze_model_metrics(self) -> Dict[str, Any]:
        """Analyze and compare model performance metrics"""
        print("\n" + "="*60)
        print("MODEL PERFORMANCE METRICS ANALYSIS")
        print("="*60)
        
        metrics_results = {}
        
        if not self.models:
            print("[ERROR] No models loaded for analysis")
            return {}
        
        # Collect metrics from all models
        model_metrics = {}
        
        for model_name, model_data in self.models.items():
            if isinstance(model_data, dict) and 'metrics' in model_data:
                metrics = model_data['metrics']
                model_metrics[model_name] = {
                    'train_r2': metrics.get('train_r2', np.nan),
                    'test_r2': metrics.get('test_r2', np.nan),
                    'train_rmse': metrics.get('train_rmse', np.nan),
                    'test_rmse': metrics.get('test_rmse', np.nan),
                    'price_elasticity': metrics.get('price_elasticity', np.nan)
                }
            else:
                print(f"[WARN] {model_name}: No metrics found")
                self.performance_issues.append(f"{model_name}: No metrics found")
        
        if not model_metrics:
            print("[ERROR] No valid metrics found in any model")
            return {}
        
        # Create metrics DataFrame
        metrics_df = pd.DataFrame(model_metrics).T
        
        print(f"\nModel Performance Summary:")
        print(f"{'Model':<25} {'Train R²':<10} {'Test R²':<10} {'Train RMSE':<12} {'Test RMSE':<12} {'Elasticity':<12}")
        print("-" * 90)
        
        for model_name, metrics in model_metrics.items():
            print(f"{model_name:<25} {metrics['train_r2']:<10.3f} {metrics['test_r2']:<10.3f} "
                  f"{metrics['train_rmse']:<12.3f} {metrics['test_rmse']:<12.3f} {metrics['price_elasticity']:<12.3f}")
        
        # Identify best model
        valid_models = metrics_df.dropna(subset=['test_r2'])
        if not valid_models.empty:
            best_model = valid_models.loc[valid_models['test_r2'].idxmax()]
            print(f"\n[OK] Best Model: {valid_models['test_r2'].idxmax()}")
            print(f"  Test R²: {best_model['test_r2']:.3f}")
            print(f"  Test RMSE: {best_model['test_rmse']:.3f}")
        else:
            print("\n[ERROR] No models with valid test R² found")
            self.performance_issues.append("No models with valid test R²")
        
        # Performance analysis
        print(f"\nPerformance Analysis:")
        
        # Check for overfitting
        overfitted_models = []
        for model_name, metrics in model_metrics.items():
            if not np.isnan(metrics['train_r2']) and not np.isnan(metrics['test_r2']):
                r2_diff = metrics['train_r2'] - metrics['test_r2']
                if r2_diff > 0.2:  # Significant overfitting
                    overfitted_models.append((model_name, r2_diff))
                    print(f"  [WARN] {model_name}: Overfitting detected (R² diff: {r2_diff:.3f})")
        
        # Check for poor performance
        poor_models = []
        for model_name, metrics in model_metrics.items():
            if not np.isnan(metrics['test_r2']):
                if metrics['test_r2'] < 0:  # Worse than baseline
                    poor_models.append((model_name, metrics['test_r2']))
                    print(f"  [ERROR] {model_name}: Poor performance (R²: {metrics['test_r2']:.3f})")
                elif metrics['test_r2'] < 0.1:  # Very low performance
                    poor_models.append((model_name, metrics['test_r2']))
                    print(f"  [WARN] {model_name}: Low performance (R²: {metrics['test_r2']:.3f})")
        
        metrics_results = {
            'model_metrics': model_metrics,
            'best_model': valid_models['test_r2'].idxmax() if not valid_models.empty else None,
            'overfitted_models': overfitted_models,
            'poor_models': poor_models,
            'total_models': len(model_metrics)
        }
        
        self.results['metrics'] = metrics_results
        return metrics_results
    
    def analyze_feature_importance(self) -> Dict[str, Any]:
        """Analyze feature importance across models"""
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        feature_results = {}
        
        if not self.models:
            print("[ERROR] No models loaded for analysis")
            return {}
        
        # Collect feature importance from all models
        feature_importance = {}
        
        for model_name, model_data in self.models.items():
            if isinstance(model_data, dict) and 'feature_importance' in model_data:
                importance = model_data['feature_importance']
                if importance:
                    feature_importance[model_name] = importance
                    print(f"\n{model_name} - Top Features:")
                    sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
                    for feature, importance_val in sorted_features[:10]:
                        print(f"  {feature}: {importance_val:.3f}")
            else:
                print(f"[WARN] {model_name}: No feature importance found")
        
        if not feature_importance:
            print("[ERROR] No feature importance data found in any model")
            return {}
        
        # Analyze common important features
        print(f"\nFeature Importance Analysis:")
        
        # Get all unique features
        all_features = set()
        for model_features in feature_importance.values():
            all_features.update(model_features.keys())
        
        # Calculate average importance for each feature
        avg_importance = {}
        for feature in all_features:
            importances = []
            for model_features in feature_importance.values():
                if feature in model_features:
                    importances.append(abs(model_features[feature]))
            if importances:
                avg_importance[feature] = np.mean(importances)
        
        # Sort by average importance
        sorted_avg = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nAverage Feature Importance (across all models):")
        for feature, importance in sorted_avg[:15]:
            print(f"  {feature}: {importance:.3f}")
        
        # Identify most consistent features
        consistent_features = []
        for feature in all_features:
            present_in_models = sum(1 for model_features in feature_importance.values() if feature in model_features)
            if present_in_models >= len(feature_importance) * 0.7:  # Present in 70% of models
                consistent_features.append(feature)
        
        print(f"\nMost Consistent Features (present in >=70% of models):")
        for feature in consistent_features[:10]:
            print(f"  {feature}")
        
        feature_results = {
            'feature_importance': feature_importance,
            'average_importance': avg_importance,
            'consistent_features': consistent_features,
            'total_features': len(all_features)
        }
        
        self.results['features'] = feature_results
        return feature_results
    
    def analyze_model_stability(self) -> Dict[str, Any]:
        """Analyze model stability and reliability"""
        print("\n" + "="*60)
        print("MODEL STABILITY ANALYSIS")
        print("="*60)
        
        stability_results = {}
        
        if not self.models:
            print("[ERROR] No models loaded for analysis")
            return {}
        
        # Check for extreme values in metrics
        print(f"\nModel Stability Checks:")
        
        stability_issues = []
        
        for model_name, model_data in self.models.items():
            if isinstance(model_data, dict) and 'metrics' in model_data:
                metrics = model_data['metrics']
                
                # Check for extreme R² values
                test_r2 = metrics.get('test_r2', np.nan)
                if not np.isnan(test_r2):
                    if test_r2 < -1000:  # Extremely negative
                        stability_issues.append(f"{model_name}: Extremely negative R² ({test_r2:.2e})")
                        print(f"  [ERROR] {model_name}: Extremely negative R² ({test_r2:.2e})")
                    elif test_r2 > 1.0:  # Impossible R²
                        stability_issues.append(f"{model_name}: Impossible R² > 1.0 ({test_r2:.3f})")
                        print(f"  [ERROR] {model_name}: Impossible R² > 1.0 ({test_r2:.3f})")
                
                # Check for extreme RMSE values
                test_rmse = metrics.get('test_rmse', np.nan)
                if not np.isnan(test_rmse):
                    if test_rmse > 1000:  # Extremely high RMSE
                        stability_issues.append(f"{model_name}: Extremely high RMSE ({test_rmse:.2e})")
                        print(f"  [ERROR] {model_name}: Extremely high RMSE ({test_rmse:.2e})")
                
                # Check for extreme elasticity values
                elasticity = metrics.get('price_elasticity', np.nan)
                if not np.isnan(elasticity):
                    if abs(elasticity) > 10:  # Unrealistic elasticity
                        stability_issues.append(f"{model_name}: Unrealistic elasticity ({elasticity:.3f})")
                        print(f"  [ERROR] {model_name}: Unrealistic elasticity ({elasticity:.3f})")
        
        # Check for model consistency
        print(f"\nModel Consistency Analysis:")
        
        # Check if all models have similar elasticity estimates
        elasticity_values = []
        for model_name, model_data in self.models.items():
            if isinstance(model_data, dict) and 'metrics' in model_data:
                elasticity = model_data['metrics'].get('price_elasticity', np.nan)
                if not np.isnan(elasticity):
                    elasticity_values.append(elasticity)
        
        if len(elasticity_values) > 1:
            elasticity_std = np.std(elasticity_values)
            elasticity_mean = np.mean(elasticity_values)
            print(f"  Elasticity consistency:")
            print(f"    Mean: {elasticity_mean:.3f}")
            print(f"    Std Dev: {elasticity_std:.3f}")
            print(f"    Range: {min(elasticity_values):.3f} to {max(elasticity_values):.3f}")
            
            if elasticity_std > 1.0:  # High variability
                stability_issues.append(f"High elasticity variability across models (std: {elasticity_std:.3f})")
                print(f"  [WARN] High elasticity variability across models")
        
        stability_results = {
            'stability_issues': stability_issues,
            'elasticity_stats': {
                'mean': elasticity_mean if len(elasticity_values) > 1 else np.nan,
                'std': elasticity_std if len(elasticity_values) > 1 else np.nan,
                'values': elasticity_values
            },
            'total_issues': len(stability_issues)
        }
        
        self.results['stability'] = stability_results
        return stability_results
    
    def generate_model_recommendations(self) -> Dict[str, Any]:
        """Generate recommendations for model selection and improvement"""
        print("\n" + "="*60)
        print("MODEL RECOMMENDATIONS")
        print("="*60)
        
        recommendations = []
        
        if not self.results:
            print("[ERROR] No analysis results available for recommendations")
            return {}
        
        # Performance-based recommendations
        if 'metrics' in self.results:
            metrics_results = self.results['metrics']
            
            # Check for poor performing models
            if metrics_results['poor_models']:
                recommendations.append("1. Consider retraining models with better data preprocessing")
                recommendations.append("2. Check for data quality issues affecting model performance")
            
            # Check for overfitting
            if metrics_results['overfitted_models']:
                recommendations.append("3. Implement regularization to reduce overfitting")
                recommendations.append("4. Increase training data or use cross-validation")
            
            # Check for best model
            if metrics_results['best_model']:
                best_model = metrics_results['best_model']
                best_r2 = metrics_results['model_metrics'][best_model]['test_r2']
                if best_r2 < 0.3:
                    recommendations.append("5. Overall model performance is poor - consider feature engineering")
                elif best_r2 < 0.6:
                    recommendations.append("6. Model performance is moderate - consider ensemble methods")
        
        # Stability-based recommendations
        if 'stability' in self.results:
            stability_results = self.results['stability']
            
            if stability_results['total_issues'] > 0:
                recommendations.append("7. Address model stability issues before deployment")
                recommendations.append("8. Implement robust error handling for extreme predictions")
        
        # Feature-based recommendations
        if 'features' in self.results:
            feature_results = self.results['features']
            
            if len(feature_results['consistent_features']) < 5:
                recommendations.append("9. Few consistent features identified - improve feature selection")
            
            if feature_results['total_features'] > 50:
                recommendations.append("10. High feature count - consider dimensionality reduction")
        
        # General recommendations
        recommendations.extend([
            "11. Implement model monitoring in production",
            "12. Set up automated retraining pipelines",
            "13. Validate model predictions against business metrics",
            "14. Consider A/B testing for model deployment"
        ])
        
        print(f"\nModel Improvement Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {rec}")
        
        recommendations_results = {
            'recommendations': recommendations,
            'total_recommendations': len(recommendations)
        }
        
        self.results['recommendations'] = recommendations_results
        return recommendations_results
    
    def generate_performance_score(self) -> Dict[str, Any]:
        """Generate overall model performance score"""
        print("\n" + "="*60)
        print("MODEL PERFORMANCE SCORE")
        print("="*60)
        
        scores = {}
        
        # Performance score based on metrics
        performance_score = 0
        if 'metrics' in self.results and self.results['metrics']['model_metrics']:
            model_metrics = self.results['metrics']['model_metrics']
            valid_r2_scores = [m['test_r2'] for m in model_metrics.values() if not np.isnan(m['test_r2'])]
            
            if valid_r2_scores:
                avg_r2 = np.mean(valid_r2_scores)
                max_r2 = max(valid_r2_scores)
                
                # Score based on best R²
                if max_r2 >= 0.8:
                    performance_score = 90
                elif max_r2 >= 0.6:
                    performance_score = 75
                elif max_r2 >= 0.3:
                    performance_score = 60
                elif max_r2 >= 0.0:
                    performance_score = 40
                else:
                    performance_score = 10
        
        # Stability score
        stability_score = 100
        if 'stability' in self.results:
            stability_issues = self.results['stability']['total_issues']
            stability_score = max(0, 100 - (stability_issues * 15))
        
        # Feature quality score
        feature_score = 100
        if 'features' in self.results:
            consistent_features = len(self.results['features']['consistent_features'])
            if consistent_features < 3:
                feature_score = 50
            elif consistent_features < 5:
                feature_score = 75
        
        # Overall score
        overall_score = (performance_score + stability_score + feature_score) / 3
        
        scores = {
            'performance': round(performance_score, 1),
            'stability': round(stability_score, 1),
            'features': round(feature_score, 1),
            'overall': round(overall_score, 1)
        }
        
        print(f"\nModel Performance Scores:")
        print(f"  Performance: {scores['performance']}/100")
        print(f"  Stability: {scores['stability']}/100")
        print(f"  Features: {scores['features']}/100")
        print(f"  Overall: {scores['overall']}/100")
        
        # Performance grade
        if overall_score >= 85:
            grade = "A (Excellent)"
        elif overall_score >= 75:
            grade = "B (Good)"
        elif overall_score >= 65:
            grade = "C (Fair)"
        elif overall_score >= 55:
            grade = "D (Poor)"
        else:
            grade = "F (Very Poor)"
        
        print(f"  Grade: {grade}")
        
        self.results['performance_scores'] = scores
        return scores
    
    def generate_report(self) -> str:
        """Generate comprehensive model performance report"""
        report = []
        report.append("EDTECH TOKEN ECONOMY - MODEL PERFORMANCE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Models Directory: {self.models_dir}")
        report.append("")
        
        # Summary
        if 'performance_scores' in self.results:
            scores = self.results['performance_scores']
            report.append("EXECUTIVE SUMMARY")
            report.append("-" * 40)
            report.append(f"Overall Model Performance Score: {scores['overall']}/100")
            report.append(f"Performance: {scores['performance']}/100")
            report.append(f"Stability: {scores['stability']}/100")
            report.append(f"Features: {scores['features']}/100")
            report.append("")
        
        # Best model
        if 'metrics' in self.results and self.results['metrics']['best_model']:
            best_model = self.results['metrics']['best_model']
            report.append("RECOMMENDED MODEL")
            report.append("-" * 40)
            report.append(f"Best Model: {best_model}")
            if best_model in self.results['metrics']['model_metrics']:
                metrics = self.results['metrics']['model_metrics'][best_model]
                report.append(f"Test R²: {metrics['test_r2']:.3f}")
                report.append(f"Test RMSE: {metrics['test_rmse']:.3f}")
                report.append(f"Price Elasticity: {metrics['price_elasticity']:.3f}")
            report.append("")
        
        # Issues
        all_issues = []
        if 'metrics' in self.results:
            all_issues.extend([f"Poor performance: {model} (R²: {r2:.3f})" for model, r2 in self.results['metrics']['poor_models']])
            all_issues.extend([f"Overfitting: {model} (R² diff: {diff:.3f})" for model, diff in self.results['metrics']['overfitted_models']])
        
        if 'stability' in self.results:
            all_issues.extend(self.results['stability']['stability_issues'])
        
        all_issues.extend(self.performance_issues)
        
        if all_issues:
            report.append("ISSUES IDENTIFIED")
            report.append("-" * 40)
            for i, issue in enumerate(all_issues, 1):
                report.append(f"{i}. {issue}")
            report.append("")
        
        # Recommendations
        if 'recommendations' in self.results:
            report.append("RECOMMENDATIONS")
            report.append("-" * 40)
            for i, rec in enumerate(self.results['recommendations']['recommendations'], 1):
                report.append(f"{i}. {rec}")
            report.append("")
        
        return "\n".join(report)
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """Run complete model performance analysis"""
        print("EDTECH TOKEN ECONOMY - MODEL PERFORMANCE ANALYSIS")
        print("=" * 80)
        
        try:
            # Run all analyses
            self.load_all_models()
            self.analyze_model_metrics()
            self.analyze_feature_importance()
            self.analyze_model_stability()
            self.generate_model_recommendations()
            self.generate_performance_score()
            
            # Generate report
            report = self.generate_report()
            print("\n" + report)
            
            return self.results
            
        except Exception as e:
            print(f"[ERROR] Analysis failed: {e}")
            raise


def main():
    """Main function for command-line execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='EdTech Token Economy Model Performance Analysis')
    parser.add_argument('--models-dir', default='models', 
                       help='Directory containing model files')
    parser.add_argument('--output', help='Output file for detailed report')
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = ModelPerformanceAnalyzer(args.models_dir)
    results = analyzer.run_full_analysis()
    
    # Save detailed report if requested
    if args.output:
        report = analyzer.generate_report()
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"\n[OK] Detailed report saved to: {args.output}")
    
    return results


if __name__ == "__main__":
    main()
