"""
Token Price Elasticity Modeling Module

This module implements token price elasticity models for the EdTech platform.
Predicts how course enrollment demand changes with token price.

Model Types:
- Linear Regression
- Generalized Additive Models (GAM)
- Polynomial Regression
- Random Forest
- Gradient Boosting

Author: EdTech Token Economy Team
Date: October 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
from dataclasses import dataclass
import pickle
import json

# ML Libraries
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Advanced ML
from pygam import LinearGAM, s, l
from xgboost import XGBRegressor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelResults:
    """Data class for storing model results"""
    model_name: str
    model: Any
    predictions: np.ndarray
    metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    training_time: float
    parameters: Dict[str, Any]


class TokenPriceElasticityModeler:
    """Main class for token price elasticity modeling"""
    
    def __init__(self, target_column: str = 'total_enrollments'):
        """
        Initialize token price elasticity modeler
        
        Args:
            target_column: Target column for modeling (enrollments)
        """
        self.target_column = target_column
        self.models = {}
        self.results = {}
        self.best_model = None
        self.scaler = StandardScaler()
    
    def prepare_elasticity_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data specifically for token price elasticity modeling
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame prepared for elasticity modeling
        """
        df_elasticity = df.copy()
        
        # Log transformations for elasticity modeling
        if 'token_price' in df_elasticity.columns:
            df_elasticity['log_token_price'] = np.log(df_elasticity['token_price'] + 1)
        
        if self.target_column in df_elasticity.columns:
            df_elasticity['log_enrollments'] = np.log(df_elasticity[self.target_column] + 1)
        
        # Price features
        if 'token_price' in df_elasticity.columns and 'original_token_price' in df_elasticity.columns:
            df_elasticity['price_discount_ratio'] = (
                (df_elasticity['original_token_price'] - df_elasticity['token_price']) / 
                df_elasticity['original_token_price']
            )
        
        # Encode categorical features
        categorical_cols = ['category', 'subcategory', 'difficulty_level', 'quality_tier']
        for col in categorical_cols:
            if col in df_elasticity.columns:
                dummies = pd.get_dummies(df_elasticity[col], prefix=col, drop_first=True)
                df_elasticity = pd.concat([df_elasticity, dummies], axis=1)
        
        # Handle NaN and inf values
        df_elasticity.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        numeric_cols = df_elasticity.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_elasticity[col].isnull().sum() > 0:
                median_val = df_elasticity[col].median()
                if pd.isna(median_val):
                    median_val = 0
                df_elasticity[col].fillna(median_val, inplace=True)
        
        df_elasticity.fillna(0, inplace=True)
        
        logger.info(f"Elasticity data prepared: {len(df_elasticity)} rows")
        return df_elasticity
    
    def train_linear_elasticity_model(self, df: pd.DataFrame,
                                     test_size: float = 0.2) -> ModelResults:
        """
        Train linear regression model for token price elasticity
        
        Args:
            df: DataFrame with features and target
            test_size: Test set size
            
        Returns:
            ModelResults object
        """
        start_time = datetime.now()
        
        # Select features
        feature_cols = [col for col in df.columns if col not in [
            self.target_column, 'log_enrollments', 'course_id', 'course_title',
            'teacher_id', 'learner_id', 'enrollment_id', 'category', 'subcategory',
            'difficulty_level', 'quality_tier', 'specialization'
        ]]
        
        X = df[feature_cols]
        y = df['log_enrollments']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Add feature scaling to prevent division warnings
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Use Ridge regression with regularization
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Feature importance
        feature_importance = dict(zip(feature_cols, model.coef_))
        
        # Price elasticity coefficient
        price_elasticity = feature_importance.get('log_token_price', 0)
        
        metrics = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'price_elasticity': price_elasticity
        }
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        return ModelResults(
            model_name='Linear Elasticity Model',
            model=model,
            predictions=y_pred_test,
            metrics=metrics,
            feature_importance=feature_importance,
            training_time=training_time,
            parameters={
                'test_size': test_size,
                'elasticity_interpretation': 'Elastic' if abs(price_elasticity) > 1 else 'Inelastic'
            }
        )
    
    def train_gam_elasticity_model(self, df: pd.DataFrame,
                                   test_size: float = 0.2,
                                   max_features: int = 15) -> ModelResults:
        """
        Train Generalized Additive Model for token price elasticity
        
        Args:
            df: DataFrame with features and target
            test_size: Test set size
            max_features: Maximum number of features to use
            
        Returns:
            ModelResults object
        """
        start_time = datetime.now()
        
        # Select features
        feature_cols = [col for col in df.columns if col not in [
            self.target_column, 'log_enrollments', 'course_id', 'course_title',
            'teacher_id', 'learner_id', 'enrollment_id', 'category', 'subcategory',
            'difficulty_level', 'quality_tier', 'specialization'
        ]]
        
        X_full = df[feature_cols]
        y = df['log_enrollments']
        
        # Feature selection based on correlation
        correlations = X_full.corrwith(y).abs().sort_values(ascending=False)
        
        # Ensure log_token_price is included
        top_features = []
        if 'log_token_price' in feature_cols:
            top_features.append('log_token_price')
            remaining = [f for f in correlations.head(max_features).index if f != 'log_token_price']
            top_features.extend(remaining[:max_features-1])
        else:
            top_features = correlations.head(max_features).index.tolist()
        
        logger.info(f"GAM using {len(top_features)} features")
        
        X = df[top_features].values
        y = df['log_enrollments'].values
        
        # Add feature scaling to prevent division warnings
        from sklearn.preprocessing import StandardScaler
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )
        
        # Build GAM term
        n_features = X_scaled.shape[1]
        price_idx = top_features.index('log_token_price') if 'log_token_price' in top_features else -1
        
        # Create GAM terms
        if price_idx == 0:
            gam_term = s(0, n_splines=10)
        else:
            gam_term = l(0)
        
        for i in range(1, n_features):
            if i == price_idx:
                gam_term = gam_term + s(i, n_splines=10)
            else:
                gam_term = gam_term + l(i)
        
        # Train GAM
        gam = LinearGAM(gam_term)
        gam.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = gam.predict(X_train)
        y_pred_test = gam.predict(X_test)
        
        # Metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Feature importance (approximate)
        feature_importance = {}
        for i, col in enumerate(top_features):
            try:
                feature_range = X_train[:, i].max() - X_train[:, i].min()
                feature_importance[col] = float(feature_range)
            except:
                feature_importance[col] = 0.0
        
        price_elasticity = feature_importance.get('log_token_price', 0.0)
        
        metrics = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'price_elasticity': price_elasticity,
            'features_used': len(top_features)
        }
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        return ModelResults(
            model_name='GAM Elasticity Model',
            model=gam,
            predictions=y_pred_test,
            metrics=metrics,
            feature_importance=feature_importance,
            training_time=training_time,
            parameters={
                'test_size': test_size,
                'max_features': len(top_features),
                'elasticity_interpretation': 'Elastic' if abs(price_elasticity) > 1 else 'Inelastic'
            }
        )
    
    def train_polynomial_elasticity_model(self, df: pd.DataFrame,
                                         degree: int = 2,
                                         test_size: float = 0.2,
                                         max_features: int = 10) -> ModelResults:
        """
        Train polynomial regression model for token price elasticity
        
        Args:
            df: DataFrame with features and target
            degree: Polynomial degree
            test_size: Test set size
            max_features: Maximum number of base features
            
        Returns:
            ModelResults object
        """
        start_time = datetime.now()
        
        # Select features
        feature_cols = [col for col in df.columns if col not in [
            self.target_column, 'log_enrollments', 'course_id', 'course_title',
            'teacher_id', 'learner_id', 'enrollment_id', 'category', 'subcategory',
            'difficulty_level', 'quality_tier', 'specialization'
        ]]
        
        X_full = df[feature_cols]
        y = df['log_enrollments']
        
        # Feature selection
        correlations = X_full.corrwith(y).abs().sort_values(ascending=False)
        
        selected_features = []
        if 'log_token_price' in feature_cols:
            selected_features.append('log_token_price')
            remaining = [f for f in correlations.head(max_features).index if f != 'log_token_price']
            selected_features.extend(remaining[:max_features-1])
        else:
            selected_features = correlations.head(max_features).index.tolist()
        
        logger.info(f"Polynomial deg {degree} using {len(selected_features)} base features")
        
        X = df[selected_features]
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        poly_feature_names = poly.get_feature_names_out(selected_features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_poly, y, test_size=test_size, random_state=42
        )
        
        # Add feature scaling to prevent division warnings
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Use Ridge regression with regularization
        alpha = 1.0 if degree <= 2 else 10.0  # Higher regularization for higher degrees
        model = Ridge(alpha=alpha, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Feature importance
        feature_importance = dict(zip(poly_feature_names, model.coef_))
        
        # Calculate price elasticity
        price_elasticity = 0
        for feature, coef in feature_importance.items():
            if 'log_token_price' in feature:
                price_elasticity += coef
        
        metrics = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'price_elasticity': price_elasticity,
            'polynomial_degree': degree,
            'base_features': len(selected_features),
            'polynomial_features': X_poly.shape[1]
        }
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        return ModelResults(
            model_name=f'Polynomial Degree {degree} Elasticity Model',
            model=Pipeline([('poly', poly), ('linear', model)]),
            predictions=y_pred_test,
            metrics=metrics,
            feature_importance=feature_importance,
            training_time=training_time,
            parameters={
                'degree': degree,
                'test_size': test_size,
                'base_features': len(selected_features),
                'elasticity_interpretation': 'Elastic' if abs(price_elasticity) > 1 else 'Inelastic'
            }
        )
    
    def train_ensemble_elasticity_model(self, df: pd.DataFrame,
                                       model_type: str = 'random_forest',
                                       test_size: float = 0.2) -> ModelResults:
        """
        Train ensemble model for token price elasticity
        
        Args:
            df: DataFrame with features and target
            model_type: Type of ensemble ('random_forest', 'gradient_boosting', 'xgboost')
            test_size: Test set size
            
        Returns:
            ModelResults object
        """
        start_time = datetime.now()
        
        # Select features
        feature_cols = [col for col in df.columns if col not in [
            self.target_column, 'log_enrollments', 'course_id', 'course_title',
            'teacher_id', 'learner_id', 'enrollment_id', 'category', 'subcategory',
            'difficulty_level', 'quality_tier', 'specialization'
        ]]
        
        X = df[feature_cols]
        y = df['log_enrollments']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Select model
        if model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif model_type == 'xgboost':
            model = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(feature_cols, model.feature_importances_))
        else:
            feature_importance = {col: 0 for col in feature_cols}
        
        price_elasticity = feature_importance.get('log_token_price', 0)
        
        metrics = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'price_elasticity': price_elasticity
        }
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        return ModelResults(
            model_name=f'{model_type.replace("_", " ").title()} Elasticity Model',
            model=model,
            predictions=y_pred_test,
            metrics=metrics,
            feature_importance=feature_importance,
            training_time=training_time,
            parameters={
                'model_type': model_type,
                'test_size': test_size,
                'elasticity_interpretation': 'Elastic' if abs(price_elasticity) > 1 else 'Inelastic'
            }
        )
    
    def compare_models(self, df: pd.DataFrame) -> Dict[str, ModelResults]:
        """
        Train and compare multiple elasticity models
        
        Args:
            df: DataFrame with features and target
            
        Returns:
            Dictionary of model results
        """
        logger.info("Starting model comparison...")
        
        # Prepare data
        df_prepared = self.prepare_elasticity_data(df)
        
        # Train models
        models_to_train = [
            ('linear', self.train_linear_elasticity_model),
            ('gam', lambda x: self.train_gam_elasticity_model(x, max_features=15)),
            ('polynomial_deg2', lambda x: self.train_polynomial_elasticity_model(x, degree=2, max_features=10)),
            ('polynomial_deg3', lambda x: self.train_polynomial_elasticity_model(x, degree=3, max_features=8)),
            ('random_forest', lambda x: self.train_ensemble_elasticity_model(x, 'random_forest')),
            ('gradient_boosting', lambda x: self.train_ensemble_elasticity_model(x, 'gradient_boosting'))
        ]
        
        results = {}
        for model_name, train_func in models_to_train:
            try:
                logger.info(f"Training {model_name} model...")
                result = train_func(df_prepared)
                results[model_name] = result
                self.results[model_name] = result
                
                logger.info(f"{model_name}: R² = {result.metrics['test_r2']:.3f}, "
                          f"Elasticity = {result.metrics['price_elasticity']:.3f}")
            
            except Exception as e:
                logger.error(f"Failed to train {model_name} model: {e}")
        
        # Select best model
        if results:
            best_model_name = max(results.keys(), key=lambda k: results[k].metrics['test_r2'])
            self.best_model = results[best_model_name]
            logger.info(f"Best model: {best_model_name} (R² = {self.best_model.metrics['test_r2']:.3f})")
        
        return results
    
    def calculate_optimal_token_price(self, course_features: Dict[str, Any],
                                      current_price: float,
                                      current_enrollments: int,
                                      elasticity_coefficient: float) -> Dict[str, float]:
        """
        Calculate optimal token price for maximum revenue using proper economic theory
        
        Args:
            course_features: Course characteristics
            current_price: Current token price
            current_enrollments: Current enrollment count
            elasticity_coefficient: Price elasticity coefficient
            
        Returns:
            Dictionary with optimal pricing recommendations
        """
        import numpy as np
        
        # Revenue maximization using economic theory
        # Revenue = Price × Quantity
        # For optimal revenue: dR/dP = 0
        # This occurs when elasticity = -1 (unitary elasticity)
        
        # Handle edge cases
        if elasticity_coefficient == 0:
            # Perfectly inelastic demand - can raise price significantly
            optimal_price = current_price * 1.2
        elif abs(elasticity_coefficient) < 0.1:
            # Very inelastic demand - small price increase
            optimal_price = current_price * 1.05
        elif abs(elasticity_coefficient) > 10:
            # Very elastic demand - small price decrease
            optimal_price = current_price * 0.95
        else:
            # Use proper economic optimization
            # For revenue maximization: optimal elasticity = -1
            # Price adjustment factor = (1 + elasticity) / elasticity
            # But we need to be careful with the sign
            
            elasticity_magnitude = abs(elasticity_coefficient)
            
            if elasticity_magnitude > 1:
                # Elastic demand: reduce price to increase revenue
                # Optimal price reduction: (elasticity - 1) / elasticity
                price_reduction_factor = (elasticity_magnitude - 1) / elasticity_magnitude
                optimal_price = current_price * (1 - price_reduction_factor * 0.1)  # Conservative 10% of optimal
            else:
                # Inelastic demand: increase price to increase revenue
                # Optimal price increase: (1 - elasticity) / elasticity
                price_increase_factor = (1 - elasticity_magnitude) / elasticity_magnitude
                optimal_price = current_price * (1 + price_increase_factor * 0.1)  # Conservative 10% of optimal
        
        # Ensure reasonable bounds (don't go below 50% or above 200% of current price)
        optimal_price = max(current_price * 0.5, min(optimal_price, current_price * 2.0))
        
        # Predict enrollments at optimal price
        price_change_pct = (optimal_price - current_price) / current_price
        enrollment_change_pct = elasticity_coefficient * price_change_pct
        predicted_enrollments = max(0, int(current_enrollments * (1 + enrollment_change_pct)))
        
        # Calculate revenues
        current_revenue = current_price * current_enrollments
        optimal_revenue = optimal_price * predicted_enrollments
        
        return {
            'current_price': current_price,
            'optimal_price': round(optimal_price, 2),
            'price_change_pct': round(price_change_pct * 100, 1),
            'current_enrollments': current_enrollments,
            'predicted_enrollments': predicted_enrollments,
            'enrollment_change_pct': round(enrollment_change_pct * 100, 1),
            'current_revenue': round(current_revenue, 2),
            'optimal_revenue': round(optimal_revenue, 2),
            'revenue_increase': round(optimal_revenue - current_revenue, 2),
            'revenue_increase_pct': round((optimal_revenue - current_revenue) / current_revenue * 100, 1) if current_revenue > 0 else 0,
            'elasticity_coefficient': elasticity_coefficient,
            'demand_type': 'Elastic' if abs(elasticity_coefficient) > 1 else 'Inelastic'
        }


if __name__ == "__main__":
    print("Testing Token Price Elasticity Modeling Module...")
    
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'course_id': [f'C{i:05d}' for i in range(1000)],
        'token_price': np.random.uniform(50, 200, 1000),
        'total_enrollments': np.random.randint(10, 500, 1000),
        'category': np.random.choice(['Programming', 'Design', 'Business'], 1000),
        'difficulty_level': np.random.choice(['beginner', 'intermediate', 'advanced'], 1000),
        'teacher_quality_score': np.random.uniform(60, 100, 1000),
        'quality_tier': np.random.choice(['Bronze', 'Silver', 'Gold'], 1000),
        'duration_hours': np.random.uniform(10, 50, 1000),
        'avg_rating': np.random.uniform(3.5, 5.0, 1000),
        'completion_rate': np.random.uniform(0.5, 0.9, 1000)
    })
    
    # Add original price
    sample_data['original_token_price'] = sample_data['token_price'] * 1.1
    
    # Test elasticity modeling
    print("\nTraining Token Price Elasticity Models...")
    modeler = TokenPriceElasticityModeler()
    results = modeler.compare_models(sample_data)
    
    if results:
        print(f"\n✓ Trained {len(results)} elasticity models")
        
        # Show results
        for model_name, result in results.items():
            print(f"\n{model_name}:")
            print(f"  R²: {result.metrics['test_r2']:.3f}")
            print(f"  RMSE: {result.metrics['test_rmse']:.3f}")
            print(f"  Price Elasticity: {result.metrics['price_elasticity']:.3f}")
        
        # Test optimal pricing
        optimal = modeler.calculate_optimal_token_price(
            course_features={'category': 'Programming'},
            current_price=100,
            current_enrollments=200,
            elasticity_coefficient=-1.2
        )
        print(f"\n✓ Optimal Pricing Recommendation:")
        print(f"  Current: {optimal['current_price']} tokens, {optimal['current_enrollments']} enrollments")
        print(f"  Optimal: {optimal['optimal_price']} tokens, {optimal['predicted_enrollments']} enrollments")
        print(f"  Revenue Increase: {optimal['revenue_increase_pct']}%")
    
    print("\nToken Price Elasticity Modeling Module test completed!")


