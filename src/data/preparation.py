"""
Data Preparation Module - EdTech Token Economy

This module handles data cleaning, feature engineering, and preprocessing
for the EdTech token economy platform.

Features:
- Learner feature engineering
- Course feature engineering
- Teacher quality metrics
- Token transaction features
- Enrollment pattern features

Author: EdTech Token Economy Pipeline
Date: October 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class EdTechDataPreprocessor:
    """Main class for EdTech data preparation and preprocessing"""

    def __init__(self, target_column: str = 'total_enrollments'):
        """
        Initialize EdTech data preprocessor

        Args:
            target_column: Name of the target column for modeling
        """
        self.target_column = target_column
        self.numeric_columns = []
        self.categorical_columns = []
        self.datetime_columns = []
        self.processed_data = None
        self.feature_columns = []
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    def load_edtech_data(self, database_path: str = "edtech_token_economy.db") -> pd.DataFrame:
        """
        Load and merge EdTech data from database

        Args:
            database_path: Path to SQLite database

        Returns:
            Merged DataFrame with all relevant EdTech data
        """
        import sqlite3

        try:
            conn = sqlite3.connect(database_path)
            
            # Load course-level aggregated data with all features
            query = """
                SELECT 
                    c.course_id,
                    c.course_name,
                    c.category,
                    c.difficulty_level,
                    c.quality_tier,
                    c.token_price,
                    c.course_duration_hours,
                    c.language,
                    c.teacher_id,
                    t.teacher_name,
                    t.teacher_quality_score,
                    t.teacher_rating,
                    t.years_of_experience,
                    t.courses_taught,
                    COUNT(DISTINCT e.enrollment_id) as total_enrollments,
                    AVG(e.completion_rate) as avg_completion_rate,
                    AVG(e.rating_given) as avg_course_rating,
                    SUM(CASE WHEN e.completion_status = 'completed' THEN 1 ELSE 0 END) as completed_enrollments,
                    COUNT(DISTINCT e.learner_id) as unique_learners
                FROM courses c
                LEFT JOIN teachers t ON c.teacher_id = t.teacher_id
                LEFT JOIN enrollments e ON c.course_id = e.course_id
                GROUP BY c.course_id, c.course_name, c.category, c.difficulty_level, 
                         c.quality_tier, c.token_price, c.course_duration_hours, c.language,
                         c.teacher_id, t.teacher_name, t.teacher_quality_score, t.teacher_rating,
                         t.years_of_experience, t.courses_taught
            """

            df = pd.read_sql_query(query, conn)
            conn.close()

            logger.info(f"Loaded EdTech data: {len(df)} courses with {len(df.columns)} features")
            return df

        except Exception as e:
            logger.error(f"Failed to load data from database: {e}")
            return pd.DataFrame()

    def identify_column_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Identify and categorize column types for EdTech data

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with column type categories
        """
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Identify datetime columns
        self.datetime_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col])
                    self.datetime_columns.append(col)
                except:
                    pass
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                self.datetime_columns.append(col)

        return {
            'numeric': self.numeric_columns,
            'categorical': self.categorical_columns,
            'datetime': self.datetime_columns
        }

    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'auto') -> pd.DataFrame:
        """
        Handle missing values in EdTech dataset

        Args:
            df: DataFrame with missing values
            strategy: Strategy for handling missing values

        Returns:
            DataFrame with handled missing values
        """
        df_cleaned = df.copy()

        # Calculate missing percentages
        missing_info = df_cleaned.isnull().sum()
        missing_pct = (missing_info / len(df_cleaned)) * 100

        logger.info(f"Missing values before cleaning: {missing_info.sum()} total")

        if strategy == 'auto':
            for col in df_cleaned.columns:
                if missing_pct[col] > 50:
                    df_cleaned.drop(col, axis=1, inplace=True)
                    logger.info(f"Dropped column {col} ({missing_pct[col]:.1f}% missing)")
                elif missing_pct[col] > 0:
                    if col in self.numeric_columns:
                        # For EdTech metrics, use 0 for counts, median for scores
                        if 'enrollments' in col or 'count' in col or 'completed' in col:
                            df_cleaned[col].fillna(0, inplace=True)
                        else:
                            df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
                    else:
                        df_cleaned[col].fillna(df_cleaned[col].mode().iloc[0] if len(df_cleaned[col].mode()) > 0 else 'Unknown', inplace=True)

        logger.info(f"Missing values after cleaning: {df_cleaned.isnull().sum().sum()} total")
        return df_cleaned

    def engineer_edtech_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer EdTech-specific features

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with engineered features
        """
        df_engineered = df.copy()

        # === PRICING FEATURES ===
        if 'token_price' in df_engineered.columns and 'category' in df_engineered.columns:
            # Price relative to category average
            df_engineered['category_avg_price'] = df_engineered.groupby('category')['token_price'].transform('mean')
            df_engineered['price_vs_category_avg'] = df_engineered['token_price'] - df_engineered['category_avg_price']
            df_engineered['price_ratio_to_category'] = df_engineered['token_price'] / df_engineered['category_avg_price']
            
            # Price percentile within category
            df_engineered['price_percentile_in_category'] = df_engineered.groupby('category')['token_price'].rank(pct=True)

        # === ENROLLMENT FEATURES ===
        if 'total_enrollments' in df_engineered.columns:
            # Log-transform for better model performance
            df_engineered['log_enrollments'] = np.log(df_engineered['total_enrollments'] + 1)
            
            # Enrollment rate (enrollments per dollar)
            if 'token_price' in df_engineered.columns:
                df_engineered['enrollment_per_token'] = df_engineered['total_enrollments'] / (df_engineered['token_price'] + 1)
            
            # Enrollment density (enrollments per hour of course content)
            if 'course_duration_hours' in df_engineered.columns:
                df_engineered['enrollment_density'] = df_engineered['total_enrollments'] / (df_engineered['course_duration_hours'] + 1)

        # === QUALITY FEATURES ===
        if 'teacher_quality_score' in df_engineered.columns and 'teacher_rating' in df_engineered.columns:
            # Combined quality metric
            df_engineered['combined_quality_score'] = (
                df_engineered['teacher_quality_score'] * 0.6 + 
                df_engineered['teacher_rating'] * 0.4
            )
        
        if 'avg_completion_rate' in df_engineered.columns and 'avg_course_rating' in df_engineered.columns:
            # Engagement quality score
            df_engineered['engagement_quality'] = (
                df_engineered['avg_completion_rate'] * 0.7 + 
                (df_engineered['avg_course_rating'] / 5.0) * 0.3
            )

        # === COMPETITION FEATURES ===
        if 'category' in df_engineered.columns:
            # Number of courses in category (competition level)
            df_engineered['category_course_count'] = df_engineered.groupby('category')['course_id'].transform('count')
            df_engineered['competitive_intensity'] = df_engineered['category_course_count'] / df_engineered['category_course_count'].max()

        # === TEACHER FEATURES ===
        if 'years_of_experience' in df_engineered.columns:
            # Experience tier
            df_engineered['experience_tier'] = pd.cut(
                df_engineered['years_of_experience'],
                bins=[0, 2, 5, 10, 100],
                labels=['Novice', 'Intermediate', 'Senior', 'Expert']
            )
        
        if 'courses_taught' in df_engineered.columns:
            # Teacher productivity
            df_engineered['teacher_productivity'] = pd.cut(
                df_engineered['courses_taught'],
                bins=[0, 5, 10, 20, 1000],
                labels=['Low', 'Medium', 'High', 'Very High']
            )

        # === DIFFICULTY FEATURES ===
        if 'difficulty_level' in df_engineered.columns:
            # Encode difficulty as ordinal
            difficulty_map = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}
            df_engineered['difficulty_numeric'] = df_engineered['difficulty_level'].map(difficulty_map)

        # === INTERACTION FEATURES ===
        if 'token_price' in df_engineered.columns and 'teacher_quality_score' in df_engineered.columns:
            # Price-Quality interaction
            df_engineered['price_quality_interaction'] = df_engineered['token_price'] * df_engineered['teacher_quality_score']
        
        if 'course_duration_hours' in df_engineered.columns and 'token_price' in df_engineered.columns:
            # Value proposition (price per hour)
            df_engineered['price_per_hour'] = df_engineered['token_price'] / (df_engineered['course_duration_hours'] + 1)

        logger.info(f"Feature engineering completed. Added {len(df_engineered.columns) - len(df.columns)} new features")
        return df_engineered

    def encode_categorical_features(self, df: pd.DataFrame, encoding_method: str = 'onehot') -> pd.DataFrame:
        """
        Encode categorical features for EdTech data

        Args:
            df: DataFrame with categorical columns
            encoding_method: Encoding method ('onehot', 'label')

        Returns:
            DataFrame with encoded categorical features
        """
        df_encoded = df.copy()

        # Store original categorical columns
        original_categorical = [col for col in self.categorical_columns if col in df_encoded.columns]

        if encoding_method == 'onehot':
            # One-hot encode low-cardinality categoricals
            categorical_for_encoding = []
            for col in original_categorical:
                unique_count = df_encoded[col].nunique()
                if unique_count <= 20 and unique_count > 1:  # Only encode if 2-20 unique values
                    categorical_for_encoding.append(col)

            if categorical_for_encoding:
                # Fit and transform
                encoded_features = self.onehot_encoder.fit_transform(df_encoded[categorical_for_encoding])
                encoded_df = pd.DataFrame(
                    encoded_features,
                    columns=self.onehot_encoder.get_feature_names_out(categorical_for_encoding),
                    index=df_encoded.index
                )

                # Drop original and add encoded
                df_encoded.drop(categorical_for_encoding, axis=1, inplace=True)
                df_encoded = pd.concat([df_encoded, encoded_df], axis=1)

        elif encoding_method == 'label':
            # Label encoding for all categoricals
            for col in original_categorical:
                if col in df_encoded.columns:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))

        logger.info(f"Categorical encoding completed using {encoding_method} method")
        return df_encoded

    def select_features(self, df: pd.DataFrame, method: str = 'correlation', k: int = None) -> pd.DataFrame:
        """
        Select most relevant features for EdTech modeling

        Args:
            df: DataFrame with features
            method: Feature selection method
            k: Number of features to select

        Returns:
            DataFrame with selected features
        """
        if self.target_column not in df.columns:
            logger.warning(f"Target column {self.target_column} not found")
            return df

        # Separate features and target
        X = df.drop(self.target_column, axis=1)
        y = df[self.target_column]

        # Only use numeric columns
        numeric_features = [col for col in X.columns if X[col].dtype in [np.float64, np.int64]]

        if not numeric_features:
            logger.warning("No numeric features for selection")
            return df

        X_numeric = X[numeric_features]

        if method == 'correlation':
            # Select based on correlation with target
            correlations = X_numeric.corrwith(y).abs()
            important_features = correlations.nlargest(k or len(numeric_features)).index.tolist()

        else:
            important_features = numeric_features

        # Keep target and selected features
        selected_columns = important_features + [self.target_column]
        df_selected = df[selected_columns]

        self.feature_columns = important_features

        logger.info(f"Feature selection: selected {len(important_features)} features")
        return df_selected

    def create_modeling_dataset(self, df: pd.DataFrame,
                              test_size: float = 0.2,
                              random_state: int = 42) -> Dict[str, pd.DataFrame]:
        """
        Create train/test splits for modeling

        Args:
            df: Preprocessed DataFrame
            test_size: Proportion for testing
            random_state: Random state

        Returns:
            Dictionary with train/test splits
        """
        if self.target_column not in df.columns:
            raise ValueError(f"Target column {self.target_column} not found")

        X = df.drop(self.target_column, axis=1)
        y = df[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        splits = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'train_df': train_df,
            'test_df': test_df
        }

        logger.info(f"Created modeling dataset: {len(train_df)} train, {len(test_df)} test")
        return splits

    def process_complete_pipeline(self, database_path: str = "edtech_token_economy.db",
                                save_intermediate: bool = True) -> Dict[str, Any]:
        """
        Execute complete EdTech data preparation pipeline

        Args:
            database_path: Path to database
            save_intermediate: Whether to save intermediate results

        Returns:
            Dictionary with processed data and metadata
        """
        logger.info("Starting EdTech data preparation pipeline...")

        # Step 1: Load data
        df = self.load_edtech_data(database_path)
        if df.empty:
            raise ValueError("Failed to load EdTech data")

        # Step 2: Identify column types
        column_types = self.identify_column_types(df)

        # Step 3: Handle missing values
        df = self.handle_missing_values(df, strategy='auto')

        # Step 4: Engineer EdTech features
        df = self.engineer_edtech_features(df)

        # Step 5: Encode categorical features
        df = self.encode_categorical_features(df, encoding_method='onehot')

        # Step 6: Final cleanup - remove any remaining NaN/inf
        logger.info("Performing final data cleanup...")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Drop columns with too many NaN
        nan_pct = df.isnull().sum() / len(df)
        cols_to_drop = nan_pct[nan_pct > 0.5].index.tolist()
        if cols_to_drop:
            logger.info(f"Dropping columns with >50% NaN: {cols_to_drop}")
            df.drop(columns=cols_to_drop, inplace=True)
        
        # Fill remaining NaN with 0
        df.fillna(0, inplace=True)

        # Step 7: Select features
        df = self.select_features(df, method='correlation')

        # Step 8: Create modeling splits
        splits = self.create_modeling_dataset(df)

        # Step 9: Save processed data
        if save_intermediate:
            import os
            os.makedirs('data/processed', exist_ok=True)

            df.to_csv('data/processed/edtech_processed_data.csv', index=False)
            splits['train_df'].to_csv('data/processed/edtech_train_data.csv', index=False)
            splits['test_df'].to_csv('data/processed/edtech_test_data.csv', index=False)

            # Save feature information
            feature_info = {
                'numeric_columns': self.numeric_columns,
                'categorical_columns': self.categorical_columns,
                'datetime_columns': self.datetime_columns,
                'feature_columns': self.feature_columns,
                'target_column': self.target_column,
                'processing_timestamp': datetime.now().isoformat()
            }

            pd.Series(feature_info).to_json('data/processed/edtech_feature_info.json')

        self.processed_data = df

        results = {
            'processed_data': df,
            'splits': splits,
            'column_types': column_types,
            'feature_info': {
                'numeric_columns': self.numeric_columns,
                'categorical_columns': self.categorical_columns,
                'datetime_columns': self.datetime_columns,
                'feature_columns': self.feature_columns,
                'target_column': self.target_column
            },
            'data_quality': {
                'original_shape': df.shape,
                'final_shape': df.shape,
                'missing_values': df.isnull().sum().sum()
            }
        }

        logger.info("EdTech data preparation pipeline completed successfully!")
        return results


# Example usage and testing
if __name__ == "__main__":
    print("Testing EdTech Data Preparation Module...")

    # Initialize preprocessor
    preprocessor = EdTechDataPreprocessor(target_column='total_enrollments')

    try:
        # Process complete pipeline
        results = preprocessor.process_complete_pipeline(save_intermediate=True)

        print(f"✅ Data preparation completed")
        print(f"   Shape: {results['data_quality']['original_shape']}")
        print(f"   Features: {len(results['feature_info']['feature_columns'])}")
        print(f"   Train size: {len(results['splits']['train_df'])}")
        print(f"   Test size: {len(results['splits']['test_df'])}")

        print("\n✅ EdTech Data Preparation Module test completed!")

    except Exception as e:
        print(f"✗ Data preparation failed: {e}")

