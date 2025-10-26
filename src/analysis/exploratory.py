"""
Exploratory Data Analysis Module - EdTech Token Economy

This module performs comprehensive exploratory data analysis (EDA)
for the EdTech token economy platform.

Features:
- Learner behavior analysis
- Course performance analysis
- Token transaction analysis
- Teacher quality analysis
- Enrollment pattern analysis

Author: EdTech Token Economy Pipeline
Date: October 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

class ExploratoryDataAnalyzer:
    """Main class for exploratory data analysis of EdTech data"""

    def __init__(self, data: pd.DataFrame, target_column: str = None):
        """
        Initialize EDA analyzer for EdTech data

        Args:
            data: DataFrame to analyze (learners, courses, enrollments, etc.)
            target_column: Name of target column for supervised analysis
        """
        self.data = data.copy()
        self.target_column = target_column
        self.numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_columns = self._identify_datetime_columns()

        logger.info(f"Initialized EDA for EdTech dataset with {len(self.data)} rows and {len(self.data.columns)} columns")

    def _identify_datetime_columns(self) -> List[str]:
        """Identify datetime columns in the dataset"""
        datetime_cols = []

        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                try:
                    pd.to_datetime(self.data[col])
                    datetime_cols.append(col)
                except:
                    pass
            elif pd.api.types.is_datetime64_any_dtype(self.data[col]):
                datetime_cols.append(col)

        return datetime_cols

    def generate_comprehensive_report(self) -> str:
        """
        Generate a comprehensive EDA report for EdTech data

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("EDTECH TOKEN ECONOMY - EXPLORATORY DATA ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Dataset Shape: {self.data.shape[0]:,} rows, {self.data.shape[1]} columns")
        report.append(f"Number of Features: {self.data.shape[1]}")
        report.append(f"Target Column: {self.target_column or 'Not specified'}")
        report.append("")

        # Dataset overview
        report.append("DATASET OVERVIEW:")
        report.append("-" * 50)
        report.append(f"Numeric Columns: {len(self.numeric_columns)}")
        report.append(f"Categorical Columns: {len(self.categorical_columns)}")
        report.append(f"Datetime Columns: {len(self.datetime_columns)}")
        report.append("")

        # EdTech-specific metrics
        report.append("EDTECH METRICS:")
        report.append("-" * 50)
        
        if 'token_price' in self.data.columns:
            report.append(f"Token Price Range: {self.data['token_price'].min():.2f} - {self.data['token_price'].max():.2f}")
            report.append(f"Average Token Price: {self.data['token_price'].mean():.2f}")
        
        if 'total_enrollments' in self.data.columns:
            report.append(f"Total Enrollments: {self.data['total_enrollments'].sum():,.0f}")
            report.append(f"Average Enrollments per Course: {self.data['total_enrollments'].mean():.2f}")
        
        if 'category' in self.data.columns:
            report.append(f"Number of Categories: {self.data['category'].nunique()}")
            report.append(f"Top Categories: {', '.join(self.data['category'].value_counts().head(3).index.tolist())}")
        
        report.append("")

        # Column details
        report.append("COLUMN DETAILS:")
        report.append("-" * 50)
        for col in self.data.columns:
            dtype = str(self.data[col].dtype)
            missing_pct = (self.data[col].isnull().sum() / len(self.data)) * 100
            unique_vals = self.data[col].nunique()

            report.append(f"{col}:")
            report.append(f"  Type: {dtype}")
            report.append(f"  Missing: {missing_pct:.2f}%")
            report.append(f"  Unique Values: {unique_vals:,}")
            report.append("")

        # Summary statistics
        if self.numeric_columns:
            report.append("NUMERIC FEATURES SUMMARY:")
            report.append("-" * 50)
            summary_stats = self.data[self.numeric_columns].describe()

            for stat in ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']:
                report.append(f"{stat.capitalize()}: {summary_stats.loc[stat].to_dict()}")

            report.append("")

        # Missing values analysis
        missing_analysis = self.analyze_missing_values()
        if missing_analysis:
            report.append("MISSING VALUES ANALYSIS:")
            report.append("-" * 50)
            for col, info in missing_analysis.items():
                report.append(f"{col}: {info['missing_count']} missing ({info['missing_percentage']:.2f}%)")
            report.append("")

        # Correlation analysis
        if len(self.numeric_columns) > 1:
            correlation_insights = self.analyze_correlations()
            if correlation_insights:
                report.append("CORRELATION INSIGHTS:")
                report.append("-" * 50)
                report.append(correlation_insights)
                report.append("")

        # Categorical analysis
        if self.categorical_columns:
            report.append("CATEGORICAL FEATURES ANALYSIS:")
            report.append("-" * 50)
            for col in self.categorical_columns[:5]:  # Top 5 categorical columns
                value_counts = self.data[col].value_counts()
                report.append(f"{col} (Top 5 values):")
                for val, count in value_counts.head().items():
                    pct = (count / len(self.data)) * 100
                    report.append(f"  {val}: {count:,} ({pct:.1f}%)")
                report.append("")

        report.append("=" * 80)
        return "\n".join(report)

    def analyze_missing_values(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze missing values in the dataset

        Returns:
            Dictionary with missing value information per column
        """
        missing_info = {}

        for col in self.data.columns:
            missing_count = self.data[col].isnull().sum()
            if missing_count > 0:
                missing_pct = (missing_count / len(self.data)) * 100
                missing_info[col] = {
                    'missing_count': missing_count,
                    'missing_percentage': missing_pct
                }

        return missing_info

    def analyze_correlations(self, method: str = 'pearson', threshold: float = 0.5) -> str:
        """
        Analyze correlations between numeric features

        Args:
            method: Correlation method ('pearson', 'spearman', 'kendall')
            threshold: Minimum absolute correlation to report

        Returns:
            Formatted string with correlation insights
        """
        if len(self.numeric_columns) < 2:
            return "Not enough numeric columns for correlation analysis"

        corr_matrix = self.data[self.numeric_columns].corr(method=method)

        insights = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]

                if abs(corr_value) >= threshold:
                    strength = "strong" if abs(corr_value) >= 0.7 else "moderate"
                    direction = "positive" if corr_value > 0 else "negative"
                    insights.append(f"  {col1} ↔ {col2}: {strength} {direction} ({corr_value:.3f})")

        if not insights:
            return f"No correlations above {threshold} threshold found"

        return "\n".join(insights)

    def analyze_token_pricing(self) -> Dict[str, Any]:
        """
        Analyze token pricing patterns specific to EdTech

        Returns:
            Dictionary with pricing insights
        """
        if 'token_price' not in self.data.columns:
            logger.warning("token_price column not found")
            return {}

        pricing_analysis = {
            'mean_price': self.data['token_price'].mean(),
            'median_price': self.data['token_price'].median(),
            'std_price': self.data['token_price'].std(),
            'min_price': self.data['token_price'].min(),
            'max_price': self.data['token_price'].max(),
            'price_range': self.data['token_price'].max() - self.data['token_price'].min()
        }

        # Price by category
        if 'category' in self.data.columns:
            pricing_analysis['price_by_category'] = self.data.groupby('category')['token_price'].agg(['mean', 'median', 'count']).to_dict()

        # Price distribution quartiles
        pricing_analysis['quartiles'] = {
            'Q1': self.data['token_price'].quantile(0.25),
            'Q2': self.data['token_price'].quantile(0.50),
            'Q3': self.data['token_price'].quantile(0.75)
        }

        logger.info(f"Token pricing analysis completed: avg=${pricing_analysis['mean_price']:.2f}")
        return pricing_analysis

    def analyze_enrollment_patterns(self) -> Dict[str, Any]:
        """
        Analyze enrollment patterns for courses

        Returns:
            Dictionary with enrollment insights
        """
        if 'total_enrollments' not in self.data.columns:
            logger.warning("total_enrollments column not found")
            return {}

        enrollment_analysis = {
            'total_enrollments': self.data['total_enrollments'].sum(),
            'mean_enrollments': self.data['total_enrollments'].mean(),
            'median_enrollments': self.data['total_enrollments'].median(),
            'std_enrollments': self.data['total_enrollments'].std()
        }

        # Enrollment by category
        if 'category' in self.data.columns:
            enrollment_analysis['by_category'] = self.data.groupby('category')['total_enrollments'].agg(['sum', 'mean', 'count']).to_dict()

        # Enrollment by difficulty
        if 'difficulty_level' in self.data.columns:
            enrollment_analysis['by_difficulty'] = self.data.groupby('difficulty_level')['total_enrollments'].agg(['sum', 'mean']).to_dict()

        logger.info(f"Enrollment analysis completed: total={enrollment_analysis['total_enrollments']:,.0f}")
        return enrollment_analysis

    def analyze_teacher_performance(self) -> Dict[str, Any]:
        """
        Analyze teacher quality and performance metrics

        Returns:
            Dictionary with teacher insights
        """
        teacher_metrics = {}

        if 'teacher_quality_score' in self.data.columns:
            teacher_metrics['quality_score'] = {
                'mean': self.data['teacher_quality_score'].mean(),
                'median': self.data['teacher_quality_score'].median(),
                'std': self.data['teacher_quality_score'].std()
            }

        if 'teacher_rating' in self.data.columns:
            teacher_metrics['rating'] = {
                'mean': self.data['teacher_rating'].mean(),
                'distribution': self.data['teacher_rating'].value_counts().to_dict()
            }

        logger.info("Teacher performance analysis completed")
        return teacher_metrics

    def create_visualization_dashboard(self, output_dir: str = "notebooks") -> Dict[str, str]:
        """
        Create comprehensive visualization dashboard for EdTech data

        Args:
            output_dir: Directory to save visualizations

        Returns:
            Dictionary mapping visualization names to file paths
        """
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        saved_files = {}

        # 1. Dataset Overview Plot
        fig_overview = self._create_dataset_overview_plot()
        overview_path = os.path.join(output_dir, "edtech_dataset_overview.png")
        fig_overview.savefig(overview_path, dpi=300, bbox_inches='tight')
        saved_files['dataset_overview'] = overview_path
        plt.close(fig_overview)

        # 2. Token Price Distribution
        if 'token_price' in self.data.columns:
            fig_price = self._create_price_distribution_plot()
            price_path = os.path.join(output_dir, "token_price_distribution.png")
            fig_price.savefig(price_path, dpi=300, bbox_inches='tight')
            saved_files['price_distribution'] = price_path
            plt.close(fig_price)

        # 3. Enrollment Analysis
        if 'total_enrollments' in self.data.columns:
            fig_enroll = self._create_enrollment_analysis_plot()
            enroll_path = os.path.join(output_dir, "enrollment_analysis.png")
            fig_enroll.savefig(enroll_path, dpi=300, bbox_inches='tight')
            saved_files['enrollment_analysis'] = enroll_path
            plt.close(fig_enroll)

        # 4. Category Analysis
        if 'category' in self.data.columns:
            fig_cat = self._create_category_analysis_plot()
            cat_path = os.path.join(output_dir, "category_analysis.png")
            fig_cat.savefig(cat_path, dpi=300, bbox_inches='tight')
            saved_files['category_analysis'] = cat_path
            plt.close(fig_cat)

        # 5. Correlation Heatmap
        if len(self.numeric_columns) > 1:
            fig_corr = self._create_correlation_heatmap()
            corr_path = os.path.join(output_dir, "correlation_heatmap.png")
            fig_corr.savefig(corr_path, dpi=300, bbox_inches='tight')
            saved_files['correlation'] = corr_path
            plt.close(fig_corr)

        # 6. Interactive Dashboard (HTML)
        if self.target_column and self.target_column in self.numeric_columns:
            dashboard_path = os.path.join(output_dir, "edtech_eda_dashboard.html")
            self._create_interactive_dashboard(dashboard_path)
            saved_files['interactive_dashboard'] = dashboard_path

        logger.info(f"Created {len(saved_files)} visualizations in {output_dir}")
        return saved_files

    def _create_dataset_overview_plot(self) -> plt.Figure:
        """Create dataset overview visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Dataset info  
        axes[0, 0].text(0.1, 0.8, f"Total Rows: {len(self.data):,}", fontsize=12, weight='bold')
        axes[0, 0].text(0.1, 0.6, f"Total Columns: {len(self.data.columns)}", fontsize=12)
        axes[0, 0].text(0.1, 0.4, f"Numeric: {len(self.numeric_columns)}", fontsize=12)
        axes[0, 0].text(0.1, 0.2, f"Categorical: {len(self.categorical_columns)}", fontsize=12)
        axes[0, 0].set_title("EdTech Dataset Overview")
        axes[0, 0].axis('off')

        # Data types distribution
        if self.numeric_columns or self.categorical_columns:
            data_types = []
            counts = []
            if self.numeric_columns:
                data_types.append('Numeric')
                counts.append(len(self.numeric_columns))
            if self.categorical_columns:
                data_types.append('Categorical')
                counts.append(len(self.categorical_columns))
            if self.datetime_columns:
                data_types.append('Datetime')
                counts.append(len(self.datetime_columns))

            axes[0, 1].pie(counts, labels=data_types, autopct='%1.1f%%', startangle=90)
            axes[0, 1].set_title("Data Types Distribution")

        # Missing values summary
        missing_pct = (self.data.isnull().sum().sum() / (len(self.data) * len(self.data.columns))) * 100
        axes[1, 0].text(0.1, 0.5, f"Overall Missing Data: {missing_pct:.2f}%", fontsize=14, weight='bold')
        axes[1, 0].set_title("Missing Data Overview")
        axes[1, 0].axis('off')

        # Memory usage
        memory_usage = self.data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        axes[1, 1].text(0.1, 0.5, f"Memory Usage: {memory_usage:.2f} MB", fontsize=14, weight='bold')
        axes[1, 1].set_title("Memory Usage")
        axes[1, 1].axis('off')

        plt.tight_layout()
        return fig

    def _create_price_distribution_plot(self) -> plt.Figure:
        """Create token price distribution plot"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Price distribution
        sns.histplot(self.data['token_price'], kde=True, ax=axes[0])
        axes[0].set_title("Token Price Distribution")
        axes[0].set_xlabel("Token Price")
        axes[0].set_ylabel("Frequency")

        # Price by category
        if 'category' in self.data.columns:
            category_prices = self.data.groupby('category')['token_price'].mean().sort_values()
            category_prices.plot(kind='barh', ax=axes[1])
            axes[1].set_title("Average Token Price by Category")
            axes[1].set_xlabel("Average Token Price")
            axes[1].set_ylabel("Category")

        plt.tight_layout()
        return fig

    def _create_enrollment_analysis_plot(self) -> plt.Figure:
        """Create enrollment analysis plot"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Enrollment distribution
        sns.histplot(self.data['total_enrollments'], kde=True, ax=axes[0])
        axes[0].set_title("Enrollment Distribution")
        axes[0].set_xlabel("Total Enrollments")
        axes[0].set_ylabel("Frequency")

        # Enrollments by category
        if 'category' in self.data.columns:
            category_enrollments = self.data.groupby('category')['total_enrollments'].sum().sort_values()
            category_enrollments.plot(kind='barh', ax=axes[1])
            axes[1].set_title("Total Enrollments by Category")
            axes[1].set_xlabel("Total Enrollments")
            axes[1].set_ylabel("Category")

        plt.tight_layout()
        return fig

    def _create_category_analysis_plot(self) -> plt.Figure:
        """Create category analysis plot"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Category distribution
        category_counts = self.data['category'].value_counts()
        category_counts.plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title("Course Distribution by Category")
        axes[0, 0].set_xlabel("Category")
        axes[0, 0].set_ylabel("Number of Courses")
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Price by category
        if 'token_price' in self.data.columns:
            self.data.boxplot(column='token_price', by='category', ax=axes[0, 1])
            axes[0, 1].set_title("Token Price Distribution by Category")
            axes[0, 1].set_xlabel("Category")
            axes[0, 1].set_ylabel("Token Price")

        # Enrollments by category
        if 'total_enrollments' in self.data.columns:
            self.data.boxplot(column='total_enrollments', by='category', ax=axes[1, 0])
            axes[1, 0].set_title("Enrollment Distribution by Category")
            axes[1, 0].set_xlabel("Category")
            axes[1, 0].set_ylabel("Total Enrollments")

        # Difficulty distribution
        if 'difficulty_level' in self.data.columns:
            difficulty_counts = self.data['difficulty_level'].value_counts()
            difficulty_counts.plot(kind='pie', ax=axes[1, 1], autopct='%1.1f%%')
            axes[1, 1].set_title("Course Distribution by Difficulty")
            axes[1, 1].set_ylabel("")

        plt.tight_layout()
        return fig

    def _create_correlation_heatmap(self) -> plt.Figure:
        """Create correlation heatmap for numeric columns"""
        # Select only numeric columns for correlation
        numeric_data = self.data[self.numeric_columns]
        
        # Remove columns with all NaN or zero variance
        numeric_data = numeric_data.loc[:, numeric_data.std() > 0]
        
        corr_matrix = numeric_data.corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title("Correlation Heatmap (EdTech Features)")

        return fig

    def _create_interactive_dashboard(self, output_path: str):
        """Create interactive dashboard with Plotly"""
        # Create subplots for interactive dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Token Price Distribution", "Enrollment Analysis", "Category Breakdown", "Price vs Enrollments"),
            specs=[[{"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "pie"}, {"type": "scatter"}]]
        )

        # Price distribution
        if 'token_price' in self.data.columns:
            fig.add_trace(
                go.Histogram(x=self.data['token_price'], name="Token Price"),
                row=1, col=1
            )

        # Enrollment distribution
        if 'total_enrollments' in self.data.columns:
            fig.add_trace(
                go.Histogram(x=self.data['total_enrollments'], name="Enrollments"),
                row=1, col=2
            )

        # Category pie chart
        if 'category' in self.data.columns:
            category_counts = self.data['category'].value_counts()
            fig.add_trace(
                go.Pie(labels=category_counts.index, values=category_counts.values, name="Categories"),
                row=2, col=1
            )

        # Price vs Enrollments scatter
        if 'token_price' in self.data.columns and 'total_enrollments' in self.data.columns:
            fig.add_trace(
                go.Scatter(x=self.data['token_price'], y=self.data['total_enrollments'],
                          mode='markers', name="Price vs Enrollments"),
                row=2, col=2
            )

        fig.update_layout(
            title="EdTech Token Economy - Interactive EDA Dashboard",
            showlegend=False,
            height=800
        )

        fig.write_html(output_path)
        logger.info(f"Interactive dashboard saved to {output_path}")

    def generate_insights_summary(self) -> str:
        """
        Generate key insights from the EDA for EdTech data

        Returns:
            Formatted insights string
        """
        insights = []
        insights.append("=" * 60)
        insights.append("KEY INSIGHTS FROM EDTECH DATA ANALYSIS")
        insights.append("=" * 60)

        # Dataset size insights
        if len(self.data) > 1000:
            insights.append("✅ Large dataset - statistically significant patterns")
        elif len(self.data) < 100:
            insights.append("⚠️ Small dataset - consider collecting more data")

        # Missing data insights
        missing_pct = (self.data.isnull().sum().sum() / (len(self.data) * len(self.data.columns))) * 100
        if missing_pct > 20:
            insights.append("⚠️ High missing data percentage - review data collection process")
        elif missing_pct == 0:
            insights.append("✅ No missing data - excellent data quality")

        # Token pricing insights
        if 'token_price' in self.data.columns:
            price_cv = (self.data['token_price'].std() / self.data['token_price'].mean()) * 100
            insights.append(f"💰 Token price variability: {price_cv:.1f}% coefficient of variation")
            
            if price_cv > 50:
                insights.append("   → High price variability - consider price segmentation strategies")

        # Enrollment insights
        if 'total_enrollments' in self.data.columns:
            low_enrollment_pct = (self.data['total_enrollments'] < 10).sum() / len(self.data) * 100
            insights.append(f"📊 Courses with low enrollments (<10): {low_enrollment_pct:.1f}%")
            
            if low_enrollment_pct > 30:
                insights.append("   → Consider marketing or pricing strategies for low-enrollment courses")

        # Category insights
        if 'category' in self.data.columns:
            n_categories = self.data['category'].nunique()
            insights.append(f"📚 Number of course categories: {n_categories}")
            
            # Category imbalance
            top_category_pct = (self.data['category'].value_counts().iloc[0] / len(self.data)) * 100
            if top_category_pct > 40:
                insights.append(f"   → One category dominates ({top_category_pct:.1f}%) - consider diversification")

        # Correlation insights
        if 'token_price' in self.data.columns and 'total_enrollments' in self.data.columns:
            corr = self.data['token_price'].corr(self.data['total_enrollments'])
            insights.append(f"🔗 Price-Enrollment correlation: {corr:.3f}")
            
            if corr < -0.3:
                insights.append("   → Negative correlation detected - price elasticity present")
            elif abs(corr) < 0.1:
                insights.append("   → Weak correlation - price may not be the primary enrollment driver")

        insights.append("")
        insights.append("RECOMMENDATIONS:")
        insights.append("-" * 30)
        insights.append("• Conduct price elasticity analysis for optimization")
        insights.append("• Implement learner segmentation for targeted strategies")
        insights.append("• Monitor enrollment patterns by category and difficulty")
        insights.append("• Consider A/B testing for pricing strategies")

        return "\n".join(insights)


# Utility functions for EdTech data profiling
def profile_edtech_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create a comprehensive profile of an EdTech DataFrame

    Args:
        df: DataFrame to profile

    Returns:
        Dictionary with profiling information
    """
    profile = {
        'shape': df.shape,
        'columns': len(df.columns),
        'rows': len(df),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'unique_values': df.nunique().to_dict(),
    }

    # Add EdTech-specific statistics
    if 'token_price' in df.columns:
        profile['token_price_stats'] = df['token_price'].describe().to_dict()
    
    if 'total_enrollments' in df.columns:
        profile['enrollment_stats'] = df['total_enrollments'].describe().to_dict()
    
    if 'category' in df.columns:
        profile['category_distribution'] = df['category'].value_counts().to_dict()

    return profile


# Example usage and testing
if __name__ == "__main__":
    print("Testing EdTech Exploratory Data Analysis Module...")

    # Create sample EdTech data for testing
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'course_id': [f'C{i:05d}' for i in range(100)],
        'token_price': np.random.uniform(50, 200, 100),
        'total_enrollments': np.random.randint(10, 500, 100),
        'category': np.random.choice(['Programming', 'Business', 'Design', 'Data Science', 'Language'], 100),
        'difficulty_level': np.random.choice(['Beginner', 'Intermediate', 'Advanced'], 100),
        'teacher_quality_score': np.random.uniform(3.0, 5.0, 100),
        'course_duration_hours': np.random.uniform(5, 40, 100)
    })

    # Initialize EDA analyzer
    eda = ExploratoryDataAnalyzer(sample_data, target_column='total_enrollments')

    # Generate comprehensive report
    report = eda.generate_comprehensive_report()
    print(report)

    # Generate insights
    insights = eda.generate_insights_summary()
    print("\n" + insights)

    # Analyze pricing
    pricing_analysis = eda.analyze_token_pricing()
    print(f"\nPricing Analysis: Mean=${pricing_analysis['mean_price']:.2f}")

    # Analyze enrollments
    enrollment_analysis = eda.analyze_enrollment_patterns()
    print(f"Enrollment Analysis: Total={enrollment_analysis['total_enrollments']:,.0f}")

    # Create visualizations
    try:
        visualizations = eda.create_visualization_dashboard()
        print(f"\nCreated {len(visualizations)} visualizations")
    except Exception as e:
        print(f"Visualization creation skipped: {e}")

    print("\nEdTech Exploratory Data Analysis Module test completed!")

