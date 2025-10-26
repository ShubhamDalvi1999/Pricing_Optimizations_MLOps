# Data Assessment Scripts

This directory contains comprehensive data quality and realism assessment scripts for the EdTech Token Economy project.

## Scripts Overview

### 1. `data_quality_assessment.py`
**Purpose**: Comprehensive data quality assessment
- **Data Completeness**: Missing values analysis
- **Data Validity**: Business rule validation
- **Statistical Analysis**: Distribution and outlier detection
- **Referential Integrity**: Foreign key relationships
- **Temporal Patterns**: Data freshness analysis
- **Quality Scoring**: Overall data quality grade (A-F)

### 2. `data_realism_validation.py`
**Purpose**: Validate data realism against real-world EdTech market patterns
- **Pricing Realism**: Market-appropriate pricing patterns
- **Enrollment Patterns**: Realistic enrollment distributions
- **Teacher Quality**: Quality vs pricing correlations
- **Market Dynamics**: Category-specific patterns
- **Price Elasticity**: Demand-price relationships
- **Realism Scoring**: Overall realism grade (A-F)

### 3. `model_performance_analysis.py`
**Purpose**: Analyze ML model performance and reliability
- **Model Metrics**: R², RMSE, accuracy comparison
- **Feature Importance**: Key predictive features
- **Model Stability**: Reliability and consistency checks
- **Performance Scoring**: Overall model grade (A-F)
- **Recommendations**: Model improvement suggestions

### 4. `comprehensive_assessment.py`
**Purpose**: Master script that runs all assessments
- **Integrated Analysis**: Combines all assessment types
- **Overall Scoring**: System-wide performance grade
- **Unified Reporting**: Single comprehensive report
- **Production Readiness**: Deployment recommendations

### 5. `run_assessment.py`
**Purpose**: Simple runner script for quick assessments
- **Quick Mode**: Fast assessment (skips model analysis)
- **Selective Mode**: Run specific assessments only
- **Easy Execution**: Simple command-line interface

## Usage Examples

### Quick Assessment (Recommended for Development)
```bash
python scripts/run_assessment.py --quick
```

### Full Assessment (Recommended for Production)
```bash
python scripts/run_assessment.py
```

### Specific Assessments
```bash
# Data quality only
python scripts/run_assessment.py --quality-only

# Data realism only
python scripts/run_assessment.py --realism-only

# Model performance only
python scripts/run_assessment.py --models-only
```

### Advanced Usage
```bash
# Run comprehensive assessment with custom paths
python scripts/comprehensive_assessment.py --db-path custom.db --models-dir custom_models --output report.txt

# Run individual assessments
python scripts/data_quality_assessment.py --db-path edtech_token_economy.db --output quality_report.txt
python scripts/data_realism_validation.py --db-path edtech_token_economy.db --output realism_report.txt
python scripts/model_performance_analysis.py --models-dir models --output performance_report.txt
```

## Assessment Criteria

### Data Quality Scoring
- **A (90-100)**: Excellent data quality, ready for production
- **B (80-89)**: Good data quality, minor improvements needed
- **C (70-79)**: Fair data quality, improvements recommended
- **D (60-69)**: Poor data quality, significant issues
- **F (0-59)**: Very poor data quality, not ready for production

### Data Realism Scoring
- **A (85-100)**: Highly realistic data patterns
- **B (75-84)**: Realistic data patterns
- **C (65-74)**: Moderately realistic patterns
- **D (55-64)**: Somewhat unrealistic patterns
- **F (0-54)**: Unrealistic patterns

### Model Performance Scoring
- **A (85-100)**: Excellent model performance
- **B (75-84)**: Good model performance
- **C (65-74)**: Fair model performance
- **D (55-64)**: Poor model performance
- **F (0-54)**: Very poor model performance

## Common Issues and Solutions

### Data Quality Issues
- **High Missing Values**: Regenerate data with better logic
- **Invalid Business Rules**: Fix data generation constraints
- **Referential Integrity**: Ensure proper foreign key relationships
- **Outlier Detection**: Adjust data generation parameters

### Data Realism Issues
- **Unrealistic Pricing**: Adjust price generation algorithms
- **Poor Enrollment Patterns**: Improve enrollment logic
- **Weak Correlations**: Enhance feature relationships
- **Market Concentration**: Diversify category distribution

### Model Performance Issues
- **Negative R²**: Model performing worse than baseline
- **Overfitting**: Implement regularization
- **Poor Feature Importance**: Improve feature engineering
- **High Variability**: Increase training data

## Output Files

### Reports Generated
- **Quality Report**: `quality_assessment_report.txt`
- **Realism Report**: `realism_validation_report.txt`
- **Performance Report**: `model_performance_report.txt`
- **Comprehensive Report**: `comprehensive_assessment_report.txt`

### Key Metrics Tracked
- **Completeness Score**: Percentage of non-missing data
- **Validity Score**: Percentage of valid records
- **Integrity Score**: Percentage of valid relationships
- **Realism Score**: Overall data realism assessment
- **Performance Score**: Model accuracy and reliability

## Integration with Pipeline

### Pre-Pipeline Assessment
```bash
# Check existing data quality before regeneration
python scripts/run_assessment.py --quality-only
```

### Post-Pipeline Assessment
```bash
# Full assessment after pipeline completion
python scripts/run_assessment.py
```

### Continuous Monitoring
```bash
# Quick check for ongoing development
python scripts/run_assessment.py --quick
```

## Troubleshooting

### Common Errors
1. **Database Not Found**: Ensure `edtech_token_economy.db` exists
2. **Models Not Found**: Ensure `models/` directory contains `.pkl` files
3. **Import Errors**: Ensure all dependencies are installed
4. **Permission Errors**: Check file permissions for output reports

### Dependencies Required
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `sqlite3` (built-in)
- `pickle` (built-in)

### Performance Notes
- **Quick Assessment**: ~30 seconds
- **Full Assessment**: ~2-3 minutes
- **Model Analysis**: ~1-2 minutes (depends on model count)

## Best Practices

1. **Run Quick Assessment** during development
2. **Run Full Assessment** before production deployment
3. **Address Critical Issues** (Grade D or F) immediately
4. **Monitor Continuously** with automated assessments
5. **Document Improvements** in assessment reports

## Support

For issues or questions about the assessment scripts:
1. Check the generated reports for specific issues
2. Review the troubleshooting section
3. Ensure all dependencies are properly installed
4. Verify database and model files exist and are accessible
