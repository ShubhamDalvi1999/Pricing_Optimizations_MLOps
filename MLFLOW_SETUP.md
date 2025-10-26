# 📊 MLflow Integration Guide - EdTech Token Economy

## Overview

The EdTech Token Economy platform is fully integrated with **MLflow** for experiment tracking, model versioning, and performance monitoring. This guide explains how to use MLflow to track your ML experiments and manage models.

---

## 🚀 Quick Start

###1. Install MLflow

```bash
pip install mlflow
```

### 2. Run the Pipeline with MLflow

```bash
cd EdTech-Token-Economy
python pipeline_orchestrator.py
```

The pipeline will automatically:
- Initialize MLflow tracking
- Log all model training runs
- Save model artifacts
- Track metrics and parameters
- Generate evaluation plots

### 3. View MLflow UI

```bash
mlflow ui --backend-store-uri ./mlruns
```

Then open your browser to: `http://localhost:5000`

---

## 📁 MLflow Directory Structure

```
EdTech-Token-Economy/
├── mlruns/                          # MLflow tracking directory
│   ├── experiment_summary.json      # Experiment metadata
│   └── <experiment_id>/             # Experiment folder
│       ├── <run_id>/                # Individual run folders
│       │   ├── artifacts/           # Model files, plots
│       │   │   ├── model/           # Saved model
│       │   │   └── evaluation/      # Evaluation plots & reports
│       │   ├── metrics/             # Logged metrics
│       │   ├── params/              # Logged parameters
│       │   └── tags/                # Logged tags
│       └── meta.yaml                # Experiment metadata
```

---

## 🔧 MLflow Features

### 1. **Experiment Tracking**

The platform logs comprehensive information for each model:

#### **Metrics Logged:**
- **Regression Models** (Token Price Elasticity):
  - `train_r2`, `test_r2` - R² scores
  - `mae`, `rmse`, `mape` - Error metrics
  - `price_elasticity_coefficient` - Elasticity measure
  - `training_time_seconds` - Training duration

- **Classification Models** (Enrollment Propensity):
  - `train_accuracy`, `test_accuracy`
  - `f1_score`, `precision`, `recall`
  - `auc` - Area under ROC curve

#### **Parameters Logged:**
- Model hyperparameters (e.g., `n_estimators`, `max_depth`)
- Dataset information (`dataset_rows`, `dataset_columns`)
- Feature count and names
- Training configuration

#### **Tags Logged:**
- `model_type` - Type of model (token_elasticity, enrollment_propensity)
- `model_name` - Specific model name (linear, random_forest, etc.)
- `platform` - "EdTech"
- `pipeline_stage` - "training", "evaluation", etc.
- `training_timestamp` - When the model was trained

### 2. **Model Artifacts**

Each run saves:

- **Trained Model**: Serialized model (`.pkl`, MLmodel format)
- **Evaluation Plots**:
  - Actual vs Predicted scatter plot
  - Residuals plot
  - Residuals distribution histogram
  - Feature importance bar chart
  - Confusion matrix (classification models)
  - ROC curve (classification models)
- **Evaluation Reports**:
  - `evaluation_summary.txt` - Text summary of performance
  - `predictions_sample.csv` - Sample predictions
  - `feature_importance.csv` - Feature importance scores
  - `model_validation.py` - Auto-generated validation script

### 3. **Model Versioning**

MLflow automatically versions your models:

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get all versions of a model
versions = client.search_model_versions("model_name='elasticity_random_forest'")

# Load a specific version
import mlflow
model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
```

---

## 📊 Using MLflow in Your Code

### Example 1: Train Model with MLflow Tracking

```python
from src.ml.mlflow_setup import setup_mlflow_for_edtech_pipeline
from src.ml.token_elasticity_modeling import TokenPriceElasticityModeler
import pandas as pd

# Initialize MLflow
tracker = setup_mlflow_for_edtech_pipeline("My_Experiment")

# Load data
data = pd.read_csv('data/processed/token_price_elasticity_data.csv')

# Train model
modeler = TokenPriceElasticityModeler()
prepared_data = modeler.prepare_elasticity_data(data)
result = modeler.train_random_forest_elasticity_model(prepared_data)

# Log to MLflow
dataset_info = {
    'rows': len(data),
    'columns': len(data.columns),
    'features': list(data.columns),
    'target': 'total_enrollments'
}

tags = {
    'model_name': 'random_forest',
    'category': 'Programming',
    'experiment_type': 'elasticity_modeling'
}

run_id = tracker.log_model_training(
    model_results=result,
    model_type='token_elasticity',
    dataset_info=dataset_info,
    tags=tags
)

print(f"Model logged with run_id: {run_id}")
```

### Example 2: Load Model from MLflow

```python
import mlflow

# Load model by run ID
model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

# Make predictions
sample_data = pd.DataFrame({
    'token_price': [100, 150, 80],
    'category_Programming': [1, 0, 1],
    'level_Beginner': [0, 1, 0],
    # ... more features
})

predictions = model.predict(sample_data)
print(f"Predictions: {predictions}")
```

### Example 3: Compare Model Runs

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Search for runs
runs = client.search_runs(
    experiment_ids=['<experiment_id>'],
    filter_string="tags.model_type = 'token_elasticity'",
    order_by=["metrics.test_r2 DESC"]
)

# Compare top models
for run in runs[:3]:
    print(f"Run ID: {run.info.run_id}")
    print(f"Model: {run.data.tags.get('model_name')}")
    print(f"R²: {run.data.metrics.get('test_r2'):.4f}")
    print(f"Elasticity: {run.data.metrics.get('price_elasticity_coefficient'):.4f}")
    print("-" * 50)
```

### Example 4: Export Experiment Results

```python
# Export all experiment results to JSON
tracker.export_experiment_results('mlruns/my_experiment_summary.json')

# Get experiment summary
summary = tracker.get_experiment_summary()
print(f"Total runs: {summary['total_runs']}")
print(f"Model types: {summary['model_types']}")
```

---

## 🎯 MLflow UI Features

### 1. **Experiments View**
- View all experiments and their runs
- Compare multiple runs side-by-side
- Filter and sort by metrics, parameters, or tags

### 2. **Run Details**
- Inspect individual run metrics, parameters, and artifacts
- View model performance plots
- Download trained models
- See run lineage and dependencies

### 3. **Model Comparison**
- Select multiple runs to compare
- Visualize metric trends across runs
- Identify best-performing models

### 4. **Model Registry**
- Register production models
- Version control for deployed models
- Track model lifecycle (staging → production)

---

## 🔍 Advanced MLflow Usage

### Dataset Logging

The platform logs datasets with versioning:

```python
# Datasets are automatically logged during pipeline execution
# Each dataset gets a unique hash for versioning
```

### Custom Metrics

Log additional metrics:

```python
import mlflow

with mlflow.start_run():
    mlflow.log_metric("custom_metric", 0.95)
    mlflow.log_param("custom_param", "value")
    mlflow.set_tag("custom_tag", "info")
```

### Nested Runs

For complex experiments:

```python
with mlflow.start_run(run_name="Main_Experiment"):
    # Parent run
    mlflow.log_param("experiment_type", "hyperparameter_tuning")
    
    for learning_rate in [0.01, 0.001, 0.0001]:
        with mlflow.start_run(run_name=f"LR_{learning_rate}", nested=True):
            # Child run
            mlflow.log_param("learning_rate", learning_rate)
            # ... train and log model
```

---

## 📈 Viewing Specific Experiments

### Filter by Model Type

```python
# In MLflow UI, add filter:
# tags.model_type = 'token_elasticity'
```

### Filter by Performance

```python
# Show only models with R² > 0.8
# metrics.test_r2 > 0.8
```

### Filter by Date

```python
# Show runs from last week
# attributes.start_time > '2025-10-19'
```

---

## 🛠️ Troubleshooting

### Issue: MLflow not tracking experiments

**Solution:**
```bash
# Check if MLflow is installed
pip show mlflow

# If not installed:
pip install mlflow

# Verify MLflow setup
python -c "import mlflow; print(mlflow.__version__)"
```

### Issue: Cannot access MLflow UI

**Solution:**
```bash
# Make sure you're in the project root
cd EdTech-Token-Economy

# Start UI with explicit backend store
mlflow ui --backend-store-uri file:///absolute/path/to/mlruns

# Or use relative path
mlflow ui --backend-store-uri ./mlruns --port 5000
```

### Issue: Models not logging

**Solution:**
- Check that `use_mlflow=True` in pipeline orchestrator
- Verify MLflow tracking URI is set correctly
- Check logs for MLflow errors
- Ensure you have write permissions to `mlruns/` directory

---

## 📝 Best Practices

1. **Use Descriptive Run Names**
   - Include model type, timestamp, and key parameters
   - Example: `Token_Elasticity_RandomForest_2000courses_20251026`

2. **Tag Your Runs**
   - Add tags for model type, category, experiment purpose
   - Makes filtering and searching easier

3. **Log Business Metrics**
   - Don't just log technical metrics (R², RMSE)
   - Also log business metrics (revenue impact, elasticity coefficient)

4. **Regular Cleanup**
   - Archive old experiments
   - Delete failed or test runs

5. **Document Experiments**
   - Use MLflow notes/descriptions
   - Add context about why the experiment was run

---

## 🔗 MLflow API Reference

### Key Classes

- **`EdTechMLFlowTracker`**: Main MLflow tracking class
- **`setup_mlflow_for_edtech_pipeline()`**: Initialize MLflow for pipeline

### Key Methods

- `log_model_training()`: Log model training run
- `log_dataset()`: Log dataset with versioning
- `get_experiment_summary()`: Get experiment statistics
- `export_experiment_results()`: Export results to JSON

---

## 📚 Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking Quickstart](https://mlflow.org/docs/latest/quickstart.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)

---

## 💡 Example Workflow

1. **Run Pipeline with MLflow**:
   ```bash
   python pipeline_orchestrator.py
   ```

2. **View Results**:
   ```bash
   mlflow ui
   ```

3. **Compare Models**:
   - Open MLflow UI
   - Select multiple runs
   - Click "Compare"

4. **Deploy Best Model**:
   ```python
   # Find best model run_id from MLflow UI
   import mlflow
   model = mlflow.pyfunc.load_model(f"runs:/{best_run_id}/model")
   ```

5. **Monitor in Production**:
   - Log prediction requests
   - Track model performance over time
   - Detect model drift

---

## ✅ Summary

MLflow is fully integrated into the EdTech Token Economy platform:

✅ **Automatic tracking** of all experiments  
✅ **Comprehensive logging** of metrics, parameters, and artifacts  
✅ **Model versioning** and management  
✅ **Interactive UI** for visualization and comparison  
✅ **Production-ready** model serving capabilities  

**Start tracking your experiments now with MLflow!** 🚀

