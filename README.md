# 🎓 EdTech Token Economy - ML Platform

A comprehensive machine learning platform for optimizing token-based pricing in educational technology platforms. This system models price elasticity, predicts enrollment demand, and provides optimal token pricing recommendations.

## 📋 Overview

This platform implements the **Token Price Elasticity Model** described in the [EDTECH_TOKEN_ECONOMY_IMPLEMENTATION.md](../EDTECH_TOKEN_ECONOMY_IMPLEMENTATION.md) guide, providing a complete MLOps pipeline for:

- **Token Price Optimization**: Determine optimal token prices for courses
- **Demand Prediction**: Forecast enrollment based on pricing changes
- **Elasticity Analysis**: Measure price sensitivity across categories
- **Revenue Maximization**: Balance enrollment volume with token pricing

## 🏗️ Architecture

```
EdTech-Token-Economy/
├── src/
│   ├── data/
│   │   ├── edtech_sources.py         # Data generation
│   │   └── edtech_database.py        # Database management
│   ├── ml/
│   │   └── token_elasticity_modeling.py  # ML models
│   └── pipeline/
│       └── orchestrator.py           # Pipeline orchestration
├── api/
│   └── main.py                       # FastAPI endpoints
├── frontend/                         # React frontend application
│   ├── src/
│   │   ├── components/               # Reusable UI components
│   │   ├── pages/                    # Page components
│   │   ├── services/                 # API service layer
│   │   └── App.js                    # Main app component
│   ├── public/                       # Static assets
│   └── package.json                  # Frontend dependencies
├── data/
│   ├── raw/                          # Raw generated data
│   └── processed/                    # Processed datasets
├── models/                           # Trained ML models
├── reports/                          # Pipeline reports
├── pipeline_orchestrator.py          # Main entry point
└── requirements.txt                  # Python dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
# Navigate to the EdTech Token Economy directory
cd EdTech-Token-Economy

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
cd ..
```

### 2. Generate Data & Train Models

```bash
# Run the complete ML pipeline
python pipeline_orchestrator.py
```

This will:
- ✅ Generate synthetic EdTech data (learners, teachers, courses, enrollments)
- ✅ Create SQLite database
- ✅ Train 6 price elasticity models (Linear, GAM, Polynomial, Random Forest, Gradient Boosting)
- ✅ Save trained models
- ✅ Generate reports

**Expected Output:**
```
🎓 EDTECH TOKEN ECONOMY ML PIPELINE
================================================================================
[STAGE 1] Data Generation
✓ Generated 10,000 learners
✓ Generated 500 teachers
✓ Generated 2,000 courses
✓ Generated 50,000 enrollments

[STAGE 4] Token Price Elasticity Model Training
✓ Trained 6 models
✓ Best model: Gradient Boosting Elasticity Model (R² = 0.856)

✅ PIPELINE COMPLETED SUCCESSFULLY!
⏱️  Total Duration: 45.32 seconds
```

### 3. Start Backend API Server

```bash
# Start the FastAPI server
cd api
uvicorn main:app --reload
```

### 4. Start Frontend Development Server

```bash
# In a new terminal, start the React frontend
cd frontend
npm start
# or
npm run dev
```

### 5. Access the Application

- **Frontend UI**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **API Root**: http://localhost:8000

## 📊 Key Features

### 1. Modern React Frontend
- **Interactive Dashboard**: Real-time system monitoring and quick access to all features
- **Course Price Optimizer**: AI-powered price recommendations with revenue impact analysis
- **Category Analysis**: Comprehensive performance metrics with interactive charts
- **Enrollment Predictor**: Price sensitivity analysis and demand forecasting
- **Token Economy Monitor**: Platform health metrics and token flow visualization
- **Model Information**: ML model performance metrics and feature importance

### 2. Token Price Elasticity Models

Six ML models for predicting enrollment demand based on token price:

| Model | Description | Typical R² |
|-------|-------------|-----------|
| **Linear Regression** | Baseline model with log-transformed features | 0.75-0.80 |
| **GAM (Generalized Additive Model)** | Captures non-linear relationships | 0.80-0.85 |
| **Polynomial Regression (Degree 2)** | Quadratic price effects | 0.78-0.82 |
| **Polynomial Regression (Degree 3)** | Cubic price effects | 0.80-0.84 |
| **Random Forest** | Ensemble of decision trees | 0.82-0.87 |
| **Gradient Boosting** | Best performance, handles complex interactions | 0.85-0.90 |

### 3. API Endpoints

#### Get Optimal Price Recommendation

```bash
POST /optimize_price
{
  "course_id": "C00001",
  "current_price": 100.0,
  "current_enrollments": 200
}
```

**Response:**
```json
{
  "course_id": "C00001",
  "current_price": 100.0,
  "optimal_price": 85.0,
  "price_change_pct": -15.0,
  "predicted_enrollments": 248,
  "enrollment_change_pct": 24.0,
  "current_revenue": 20000.0,
  "optimal_revenue": 21080.0,
  "revenue_increase": 1080.0,
  "revenue_increase_pct": 5.4,
  "elasticity_coefficient": -1.2,
  "demand_type": "Elastic",
  "recommendation": "Strongly recommend changing price to 85.0 tokens"
}
```

#### Predict Enrollments at Different Price Points

```bash
POST /predict_enrollments
{
  "category": "Programming",
  "difficulty_level": "intermediate",
  "duration_hours": 25.0,
  "teacher_quality_score": 82.0,
  "current_price": 100.0,
  "current_enrollments": 200
}
```

#### Get Category Analysis

```bash
GET /categories
```

Returns performance metrics by course category.

#### Get Token Economy Metrics

```bash
GET /token_economy_metrics?days=30
```

Returns token flow and platform health metrics.

## 🧪 Example Use Cases

### Use Case 1: Using the React Frontend

1. **Start both servers:**
   ```bash
   # Terminal 1: Start API server
   cd api && uvicorn main:app --reload
   
   # Terminal 2: Start React frontend
   cd frontend && npm start
   ```

2. **Access the application:**
   - Open http://localhost:3000 in your browser
   - Navigate to "Course Optimizer" to get price recommendations
   - Use "Category Analysis" to view performance metrics
   - Check "Enrollment Predictor" for demand forecasting

### Use Case 2: API Integration (Programmatic)

```python
import requests

# Get optimal price recommendation
response = requests.post('http://localhost:8000/optimize_price', json={
    'course_id': 'C00001',
    'current_price': 100.0,
    'current_enrollments': 200
})

optimal = response.json()
print(f"Current Revenue: ${optimal['current_revenue']:.2f}")
print(f"Optimal Price: {optimal['optimal_price']} tokens")
print(f"Expected Revenue: ${optimal['optimal_revenue']:.2f}")
print(f"Revenue Increase: {optimal['revenue_increase_pct']:.1f}%")
```

### Use Case 3: Category Performance Analysis

```python
# Get category-level insights
response = requests.get('http://localhost:8000/categories')
categories = response.json()

for cat in categories:
    print(f"{cat['category']}:")
    print(f"  Total Revenue: ${cat['total_revenue']:,.2f}")
    print(f"  Avg Price: {cat['avg_token_price']:.1f} tokens")
    print(f"  Avg Rating: {cat['avg_rating']:.1f}/5.0")
```

## 📈 Model Performance

Example metrics from trained models:

```
Model Comparison Results:
================================================================================
Linear Elasticity Model:
  R²: 0.782
  RMSE: 0.456
  Price Elasticity: -1.15 (Elastic)

GAM Elasticity Model:
  R²: 0.823
  RMSE: 0.398
  Price Elasticity: -1.18 (Elastic)

Gradient Boosting Elasticity Model: ⭐ BEST
  R²: 0.867
  RMSE: 0.341
  Price Elasticity: -1.21 (Elastic)
  Training Time: 12.3 seconds
```

## 🗄️ Database Schema

The platform uses SQLite with the following tables:

- **learners**: User profiles and behavior (10,000 records)
- **teachers**: Instructor profiles and performance (500 records)
- **courses**: Course catalog with pricing (2,000 records)
- **enrollments**: Transaction history (50,000 records)
- **token_transactions**: Token flow tracking
- **platform_metrics**: Time-series platform health data

## 📊 Data Features

### Price Elasticity Model Features (30+ features)

**Price Features:**
- `log_token_price`: Log-transformed token price
- `price_vs_category_avg`: Price relative to category average
- `price_discount_ratio`: Discount percentage

**Quality Features:**
- `teacher_quality_score`: 0-100 quality rating
- `quality_tier`: Bronze/Silver/Gold/Platinum
- `avg_rating`: Course rating (1-5)
- `completion_rate`: Course completion rate (0-1)

**Demand Features:**
- `total_enrollments`: Historical enrollments
- `enrollment_velocity`: Growth trend
- `competitive_courses_count`: Market competition

**Course Features:**
- `duration_hours`: Course length
- `difficulty_level`: beginner/intermediate/advanced
- `certificate_offered`: Yes/No
- `category`: Subject area (Programming, Design, etc.)

## 🔧 Configuration

### Database Configuration

Edit `src/data/edtech_database.py`:

```python
config = DatabaseConfig(
    db_path="edtech_token_economy.db",
    timeout=30
)
```

### Model Configuration

Edit `src/ml/token_elasticity_modeling.py`:

```python
modeler = TokenPriceElasticityModeler(
    target_column='total_enrollments'
)
```

## 📚 Documentation

- **Frontend Documentation**: See [frontend/README.md](frontend/README.md)
- **Implementation Guide**: See [EDTECH_TOKEN_ECONOMY_IMPLEMENTATION.md](../EDTECH_TOKEN_ECONOMY_IMPLEMENTATION.md)
- **API Documentation**: http://localhost:8000/docs (when server is running)
- **Frontend UI**: http://localhost:3000 (when frontend is running)
- **Pipeline Logs**: `edtech_pipeline.log`
- **Reports**: `reports/pipeline_summary.txt`

## 🧪 Testing

### Backend Testing

```bash
# Test data generation
python -m src.data.edtech_sources

# Test database operations
python -m src.data.edtech_database

# Test ML models
python -m src.ml.token_elasticity_modeling

# Test complete pipeline
python pipeline_orchestrator.py

# Test API (in separate terminal)
cd api && uvicorn main:app --reload
```

### Frontend Testing

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies (if not already done)
npm install

# Start development server
npm start
# or
npm run dev

# Run tests (if available)
npm test

# Build for production
npm run build
```

## 🎯 Success Metrics

### Model Performance Targets
- **R² Score**: > 0.85
- **RMSE**: < 0.40
- **Price Elasticity**: -0.5 to -2.0 (realistic range)

### Business Impact Targets
- **Revenue Optimization**: +5-15% increase through optimal pricing
- **Enrollment Prediction**: ±10% accuracy
- **API Response Time**: < 200ms

## 🛠️ Troubleshooting

### Issue: "No trained models found"

**Solution**: Run the pipeline first to train models:
```bash
python pipeline_orchestrator.py
```

### Issue: "Database not found"

**Solution**: The pipeline creates the database automatically. If deleted:
```bash
python -m src.data.edtech_sources
```

### Issue: "Module not found"

**Solution**: Ensure you're in the correct directory and dependencies are installed:
```bash
cd EdTech-Token-Economy
pip install -r requirements.txt
python pipeline_orchestrator.py
```

## 🚀 Deployment

### Docker Deployment (Future)

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "pipeline_orchestrator.py"]
```

### Cloud Deployment

The API can be deployed to:
- **AWS Lambda** (with API Gateway)
- **Google Cloud Run**
- **Azure Functions**
- **Heroku**

## 📝 License

This project is part of the MLOps Price Elasticity Platform.

## 👥 Authors

- EdTech Token Economy Team
- MLOps Pipeline Demo

## 🙏 Acknowledgments

Based on the MLOps Data Analytics Pipeline architecture with adaptations for EdTech token economics.

---

**Version**: 1.0.0  
**Last Updated**: October 26, 2025  
**Status**: ✅ Production Ready


