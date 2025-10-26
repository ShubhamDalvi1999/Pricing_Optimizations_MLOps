"""
EdTech Token Economy API

FastAPI endpoints for the EdTech Token Economy platform.
Provides access to token price elasticity models and recommendations.

Author: EdTech Token Economy Team
Date: October 2025
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import pickle
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.edtech_database import EdTechDatabaseManager, DatabaseConfig
from src.ml.token_elasticity_modeling import TokenPriceElasticityModeler

# Create FastAPI app
app = FastAPI(
    title="EdTech Token Economy API",
    description="API for token price elasticity modeling and optimization",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class CourseFeatures(BaseModel):
    category: str = Field(..., example="Programming")
    subcategory: str = Field(..., example="Python")
    difficulty_level: str = Field(..., example="intermediate")
    duration_hours: float = Field(..., example=25.0)
    teacher_quality_score: float = Field(..., example=82.0)
    current_price: float = Field(..., example=100.0)
    current_enrollments: int = Field(..., example=200)


class PriceOptimizationRequest(BaseModel):
    course_id: str = Field(..., example="C00001")
    current_price: float = Field(..., example=100.0)
    current_enrollments: int = Field(..., example=200)


class PriceOptimizationResponse(BaseModel):
    course_id: str
    current_price: float
    optimal_price: float
    price_change_pct: float
    predicted_enrollments: int
    enrollment_change_pct: float
    current_revenue: float
    optimal_revenue: float
    revenue_increase: float
    revenue_increase_pct: float
    elasticity_coefficient: float
    demand_type: str
    recommendation: str


class CategoryAnalysis(BaseModel):
    category: str
    total_courses: int
    total_enrollments: int
    total_revenue: float
    avg_token_price: float
    avg_rating: float
    avg_completion_rate: float


# Global variables for caching
db_manager = None
best_model = None


def get_db_manager():
    """Get or create database manager"""
    global db_manager
    if db_manager is None:
        # Try both current directory and parent directory for database
        db_paths = ["edtech_token_economy.db", "../edtech_token_economy.db"]
        db_path = None
        
        for path in db_paths:
            if os.path.exists(path):
                db_path = path
                break
        
        if db_path is None:
            raise HTTPException(status_code=500, detail="Database not found. Please run the pipeline first.")
        
        config = DatabaseConfig(db_path=db_path)
        db_manager = EdTechDatabaseManager(config)
        db_manager.connect()
    return db_manager


def load_best_model():
    """Load the best trained model"""
    global best_model
    if best_model is None:
        # Try to load models in order of preference
        # Look in both current directory and parent directory
        # Start with simpler models that are more compatible
        model_files = [
            'models/elasticity_linear.pkl',
            '../models/elasticity_linear.pkl',
            'models/elasticity_polynomial_deg2.pkl',
            '../models/elasticity_polynomial_deg2.pkl',
            'models/elasticity_random_forest.pkl',
            '../models/elasticity_random_forest.pkl',
            'models/elasticity_gradient_boosting.pkl',
            '../models/elasticity_gradient_boosting.pkl'
        ]
        
        for model_file in model_files:
            if os.path.exists(model_file):
                try:
                    with open(model_file, 'rb') as f:
                        best_model = pickle.load(f)
                        print(f"✓ Loaded model from {model_file}")
                        break
                except Exception as e:
                    print(f"✗ Failed to load {model_file}: {e}")
                    continue
        
        if best_model is None:
            raise HTTPException(status_code=500, detail="No trained models found. Please run the pipeline first.")
    
    return best_model


@app.get("/")
def read_root():
    """API root endpoint"""
    return {
        "message": "EdTech Token Economy API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "courses": "/courses",
            "categories": "/categories",
            "optimize_price": "/optimize_price",
            "predict_enrollments": "/predict_enrollments",
            "model_info": "/model_info"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        db = get_db_manager()
        db_status = "connected"
    except Exception as e:
        db_status = "disconnected"
        db_error = str(e)
    
    try:
        # Test model loading (but don't fail if models can't load)
        model = load_best_model()
        model_status = "loaded"
    except Exception as e:
        model_status = "not loaded"
        model_error = str(e)
    
    # Return healthy if database is connected, even if models can't load
    if db_status == "connected":
        return {
            "status": "healthy",
            "database": "connected",
            "model": model_status,
            "timestamp": datetime.now().isoformat()
        }
    else:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "model": model_status,
            "error": f"Database: {db_error}" if 'db_error' in locals() else f"Model: {model_error}" if 'model_error' in locals() else "Unknown error",
            "timestamp": datetime.now().isoformat()
        }


@app.get("/courses", response_model=List[Dict])
def get_courses(
    category: Optional[str] = Query(None, description="Filter by category"),
    min_price: Optional[float] = Query(None, description="Minimum token price"),
    max_price: Optional[float] = Query(None, description="Maximum token price"),
    limit: int = Query(100, description="Maximum number of results")
):
    """Get courses with optional filtering"""
    try:
        db = get_db_manager()
        
        # Debug: Check if courses table exists
        tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
        tables_df = db.execute_query(tables_query)
        print(f"Available tables: {tables_df['name'].tolist()}")
        
        query = "SELECT * FROM courses WHERE 1=1"
        params = []
        
        if category:
            query += " AND category = ?"
            params.append(category)
        
        if min_price is not None:
            query += " AND token_price >= ?"
            params.append(min_price)
        
        if max_price is not None:
            query += " AND token_price <= ?"
            params.append(max_price)
        
        query += f" LIMIT {limit}"
        
        print(f"Executing query: {query}")
        df = db.execute_query(query, tuple(params) if params else None)
        print(f"Query returned {len(df)} rows")
        return df.to_dict('records')
    
    except Exception as e:
        print(f"Error in courses endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/categories", response_model=List[CategoryAnalysis])
def get_category_analysis():
    """Get performance analysis by category"""
    try:
        db = get_db_manager()
        df = db.get_category_analysis()
        
        results = []
        for _, row in df.iterrows():
            results.append(CategoryAnalysis(
                category=row['category'],
                total_courses=int(row['total_courses']),
                total_enrollments=int(row['total_enrollments']),
                total_revenue=float(row['total_revenue']),
                avg_token_price=float(row['avg_token_price']),
                avg_rating=float(row['avg_rating']) if pd.notna(row['avg_rating']) else 0.0,
                avg_completion_rate=float(row['avg_completion_rate']) if pd.notna(row['avg_completion_rate']) else 0.0
            ))
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize_price", response_model=PriceOptimizationResponse)
def optimize_course_price(request: PriceOptimizationRequest):
    """Get optimal token price recommendation for a course"""
    try:
        db = get_db_manager()
        model_data = load_best_model()
        
        # Get course details
        course_query = "SELECT * FROM courses WHERE course_id = ?"
        course_df = db.execute_query(course_query, (request.course_id,))
        
        if course_df.empty:
            raise HTTPException(status_code=404, detail=f"Course {request.course_id} not found")
        
        course = course_df.iloc[0]
        
        # Get actual enrollments from database
        enrollments_query = "SELECT COUNT(*) as enrollment_count FROM enrollments WHERE course_id = ?"
        enrollments_df = db.execute_query(enrollments_query, (request.course_id,))
        actual_enrollments = int(enrollments_df.iloc[0]['enrollment_count']) if not enrollments_df.empty else 0
        
        # Use actual enrollments if provided, otherwise use database value
        current_enrollments = request.current_enrollments if request.current_enrollments > 0 else actual_enrollments
        
        # Get elasticity coefficient from model metrics
        elasticity_coefficient = model_data['metrics'].get('price_elasticity', -1.2)
        
        # Calculate optimal price
        modeler = TokenPriceElasticityModeler()
        optimal = modeler.calculate_optimal_token_price(
            course_features={
                'category': course['category'],
                'difficulty_level': course['difficulty_level']
            },
            current_price=request.current_price,
            current_enrollments=current_enrollments,
            elasticity_coefficient=elasticity_coefficient
        )
        
        # Generate recommendation
        if optimal['revenue_increase_pct'] > 5:
            recommendation = f"Strongly recommend changing price to {optimal['optimal_price']} tokens for {optimal['revenue_increase_pct']:.1f}% revenue increase"
        elif optimal['revenue_increase_pct'] > 0:
            recommendation = f"Consider changing price to {optimal['optimal_price']} tokens for marginal revenue increase"
        else:
            recommendation = "Current price is optimal"
        
        return PriceOptimizationResponse(
            course_id=request.course_id,
            current_price=optimal['current_price'],
            optimal_price=optimal['optimal_price'],
            price_change_pct=optimal['price_change_pct'],
            predicted_enrollments=optimal['predicted_enrollments'],
            enrollment_change_pct=optimal['enrollment_change_pct'],
            current_revenue=optimal['current_revenue'],
            optimal_revenue=optimal['optimal_revenue'],
            revenue_increase=optimal['revenue_increase'],
            revenue_increase_pct=optimal['revenue_increase_pct'],
            elasticity_coefficient=optimal['elasticity_coefficient'],
            demand_type=optimal['demand_type'],
            recommendation=recommendation
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_enrollments")
def predict_enrollments(features: CourseFeatures):
    """Predict enrollments for a course at different price points"""
    try:
        model_data = load_best_model()
        
        # Get elasticity coefficient
        elasticity_coefficient = model_data['metrics'].get('price_elasticity', -1.2)
        
        # Calculate predictions at different price points
        base_price = features.current_price
        price_points = [
            base_price * 0.7,  # -30%
            base_price * 0.85,  # -15%
            base_price,  # Current
            base_price * 1.15,  # +15%
            base_price * 1.30   # +30%
        ]
        
        predictions = []
        for price in price_points:
            price_change = (price - base_price) / base_price
            enrollment_change = elasticity_coefficient * price_change
            predicted_enrollments = int(features.current_enrollments * (1 + enrollment_change))
            predicted_revenue = price * predicted_enrollments
            
            predictions.append({
                'token_price': round(price, 2),
                'price_change_pct': round(price_change * 100, 1),
                'predicted_enrollments': max(0, predicted_enrollments),
                'predicted_revenue': round(predicted_revenue, 2)
            })
        
        return {
            'course_features': features.dict(),
            'elasticity_coefficient': elasticity_coefficient,
            'predictions': predictions
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model_info")
def get_model_info():
    """Get information about the loaded model"""
    try:
        model_data = load_best_model()
        
        return {
            'metrics': model_data.get('metrics', {}),
            'parameters': model_data.get('parameters', {}),
            'feature_importance': {
                k: float(v) for k, v in list(model_data.get('feature_importance', {}).items())[:10]
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/token_economy_metrics")
def get_token_economy_metrics(days: int = Query(30, description="Number of days")):
    """Get token economy health metrics"""
    try:
        db = get_db_manager()
        metrics = db.get_token_economy_metrics(days=days)
        
        return {
            'period_days': days,
            'metrics': metrics.to_dict('records'),
            'summary': {
                'total_tokens_burned': float(metrics['daily_tokens_burned'].sum()),
                'avg_daily_enrollments': float(metrics['daily_enrollments'].mean()),
                'total_revenue': float(metrics['daily_platform_revenue'].sum())
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting EdTech Token Economy API...")
    print("📚 Documentation: http://localhost:8000/docs")
    print("🔍 Health check: http://localhost:8000/health")
    uvicorn.run(app, host="0.0.0.0", port=8000)


