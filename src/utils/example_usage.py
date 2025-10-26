"""
Example Usage - EdTech Token Economy Platform

This module provides examples of how to use the EdTech platform components.

Examples:
- Data generation
- Exploratory data analysis
- Business metrics calculation
- ML model training
- API usage

Author: EdTech Token Economy Pipeline
Date: October 2025
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

# Example 1: Generate EdTech Data
def example_generate_data():
    """Example: Generate synthetic EdTech data"""
    print("="*80)
    print("EXAMPLE 1: Generating EdTech Data")
    print("="*80)
    
    from data.edtech_sources import EdTechTokenEconomyGenerator
    
    # Initialize generator
    generator = EdTechTokenEconomyGenerator(db_path="edtech_example.db")
    
    # Generate data
    print("Generating learners...")
    learners = generator.generate_learners(n_learners=1000)
    print(f"✅ Generated {len(learners)} learners")
    
    print("Generating teachers...")
    teachers = generator.generate_teachers(n_teachers=50)
    print(f"✅ Generated {len(teachers)} teachers")
    
    print("Generating courses...")
    courses = generator.generate_courses(n_courses=200, teachers_df=teachers)
    print(f"✅ Generated {len(courses)} courses")
    
    print("Generating enrollments...")
    enrollments = generator.generate_enrollments(n_enrollments=5000, learners_df=learners, courses_df=courses)
    print(f"✅ Generated {len(enrollments)} enrollments")
    
    # Save to database
    generator.save_to_database()
    print("✅ Data saved to database")
    print()


# Example 2: Exploratory Data Analysis
def example_exploratory_analysis():
    """Example: Perform exploratory data analysis"""
    print("="*80)
    print("EXAMPLE 2: Exploratory Data Analysis")
    print("="*80)
    
    from analysis.exploratory import ExploratoryDataAnalyzer
    from data.edtech_database import EdTechDatabaseManager, DatabaseConfig
    
    # Load data
    config = DatabaseConfig(db_path="edtech_example.db")
    db_manager = EdTechDatabaseManager(config)
    db_manager.connect()
    
    data = db_manager.get_price_elasticity_data()
    print(f"✅ Loaded {len(data)} courses for analysis")
    
    # Initialize EDA analyzer
    eda = ExploratoryDataAnalyzer(data, target_column='total_enrollments')
    
    # Generate report
    report = eda.generate_comprehensive_report()
    print(report[:500] + "...")  # Show first 500 chars
    
    # Analyze token pricing
    pricing_analysis = eda.analyze_token_pricing()
    print(f"\n✅ Token Price Analysis:")
    print(f"   Mean: {pricing_analysis['mean_price']:.2f} tokens")
    print(f"   Range: {pricing_analysis['min_price']:.2f} - {pricing_analysis['max_price']:.2f} tokens")
    
    # Analyze enrollments
    enrollment_analysis = eda.analyze_enrollment_patterns()
    print(f"\n✅ Enrollment Analysis:")
    print(f"   Total: {enrollment_analysis['total_enrollments']:,.0f}")
    print(f"   Average per course: {enrollment_analysis['mean_enrollments']:.2f}")
    
    db_manager.disconnect()
    print()


# Example 3: Business Metrics Calculation
def example_business_metrics():
    """Example: Calculate business metrics"""
    print("="*80)
    print("EXAMPLE 3: Business Metrics Calculation")
    print("="*80)
    
    from business.understanding import TokenEconomyCostCalculator, create_default_token_economy_config
    
    # Initialize calculator
    config = create_default_token_economy_config()
    calculator = TokenEconomyCostCalculator(config)
    
    # Calculate course revenue
    course_revenue = calculator.calculate_course_revenue(
        token_price=100,
        enrollments=200
    )
    print(f"✅ Course Revenue Analysis:")
    print(f"   Total Revenue: ${course_revenue['total_usd_revenue']:,.2f}")
    print(f"   Platform Commission: ${course_revenue['platform_commission']:,.2f}")
    print(f"   Teacher Earnings: ${course_revenue['teacher_earnings']:,.2f}")
    print(f"   Net Profit: ${course_revenue['net_profit']:,.2f}")
    
    # Calculate learner LTV
    ltv = calculator.calculate_learner_lifetime_value(
        avg_tokens_spent_per_course=100,
        courses_per_year=3,
        years_active=3
    )
    print(f"\n✅ Learner Lifetime Value: ${ltv:,.2f}")
    
    # Calculate teacher ROI
    teacher_roi = calculator.calculate_teacher_roi(
        time_invested_hours=40,
        hourly_rate_alternative=50,
        total_earnings_from_course=3000
    )
    print(f"\n✅ Teacher ROI: {teacher_roi['roi_percentage']:.1f}%")
    print(f"   Effective Hourly Rate: ${teacher_roi['effective_hourly_rate']:.2f}")
    print()


# Example 4: Price Elasticity Modeling
def example_price_elasticity():
    """Example: Train price elasticity models"""
    print("="*80)
    print("EXAMPLE 4: Price Elasticity Modeling")
    print("="*80)
    
    from ml.token_elasticity_modeling import TokenPriceElasticityModeler
    from data.edtech_database import EdTechDatabaseManager, DatabaseConfig
    
    # Load data
    config = DatabaseConfig(db_path="edtech_example.db")
    db_manager = EdTechDatabaseManager(config)
    db_manager.connect()
    
    data = db_manager.get_price_elasticity_data()
    print(f"✅ Loaded {len(data)} courses")
    
    # Initialize modeler
    modeler = TokenPriceElasticityModeler()
    
    # Prepare data
    prepared_data = modeler.prepare_elasticity_data(data)
    print(f"✅ Data prepared with {len(prepared_data.columns)} features")
    
    # Train models
    print("\nTraining elasticity models...")
    results = modeler.compare_models(prepared_data)
    
    print(f"\n✅ Model Comparison:")
    for model_name, result in results.items():
        print(f"   {model_name}:")
        print(f"      R²: {result.metrics['test_r2']:.3f}")
        print(f"      RMSE: {result.metrics['test_rmse']:.3f}")
        print(f"      Elasticity: {result.metrics['price_elasticity']:.3f}")
    
    # Calculate optimal price
    optimal = modeler.calculate_optimal_token_price(
        course_features={'category': 'Programming'},
        current_price=100,
        current_enrollments=200,
        elasticity_coefficient=-1.2
    )
    
    print(f"\n✅ Price Optimization:")
    print(f"   Current Price: {optimal['current_price']} tokens")
    print(f"   Optimal Price: {optimal['optimal_price']:.2f} tokens")
    print(f"   Revenue Increase: {optimal['revenue_increase_pct']:.1f}%")
    
    db_manager.disconnect()
    print()


# Example 5: Token Strategy Analysis
def example_token_strategy():
    """Example: Generate token economy strategies"""
    print("="*80)
    print("EXAMPLE 5: Token Strategy Analysis")
    print("="*80)
    
    from business.token_strategy import TokenStrategyAnalyzer, create_default_strategy_config
    
    # Initialize analyzer
    config = create_default_strategy_config()
    analyzer = TokenStrategyAnalyzer(config)
    
    # Sample learner data
    np.random.seed(42)
    sample_learners = pd.DataFrame({
        'learner_id': range(500),
        'enrollment_propensity': np.random.uniform(0, 1, 500),
        'price_sensitivity': np.random.uniform(0, 1, 500)
    })
    
    # Calculate learner value
    learner_value = analyzer.calculate_enrollment_propensity_value(sample_learners)
    print(f"✅ Learner Value Analysis:")
    print(f"   Average LTV: ${learner_value['lifetime_value'].mean():,.2f}")
    print(f"   Average LTV:CAC Ratio: {learner_value['ltv_cac_ratio'].mean():.2f}x")
    
    # Optimize thresholds
    thresholds = analyzer.optimize_enrollment_propensity_thresholds(sample_learners)
    print(f"\n✅ Propensity Thresholds:")
    print(f"   High: {thresholds.high_propensity_threshold:.3f}")
    print(f"   Medium: {thresholds.medium_propensity_threshold:.3f}")
    print(f"   Low: {thresholds.low_propensity_threshold:.3f}")
    
    # Generate pricing strategies
    market_analysis = {'elasticity': -1.2, 'competition_level': 'medium'}
    strategies = analyzer.generate_pricing_strategies(market_analysis)
    
    print(f"\n✅ Top 3 Pricing Strategies:")
    for i, strategy in enumerate(strategies[:3], 1):
        print(f"   {i}. {strategy.strategy_name}")
        print(f"      Risk: {strategy.risk_level} | Timeline: {strategy.timeline_days} days")
        print(f"      Expected Impact: {list(strategy.expected_impact.values())}")
    print()


# Example 6: Complete Pipeline
def example_complete_pipeline():
    """Example: Run complete ML pipeline"""
    print("="*80)
    print("EXAMPLE 6: Complete ML Pipeline")
    print("="*80)
    
    from pipeline.orchestrator import PipelineOrchestrator
    
    # Initialize orchestrator
    orchestrator = PipelineOrchestrator()
    
    print("Running complete pipeline...")
    print("Note: This will take a few minutes...")
    
    try:
        results = orchestrator.run_complete_pipeline()
        
        print(f"\n✅ Pipeline Completed Successfully!")
        print(f"   Status: {results['overall_status']}")
        print(f"   Stages Completed: {len([s for s in results['stages'].values() if s['status'] == 'completed'])}")
        
        if 'model_training' in results['stages']:
            model_results = results['stages']['model_training'].get('results', {})
            if model_results:
                print(f"\n   Best Model: {model_results.get('best_model_name', 'N/A')}")
                print(f"   R² Score: {model_results.get('best_r2', 0):.3f}")
    
    except Exception as e:
        print(f"⚠️ Pipeline execution error: {e}")
    
    print()


# Main function to run all examples
def run_all_examples():
    """Run all examples"""
    print("\n" + "="*80)
    print("EDTECH TOKEN ECONOMY - EXAMPLE USAGE")
    print("="*80 + "\n")
    
    try:
        # Run examples
        example_generate_data()
        example_exploratory_analysis()
        example_business_metrics()
        example_price_elasticity()
        example_token_strategy()
        
        # Optional: Run complete pipeline (takes longer)
        # example_complete_pipeline()
        
        print("="*80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_examples()

