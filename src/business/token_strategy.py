"""
Token Economy Strategy Module - ROI and Business Strategy

This module provides comprehensive ROI analysis, enrollment optimization,
pricing strategies, and business recommendations for the EdTech token economy.

Features:
- Enrollment propensity optimization
- Dynamic pricing strategies
- Market segmentation
- Token portfolio optimization
- Churn prevention strategies

Author: EdTech Token Economy Pipeline
Date: October 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
from scipy import optimize
from sklearn.metrics import roc_auc_score

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

@dataclass
class TokenStrategyConfig:
    """Configuration for token economy strategy analysis"""
    avg_token_price: float
    avg_enrollments_per_course: int
    platform_commission_rate: float
    learner_acquisition_cost: float
    teacher_payout_percentage: float
    churn_rate_monthly: float
    discount_rate_annual: float

@dataclass
class EnrollmentThresholds:
    """Optimal thresholds for enrollment propensity scoring"""
    high_propensity_threshold: float
    medium_propensity_threshold: float
    low_propensity_threshold: float
    expected_conversion_rates: Dict[str, float]
    expected_revenue_per_segment: Dict[str, float]

@dataclass
class PricingStrategy:
    """Pricing strategy recommendations"""
    strategy_name: str
    description: str
    expected_impact: Dict[str, float]
    implementation_cost: float
    risk_level: str
    timeline_days: int
    success_metrics: List[str]

class TokenStrategyAnalyzer:
    """Main class for token economy strategy analysis and optimization"""

    def __init__(self, config: TokenStrategyConfig):
        """
        Initialize token strategy analyzer

        Args:
            config: TokenStrategyConfig with business parameters
        """
        self.config = config
        self.analysis_results = {}

    def calculate_enrollment_propensity_value(self, learner_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate value of learners based on enrollment propensity

        Args:
            learner_data: DataFrame with learner features and propensity scores

        Returns:
            DataFrame with calculated value metrics
        """
        learner_value = learner_data.copy()

        # Calculate expected value based on propensity
        if 'enrollment_propensity' in learner_value.columns:
            # Expected enrollments per year
            learner_value['expected_annual_enrollments'] = learner_value['enrollment_propensity'] * 3  # Assume 3 potential enrollments/year
            
            # Expected annual revenue per learner
            learner_value['expected_annual_revenue'] = (
                learner_value['expected_annual_enrollments'] * 
                self.config.avg_token_price * 
                self.config.platform_commission_rate
            )
            
            # Lifetime value (3-year horizon)
            years = 3
            discount_rate = self.config.discount_rate_annual
            learner_value['lifetime_value'] = sum([
                learner_value['expected_annual_revenue'] / ((1 + discount_rate) ** year)
                for year in range(1, years + 1)
            ])
            
            # ROI (LTV / CAC)
            learner_value['ltv_cac_ratio'] = learner_value['lifetime_value'] / self.config.learner_acquisition_cost

        logger.info(f"Enrollment propensity value calculated for {len(learner_value)} learners")
        return learner_value

    def optimize_enrollment_propensity_thresholds(self, 
                                                 learner_scores: pd.DataFrame,
                                                 actual_enrollments: pd.Series = None) -> EnrollmentThresholds:
        """
        Optimize propensity score thresholds for learner segmentation

        Args:
            learner_scores: DataFrame with propensity scores
            actual_enrollments: Actual enrollment outcomes (if available)

        Returns:
            Optimized threshold recommendations
        """
        # Default thresholds based on percentiles
        thresholds = {
            'high': learner_scores['enrollment_propensity'].quantile(0.75),  # Top 25%
            'medium': learner_scores['enrollment_propensity'].quantile(0.50),  # Top 50%
            'low': learner_scores['enrollment_propensity'].quantile(0.25)  # Top 75%
        }

        # Calculate expected conversion rates for each segment
        expected_conversion_rates = {}
        expected_revenue_per_segment = {}

        for segment, threshold in thresholds.items():
            if segment == 'high':
                mask = learner_scores['enrollment_propensity'] >= thresholds['high']
            elif segment == 'medium':
                mask = (learner_scores['enrollment_propensity'] >= thresholds['medium']) & \
                       (learner_scores['enrollment_propensity'] < thresholds['high'])
            else:  # low
                mask = learner_scores['enrollment_propensity'] < thresholds['medium']

            segment_data = learner_scores[mask]

            # Estimate conversion rate
            if actual_enrollments is not None and len(segment_data) > 0:
                expected_conversion_rates[segment] = actual_enrollments[mask].mean()
            else:
                # Estimate based on propensity score
                expected_conversion_rates[segment] = segment_data['enrollment_propensity'].mean()

            # Calculate expected revenue
            expected_revenue_per_segment[segment] = (
                len(segment_data) * 
                expected_conversion_rates[segment] * 
                self.config.avg_token_price * 
                self.config.platform_commission_rate
            )

        return EnrollmentThresholds(
            high_propensity_threshold=thresholds['high'],
            medium_propensity_threshold=thresholds['medium'],
            low_propensity_threshold=thresholds['low'],
            expected_conversion_rates=expected_conversion_rates,
            expected_revenue_per_segment=expected_revenue_per_segment
        )

    def generate_pricing_strategies(self, 
                                   market_analysis: Dict[str, Any],
                                   current_performance: Dict[str, float] = None) -> List[PricingStrategy]:
        """
        Generate comprehensive pricing strategy recommendations

        Args:
            market_analysis: Market analysis results (elasticity, competition, etc.)
            current_performance: Current business performance metrics

        Returns:
            List of pricing strategy recommendations
        """
        strategies = []

        # Strategy 1: Dynamic Price Optimization
        strategies.append(PricingStrategy(
            strategy_name="Dynamic Price Optimization",
            description="Implement ML-based dynamic pricing using elasticity models to maximize revenue per course",
            expected_impact={
                'revenue_increase_percentage': 12,
                'enrollment_increase_percentage': 8,
                'profit_margin_improvement': 5
            },
            implementation_cost=50000,
            risk_level="Medium",
            timeline_days=90,
            success_metrics=[
                "Revenue increase > 10%",
                "Price elasticity model R² > 0.85",
                "Average course profit increase > 8%"
            ]
        ))

        # Strategy 2: Propensity-Based Personalization
        strategies.append(PricingStrategy(
            strategy_name="Propensity-Based Personalized Pricing",
            description="Offer personalized discounts and bundles based on learner enrollment propensity",
            expected_impact={
                'conversion_rate_increase_percentage': 25,
                'learner_lifetime_value_increase': 18,
                'churn_reduction_percentage': 15
            },
            implementation_cost=75000,
            risk_level="Medium",
            timeline_days=120,
            success_metrics=[
                "Conversion rate improvement > 20%",
                "LTV increase > 15%",
                "Personalization accuracy > 80%"
            ]
        ))

        # Strategy 3: Category-Level Optimization
        strategies.append(PricingStrategy(
            strategy_name="Category-Level Price Optimization",
            description="Optimize pricing strategies by course category based on demand patterns and competition",
            expected_impact={
                'revenue_per_category_increase': 15,
                'market_share_gain_percentage': 10,
                'category_profit_margin_improvement': 8
            },
            implementation_cost=40000,
            risk_level="Low",
            timeline_days=60,
            success_metrics=[
                "Revenue increase in top 3 categories > 12%",
                "Competitive positioning improved in 80% of categories",
                "Category-level elasticity accuracy > 0.80"
            ]
        ))

        # Strategy 4: Bundle Pricing
        strategies.append(PricingStrategy(
            strategy_name="Strategic Course Bundling",
            description="Create course bundles based on learner pathways and complementary skills",
            expected_impact={
                'average_order_value_increase': 35,
                'enrollment_per_learner_increase': 40,
                'cross_sell_rate_improvement': 30
            },
            implementation_cost=30000,
            risk_level="Low",
            timeline_days=45,
            success_metrics=[
                "Bundle adoption rate > 25%",
                "Average order value increase > 30%",
                "Bundle revenue > 20% of total revenue"
            ]
        ))

        # Strategy 5: Tiered Pricing Model
        strategies.append(PricingStrategy(
            strategy_name="Tiered Subscription Model",
            description="Implement subscription tiers (Basic/Premium/Enterprise) for recurring revenue",
            expected_impact={
                'recurring_revenue_percentage': 40,
                'learner_retention_increase': 35,
                'predictable_revenue_stream': 50
            },
            implementation_cost=100000,
            risk_level="High",
            timeline_days=180,
            success_metrics=[
                "Subscription adoption > 30%",
                "MRR growth > 20% month-over-month",
                "Churn rate < 5% for subscribers"
            ]
        ))

        # Sort strategies by expected ROI
        strategies.sort(key=lambda x: self._calculate_strategy_roi(x), reverse=True)

        logger.info(f"Generated {len(strategies)} pricing strategy recommendations")
        return strategies

    def _calculate_strategy_roi(self, strategy: PricingStrategy) -> float:
        """Calculate expected ROI for a strategy"""
        total_impact = sum(strategy.expected_impact.values())
        if strategy.implementation_cost > 0:
            return (total_impact / strategy.implementation_cost) * 100
        return 0

    def calculate_churn_prevention_value(self, 
                                       learners_at_risk: pd.DataFrame,
                                       intervention_cost_per_learner: float) -> Dict[str, float]:
        """
        Calculate the value of churn prevention interventions

        Args:
            learners_at_risk: DataFrame with learners at risk of churning
            intervention_cost_per_learner: Cost to intervene per learner

        Returns:
            Dictionary with churn prevention analysis
        """
        n_at_risk = len(learners_at_risk)
        
        # Expected churn without intervention
        expected_churn = n_at_risk * self.config.churn_rate_monthly
        
        # Value of preventing churn (LTV per learner)
        avg_ltv_per_learner = self.config.avg_token_price * 3 * self.config.platform_commission_rate  # 3 courses/year * commission
        
        # Total value at risk
        total_value_at_risk = expected_churn * avg_ltv_per_learner
        
        # Cost of intervention
        intervention_cost = n_at_risk * intervention_cost_per_learner
        
        # Assuming 50% effectiveness of intervention
        intervention_effectiveness = 0.5
        value_saved = total_value_at_risk * intervention_effectiveness
        
        # Net benefit
        net_benefit = value_saved - intervention_cost
        roi = (net_benefit / intervention_cost * 100) if intervention_cost > 0 else 0

        churn_prevention = {
            'learners_at_risk': n_at_risk,
            'expected_churn_count': expected_churn,
            'total_value_at_risk': total_value_at_risk,
            'intervention_cost': intervention_cost,
            'value_saved': value_saved,
            'net_benefit': net_benefit,
            'roi_percentage': roi,
            'cost_per_learner_saved': intervention_cost / (expected_churn * intervention_effectiveness) if expected_churn > 0 else 0
        }

        logger.info(f"Churn prevention: {n_at_risk} at risk, ${net_benefit:,.2f} net benefit, {roi:.1f}% ROI")
        return churn_prevention

    def optimize_course_portfolio(self, courses_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Optimize course portfolio for maximum revenue and market coverage

        Args:
            courses_df: DataFrame with course data

        Returns:
            Dictionary with portfolio optimization recommendations
        """
        optimization_results = {}

        # Calculate revenue per course
        courses_df['revenue'] = courses_df['token_price'] * courses_df['total_enrollments'] * self.config.platform_commission_rate
        
        # Identify high-performers (top 20%)
        revenue_threshold_high = courses_df['revenue'].quantile(0.80)
        high_performers = courses_df[courses_df['revenue'] >= revenue_threshold_high]
        
        # Identify low-performers (bottom 20%)
        revenue_threshold_low = courses_df['revenue'].quantile(0.20)
        low_performers = courses_df[courses_df['revenue'] <= revenue_threshold_low]
        
        # Calculate portfolio concentration
        category_concentration = courses_df['category'].value_counts(normalize=True).iloc[0] if 'category' in courses_df.columns else 0
        
        # Revenue concentration (what % of revenue comes from top 20% of courses)
        revenue_concentration = high_performers['revenue'].sum() / courses_df['revenue'].sum() if courses_df['revenue'].sum() > 0 else 0

        optimization_results = {
            'total_courses': len(courses_df),
            'high_performers_count': len(high_performers),
            'low_performers_count': len(low_performers),
            'revenue_concentration': revenue_concentration,
            'category_concentration': category_concentration,
            'recommendations': []
        }

        # Generate recommendations
        if revenue_concentration > 0.6:
            optimization_results['recommendations'].append({
                'priority': 'High',
                'action': 'Diversify revenue sources',
                'details': f'{revenue_concentration*100:.0f}% of revenue from top 20% courses - high risk'
            })

        if category_concentration > 0.5:
            optimization_results['recommendations'].append({
                'priority': 'Medium',
                'action': 'Expand into underrepresented categories',
                'details': f'One category represents {category_concentration*100:.0f}% of portfolio'
            })

        if len(low_performers) > len(courses_df) * 0.3:
            optimization_results['recommendations'].append({
                'priority': 'High',
                'action': 'Optimize or retire low-performing courses',
                'details': f'{len(low_performers)} courses ({len(low_performers)/len(courses_df)*100:.0f}%) are underperforming'
            })

        logger.info(f"Portfolio optimization: {len(optimization_results['recommendations'])} recommendations generated")
        return optimization_results

    def generate_comprehensive_report(self, 
                                    enrollment_thresholds: EnrollmentThresholds,
                                    pricing_strategies: List[PricingStrategy],
                                    portfolio_optimization: Dict[str, Any],
                                    churn_prevention: Dict[str, float] = None) -> str:
        """
        Generate comprehensive strategy report

        Args:
            enrollment_thresholds: Enrollment propensity thresholds
            pricing_strategies: List of pricing strategies
            portfolio_optimization: Portfolio optimization results
            churn_prevention: Churn prevention analysis

        Returns:
            Formatted comprehensive report
        """
        report = []
        report.append("=" * 100)
        report.append("EDTECH TOKEN ECONOMY - COMPREHENSIVE STRATEGY REPORT")
        report.append("=" * 100)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Enrollment Propensity Optimization
        report.append("ENROLLMENT PROPENSITY OPTIMIZATION:")
        report.append("-" * 50)
        report.append(f"High Propensity Threshold: {enrollment_thresholds.high_propensity_threshold:.3f}")
        report.append(f"Medium Propensity Threshold: {enrollment_thresholds.medium_propensity_threshold:.3f}")
        report.append(f"Low Propensity Threshold: {enrollment_thresholds.low_propensity_threshold:.3f}")
        report.append("")
        
        report.append("EXPECTED PERFORMANCE BY SEGMENT:")
        for segment in ['high', 'medium', 'low']:
            conv_rate = enrollment_thresholds.expected_conversion_rates.get(segment, 0)
            revenue = enrollment_thresholds.expected_revenue_per_segment.get(segment, 0)
            report.append(f"{segment.upper()}: {conv_rate:.1%} conversion, ${revenue:,.0f} expected revenue")
        report.append("")

        # Pricing Strategies
        report.append("TOP PRICING STRATEGIES:")
        report.append("-" * 50)
        for i, strategy in enumerate(pricing_strategies[:3], 1):  # Top 3 strategies
            report.append(f"\n{i}. {strategy.strategy_name}")
            report.append(f"   Description: {strategy.description}")
            report.append(f"   Expected Impact: {strategy.expected_impact}")
            report.append(f"   Implementation Cost: ${strategy.implementation_cost:,.0f}")
            report.append(f"   Risk Level: {strategy.risk_level} | Timeline: {strategy.timeline_days} days")
        report.append("")

        # Portfolio Optimization
        report.append("PORTFOLIO OPTIMIZATION:")
        report.append("-" * 50)
        report.append(f"Total Courses: {portfolio_optimization['total_courses']}")
        report.append(f"High Performers: {portfolio_optimization['high_performers_count']}")
        report.append(f"Low Performers: {portfolio_optimization['low_performers_count']}")
        report.append(f"Revenue Concentration: {portfolio_optimization['revenue_concentration']:.1%}")
        report.append("")
        
        if portfolio_optimization['recommendations']:
            report.append("PORTFOLIO RECOMMENDATIONS:")
            for rec in portfolio_optimization['recommendations']:
                report.append(f"  [{rec['priority']}] {rec['action']}")
                report.append(f"       → {rec['details']}")
        report.append("")

        # Churn Prevention
        if churn_prevention:
            report.append("CHURN PREVENTION ANALYSIS:")
            report.append("-" * 50)
            report.append(f"Learners at Risk: {churn_prevention['learners_at_risk']:,}")
            report.append(f"Total Value at Risk: ${churn_prevention['total_value_at_risk']:,.2f}")
            report.append(f"Intervention Cost: ${churn_prevention['intervention_cost']:,.2f}")
            report.append(f"Net Benefit: ${churn_prevention['net_benefit']:,.2f}")
            report.append(f"ROI: {churn_prevention['roi_percentage']:.1f}%")
            report.append("")

        # Implementation Roadmap
        report.append("IMPLEMENTATION ROADMAP:")
        report.append("-" * 50)
        current_date = datetime.now()
        phases = [
            ("Phase 1: Quick Wins", "Category optimization & bundle pricing", 60),
            ("Phase 2: Propensity-Based Strategies", "Personalization & segmentation", 120),
            ("Phase 3: Advanced Pricing", "Dynamic pricing & elasticity optimization", 180),
            ("Phase 4: Subscription Model", "Tiered pricing & recurring revenue", 240)
        ]

        for phase_name, description, duration in phases:
            end_date = current_date + timedelta(days=duration)
            report.append(f"{phase_name} ({duration} days)")
            report.append(f"   Target: {end_date.strftime('%Y-%m-%d')}")
            report.append(f"   Focus: {description}")
        report.append("")

        # Key Success Metrics
        report.append("KEY SUCCESS METRICS TO TRACK:")
        report.append("-" * 50)
        report.append("• Revenue per course (target: +15% increase)")
        report.append("• Enrollment conversion rate (target: +25% improvement)")
        report.append("• Learner LTV (target: +20% increase)")
        report.append("• Churn rate (target: <5% monthly)")
        report.append("• Portfolio profit margin (target: >25%)")
        report.append("")
        report.append("=" * 100)

        return "\n".join(report)


def create_default_strategy_config() -> TokenStrategyConfig:
    """Create default strategy configuration"""
    return TokenStrategyConfig(
        avg_token_price=100.0,
        avg_enrollments_per_course=150,
        platform_commission_rate=0.20,
        learner_acquisition_cost=25.0,
        teacher_payout_percentage=0.80,
        churn_rate_monthly=0.05,
        discount_rate_annual=0.10
    )


# Example usage and testing
if __name__ == "__main__":
    print("Testing Token Economy Strategy Analyzer...")

    # Create configuration
    config = create_default_strategy_config()
    analyzer = TokenStrategyAnalyzer(config)

    # Sample learner data
    np.random.seed(42)
    sample_learners = pd.DataFrame({
        'learner_id': range(1000),
        'enrollment_propensity': np.random.uniform(0, 1, 1000),
        'price_sensitivity': np.random.uniform(0, 1, 1000),
        'total_courses_completed': np.random.randint(0, 10, 1000)
    })

    # Test propensity value calculation
    learner_value = analyzer.calculate_enrollment_propensity_value(sample_learners)
    print(f"✅ Learner value calculated: Avg LTV=${learner_value['lifetime_value'].mean():,.2f}")

    # Test threshold optimization
    thresholds = analyzer.optimize_enrollment_propensity_thresholds(sample_learners)
    print(f"✅ Thresholds optimized: High={thresholds.high_propensity_threshold:.3f}")

    # Test pricing strategies
    market_analysis = {'elasticity': -1.2, 'competition_level': 'medium'}
    strategies = analyzer.generate_pricing_strategies(market_analysis)
    print(f"✅ Generated {len(strategies)} pricing strategies")

    # Test portfolio optimization
    sample_courses = pd.DataFrame({
        'course_id': [f'C{i:05d}' for i in range(100)],
        'token_price': np.random.uniform(50, 200, 100),
        'total_enrollments': np.random.randint(10, 500, 100),
        'category': np.random.choice(['Programming', 'Business', 'Design'], 100)
    })
    
    portfolio_opt = analyzer.optimize_course_portfolio(sample_courses)
    print(f"✅ Portfolio optimization: {len(portfolio_opt['recommendations'])} recommendations")

    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report(thresholds, strategies, portfolio_opt)
    print("\n" + "="*100)
    print("STRATEGY REPORT:")
    print("="*100)
    print(report[:1500] + "...")  # Show first 1500 characters

    print("\nToken Economy Strategy Analyzer test completed successfully!")

