"""
Business Understanding Module - Token Economy Cost Calculations

This module handles cost calculations, ROI analysis, and business metrics
for the EdTech token economy platform.

Features:
- Token revenue calculations
- Teacher earnings analysis
- Platform economics
- ROI for learners
- Cost-benefit analysis

Author: EdTech Token Economy Pipeline
Date: October 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TokenEconomyConfig:
    """Data class for token economy parameters"""
    platform_commission_rate: float  # Platform commission (e.g., 0.20 for 20%)
    token_to_usd_rate: float  # Conversion rate (tokens to USD)
    teacher_payout_percentage: float  # Teacher payout after commission
    operational_cost_per_enrollment: float  # Platform cost per enrollment
    marketing_cost_per_learner: float  # Marketing cost to acquire learner

@dataclass
class TokenMetrics:
    """Data class for calculated token metrics"""
    total_token_revenue: float
    platform_commission: float
    teacher_earnings: float
    learner_lifetime_value: float
    cost_per_enrollment: float
    profit_margin_percentage: float
    roi_percentage: float

class TokenEconomyCostCalculator:
    """Main class for token economy cost calculations and ROI analysis"""

    def __init__(self, config: TokenEconomyConfig):
        """
        Initialize the token economy cost calculator

        Args:
            config: TokenEconomyConfig object with business parameters
        """
        self.config = config
        self.metrics = None

    def calculate_course_revenue(self, token_price: float, enrollments: int) -> Dict[str, float]:
        """
        Calculate revenue breakdown for a course

        Args:
            token_price: Price in tokens
            enrollments: Number of enrollments

        Returns:
            Dictionary with revenue breakdown
        """
        # Total revenue in tokens
        total_token_revenue = token_price * enrollments
        
        # Convert to USD
        total_usd_revenue = total_token_revenue * self.config.token_to_usd_rate
        
        # Platform commission
        platform_commission = total_usd_revenue * self.config.platform_commission_rate
        
        # Teacher earnings (after commission)
        teacher_earnings = total_usd_revenue * self.config.teacher_payout_percentage
        
        # Operational costs
        operational_costs = enrollments * self.config.operational_cost_per_enrollment
        
        # Net platform profit
        net_profit = platform_commission - operational_costs

        revenue_breakdown = {
            'total_token_revenue': total_token_revenue,
            'total_usd_revenue': total_usd_revenue,
            'platform_commission': platform_commission,
            'teacher_earnings': teacher_earnings,
            'operational_costs': operational_costs,
            'net_profit': net_profit,
            'profit_margin': (net_profit / total_usd_revenue * 100) if total_usd_revenue > 0 else 0
        }

        logger.info(f"Course revenue calculated: ${total_usd_revenue:,.2f} revenue, ${net_profit:,.2f} profit")
        return revenue_breakdown

    def calculate_learner_lifetime_value(self, avg_tokens_spent_per_course: float,
                                        courses_per_year: float,
                                        years_active: int,
                                        discount_rate: float = 0.10) -> float:
        """
        Calculate Learner Lifetime Value (LTV)

        Args:
            avg_tokens_spent_per_course: Average tokens spent per course
            courses_per_year: Number of courses taken per year
            years_active: Expected years of platform activity
            discount_rate: Annual discount rate for NPV calculation

        Returns:
            Learner lifetime value in USD
        """
        # Annual spending in tokens
        annual_token_spend = avg_tokens_spent_per_course * courses_per_year
        
        # Convert to USD
        annual_usd_spend = annual_token_spend * self.config.token_to_usd_rate
        
        # Platform revenue from learner (commission)
        annual_platform_revenue = annual_usd_spend * self.config.platform_commission_rate
        
        # Calculate NPV of future cash flows
        ltv = 0
        for year in range(1, years_active + 1):
            discounted_revenue = annual_platform_revenue / ((1 + discount_rate) ** year)
            ltv += discounted_revenue

        logger.info(f"Learner LTV calculated: ${ltv:,.2f}")
        return ltv

    def calculate_teacher_roi(self, time_invested_hours: float,
                            hourly_rate_alternative: float,
                            total_earnings_from_course: float) -> Dict[str, float]:
        """
        Calculate Return on Investment for teachers

        Args:
            time_invested_hours: Total hours invested in creating/maintaining course
            hourly_rate_alternative: Teacher's alternative hourly rate (opportunity cost)
            total_earnings_from_course: Total earnings from the course

        Returns:
            Dictionary with ROI metrics
        """
        # Opportunity cost
        opportunity_cost = time_invested_hours * hourly_rate_alternative
        
        # ROI calculation
        net_return = total_earnings_from_course - opportunity_cost
        roi_percentage = (net_return / opportunity_cost * 100) if opportunity_cost > 0 else 0
        
        # Break-even enrollments needed
        earnings_per_enrollment = total_earnings_from_course / max(1, time_invested_hours)  # Simplified
        
        teacher_roi = {
            'time_invested_hours': time_invested_hours,
            'opportunity_cost': opportunity_cost,
            'total_earnings': total_earnings_from_course,
            'net_return': net_return,
            'roi_percentage': roi_percentage,
            'effective_hourly_rate': total_earnings_from_course / time_invested_hours if time_invested_hours > 0 else 0
        }

        logger.info(f"Teacher ROI calculated: {roi_percentage:.1f}% ROI, ${teacher_roi['effective_hourly_rate']:.2f}/hour")
        return teacher_roi

    def calculate_cost_per_enrollment(self, marketing_spend: float,
                                     total_enrollments: int,
                                     organic_enrollment_rate: float = 0.3) -> Dict[str, float]:
        """
        Calculate cost per enrollment considering organic and paid acquisitions

        Args:
            marketing_spend: Total marketing spend
            total_enrollments: Total number of enrollments
            organic_enrollment_rate: Percentage of enrollments that are organic (no cost)

        Returns:
            Dictionary with enrollment cost metrics
        """
        # Paid enrollments
        paid_enrollments = total_enrollments * (1 - organic_enrollment_rate)
        organic_enrollments = total_enrollments * organic_enrollment_rate
        
        # Cost per paid enrollment
        cost_per_paid_enrollment = marketing_spend / paid_enrollments if paid_enrollments > 0 else 0
        
        # Blended cost per enrollment (including organic)
        blended_cost_per_enrollment = marketing_spend / total_enrollments if total_enrollments > 0 else 0

        enrollment_costs = {
            'total_enrollments': total_enrollments,
            'paid_enrollments': paid_enrollments,
            'organic_enrollments': organic_enrollments,
            'organic_rate': organic_enrollment_rate * 100,
            'marketing_spend': marketing_spend,
            'cost_per_paid_enrollment': cost_per_paid_enrollment,
            'blended_cost_per_enrollment': blended_cost_per_enrollment
        }

        logger.info(f"Cost per enrollment: ${cost_per_paid_enrollment:.2f} (paid), ${blended_cost_per_enrollment:.2f} (blended)")
        return enrollment_costs

    def calculate_price_elasticity_impact(self, current_price: float,
                                       new_price: float,
                                       current_enrollments: float,
                                       price_elasticity: float) -> Dict[str, float]:
        """
        Calculate the impact of price changes on enrollments and revenue

        Args:
            current_price: Current token price
            new_price: Proposed new token price
            current_enrollments: Current number of enrollments
            price_elasticity: Price elasticity coefficient

        Returns:
            Dictionary with impact analysis
        """
        # Calculate percentage price change
        price_change_pct = (new_price - current_price) / current_price
        
        # Calculate expected enrollment change using elasticity
        enrollment_change_pct = price_elasticity * price_change_pct
        new_enrollments = current_enrollments * (1 + enrollment_change_pct)
        
        # Revenue calculations
        current_revenue = current_price * current_enrollments * self.config.token_to_usd_rate
        new_revenue = new_price * new_enrollments * self.config.token_to_usd_rate
        revenue_change = new_revenue - current_revenue
        revenue_change_pct = (revenue_change / current_revenue * 100) if current_revenue > 0 else 0
        
        # Determine if demand is elastic or inelastic
        is_elastic = abs(price_elasticity) > 1

        impact = {
            'current_price': current_price,
            'new_price': new_price,
            'price_change_percentage': price_change_pct * 100,
            'current_enrollments': current_enrollments,
            'new_enrollments': new_enrollments,
            'enrollment_change': new_enrollments - current_enrollments,
            'enrollment_change_percentage': enrollment_change_pct * 100,
            'current_revenue_usd': current_revenue,
            'new_revenue_usd': new_revenue,
            'revenue_change_usd': revenue_change,
            'revenue_change_percentage': revenue_change_pct,
            'is_elastic': is_elastic,
            'elasticity_coefficient': price_elasticity,
            'recommendation': self._get_pricing_recommendation(price_elasticity, price_change_pct, revenue_change)
        }

        logger.info(f"Price impact: {impact['price_change_percentage']:+.1f}% price → {impact['revenue_change_percentage']:+.1f}% revenue")
        return impact

    def _get_pricing_recommendation(self, elasticity: float, price_change: float, revenue_change: float) -> str:
        """Generate pricing recommendation based on elasticity and revenue impact"""
        if revenue_change > 0:
            return "✅ Price change increases revenue - recommended"
        elif abs(elasticity) > 1.5 and price_change > 0:
            return "⚠️ Demand is highly elastic - price increase may hurt volume significantly"
        elif abs(elasticity) < 0.5 and price_change < 0:
            return "⚠️ Demand is inelastic - price decrease may not boost enrollments enough"
        else:
            return "❌ Price change decreases revenue - not recommended"

    def calculate_portfolio_metrics(self, courses_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive portfolio-level metrics

        Args:
            courses_df: DataFrame with course data (token_price, total_enrollments, category, etc.)

        Returns:
            Dictionary with portfolio metrics
        """
        # Total metrics
        total_courses = len(courses_df)
        total_enrollments = courses_df['total_enrollments'].sum()
        
        # Revenue calculations
        courses_df['token_revenue'] = courses_df['token_price'] * courses_df['total_enrollments']
        total_token_revenue = courses_df['token_revenue'].sum()
        total_usd_revenue = total_token_revenue * self.config.token_to_usd_rate
        
        # Platform commission and costs
        platform_commission = total_usd_revenue * self.config.platform_commission_rate
        total_operational_costs = total_enrollments * self.config.operational_cost_per_enrollment
        net_profit = platform_commission - total_operational_costs
        
        # Teacher earnings
        total_teacher_earnings = total_usd_revenue * self.config.teacher_payout_percentage
        
        # Averages
        avg_price = courses_df['token_price'].mean()
        avg_enrollments = courses_df['total_enrollments'].mean()
        avg_revenue_per_course = total_usd_revenue / total_courses if total_courses > 0 else 0
        
        # Category breakdown
        category_revenue = {}
        if 'category' in courses_df.columns:
            category_revenue = courses_df.groupby('category').agg({
                'token_revenue': 'sum',
                'total_enrollments': 'sum',
                'token_price': 'mean'
            }).to_dict('index')

        portfolio_metrics = {
            'total_courses': total_courses,
            'total_enrollments': int(total_enrollments),
            'total_token_revenue': total_token_revenue,
            'total_usd_revenue': total_usd_revenue,
            'platform_commission': platform_commission,
            'total_teacher_earnings': total_teacher_earnings,
            'operational_costs': total_operational_costs,
            'net_profit': net_profit,
            'profit_margin_percentage': (net_profit / total_usd_revenue * 100) if total_usd_revenue > 0 else 0,
            'avg_price_tokens': avg_price,
            'avg_enrollments_per_course': avg_enrollments,
            'avg_revenue_per_course_usd': avg_revenue_per_course,
            'category_breakdown': category_revenue
        }

        logger.info(f"Portfolio metrics: {total_courses} courses, ${total_usd_revenue:,.2f} revenue, {portfolio_metrics['profit_margin_percentage']:.1f}% margin")
        return portfolio_metrics

    def calculate_break_even_analysis(self, fixed_costs_monthly: float,
                                     variable_cost_per_enrollment: float,
                                     avg_token_price: float,
                                     avg_enrollments_per_course: int,
                                     new_courses_per_month: int) -> Dict[str, float]:
        """
        Calculate break-even analysis for the platform

        Args:
            fixed_costs_monthly: Monthly fixed costs (servers, staff, etc.)
            variable_cost_per_enrollment: Variable cost per enrollment
            avg_token_price: Average token price per course
            avg_enrollments_per_course: Average enrollments per course
            new_courses_per_month: Number of new courses added monthly

        Returns:
            Dictionary with break-even metrics
        """
        # Revenue per course
        revenue_per_course = avg_token_price * avg_enrollments_per_course * self.config.token_to_usd_rate * self.config.platform_commission_rate
        
        # Variable costs per course
        variable_costs_per_course = avg_enrollments_per_course * variable_cost_per_enrollment
        
        # Contribution margin per course
        contribution_margin = revenue_per_course - variable_costs_per_course
        
        # Break-even courses
        if contribution_margin > 0:
            break_even_courses = fixed_costs_monthly / contribution_margin
            months_to_break_even = break_even_courses / new_courses_per_month if new_courses_per_month > 0 else float('inf')
        else:
            break_even_courses = float('inf')
            months_to_break_even = float('inf')

        break_even = {
            'fixed_costs_monthly': fixed_costs_monthly,
            'variable_cost_per_enrollment': variable_cost_per_enrollment,
            'revenue_per_course': revenue_per_course,
            'variable_costs_per_course': variable_costs_per_course,
            'contribution_margin_per_course': contribution_margin,
            'break_even_courses': break_even_courses,
            'current_new_courses_per_month': new_courses_per_month,
            'months_to_break_even': months_to_break_even
        }

        logger.info(f"Break-even: {break_even_courses:.0f} courses needed ({months_to_break_even:.1f} months at current rate)")
        return break_even

    def generate_business_report(self, portfolio_metrics: Dict[str, Any],
                               ltv: float = None,
                               teacher_roi_avg: Dict[str, float] = None) -> str:
        """
        Generate a formatted business report

        Args:
            portfolio_metrics: Portfolio-level metrics
            ltv: Learner lifetime value
            teacher_roi_avg: Average teacher ROI metrics

        Returns:
            Formatted business report string
        """
        report = []
        report.append("=" * 80)
        report.append("EDTECH TOKEN ECONOMY - BUSINESS ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Portfolio Overview
        report.append("PORTFOLIO OVERVIEW:")
        report.append("-" * 50)
        report.append(f"Total Courses: {portfolio_metrics['total_courses']:,}")
        report.append(f"Total Enrollments: {portfolio_metrics['total_enrollments']:,}")
        report.append(f"Total Token Revenue: {portfolio_metrics['total_token_revenue']:,.0f} tokens")
        report.append(f"Total USD Revenue: ${portfolio_metrics['total_usd_revenue']:,.2f}")
        report.append("")

        # Revenue Breakdown
        report.append("REVENUE BREAKDOWN:")
        report.append("-" * 50)
        report.append(f"Platform Commission: ${portfolio_metrics['platform_commission']:,.2f} ({self.config.platform_commission_rate*100:.0f}%)")
        report.append(f"Teacher Earnings: ${portfolio_metrics['total_teacher_earnings']:,.2f} ({self.config.teacher_payout_percentage*100:.0f}%)")
        report.append(f"Operational Costs: ${portfolio_metrics['operational_costs']:,.2f}")
        report.append(f"Net Profit: ${portfolio_metrics['net_profit']:,.2f}")
        report.append(f"Profit Margin: {portfolio_metrics['profit_margin_percentage']:.2f}%")
        report.append("")

        # Per-Course Averages
        report.append("PER-COURSE AVERAGES:")
        report.append("-" * 50)
        report.append(f"Average Token Price: {portfolio_metrics['avg_price_tokens']:.2f} tokens")
        report.append(f"Average Enrollments: {portfolio_metrics['avg_enrollments_per_course']:.2f}")
        report.append(f"Average Revenue per Course: ${portfolio_metrics['avg_revenue_per_course_usd']:,.2f}")
        report.append("")

        # Learner LTV
        if ltv:
            report.append("LEARNER METRICS:")
            report.append("-" * 50)
            report.append(f"Learner Lifetime Value: ${ltv:,.2f}")
            report.append(f"Marketing Cost per Learner: ${self.config.marketing_cost_per_learner:.2f}")
            report.append(f"LTV to CAC Ratio: {ltv / self.config.marketing_cost_per_learner:.2f}x")
            report.append("")

        # Teacher ROI
        if teacher_roi_avg:
            report.append("TEACHER METRICS:")
            report.append("-" * 50)
            report.append(f"Average ROI: {teacher_roi_avg['roi_percentage']:.1f}%")
            report.append(f"Average Effective Hourly Rate: ${teacher_roi_avg['effective_hourly_rate']:.2f}")
            report.append(f"Average Earnings per Course: ${teacher_roi_avg['total_earnings']:,.2f}")
            report.append("")

        # Category Breakdown
        if portfolio_metrics.get('category_breakdown'):
            report.append("CATEGORY PERFORMANCE:")
            report.append("-" * 50)
            for category, metrics in list(portfolio_metrics['category_breakdown'].items())[:5]:
                token_rev = metrics['token_revenue']
                enrollments = metrics['total_enrollments']
                avg_price = metrics['token_price']
                report.append(f"{category}:")
                report.append(f"  Revenue: {token_rev:,.0f} tokens | Enrollments: {enrollments:,.0f} | Avg Price: {avg_price:.2f} tokens")
            report.append("")

        # Recommendations
        report.append("STRATEGIC RECOMMENDATIONS:")
        report.append("-" * 50)
        
        # Profit margin recommendations
        if portfolio_metrics['profit_margin_percentage'] < 20:
            report.append("⚠️ Low profit margin - consider:")
            report.append("   • Optimizing operational costs")
            report.append("   • Adjusting commission structure")
            report.append("   • Implementing dynamic pricing")
        else:
            report.append("✅ Healthy profit margins")
        
        # LTV:CAC recommendations
        if ltv and ltv / self.config.marketing_cost_per_learner < 3:
            report.append("⚠️ Low LTV:CAC ratio - consider:")
            report.append("   • Improving learner retention")
            report.append("   • Reducing marketing costs")
            report.append("   • Increasing course engagement")
        elif ltv:
            report.append("✅ Strong LTV:CAC ratio - sustainable growth")
        
        report.append("")
        report.append("KEY ACTIONS:")
        report.append("• Monitor price elasticity for optimization opportunities")
        report.append("• Focus on high-performing categories")
        report.append("• Implement learner retention programs")
        report.append("• Support teacher success with quality metrics")
        report.append("")
        report.append("=" * 80)

        return "\n".join(report)


def create_default_token_economy_config() -> TokenEconomyConfig:
    """Create a default token economy configuration"""
    return TokenEconomyConfig(
        platform_commission_rate=0.20,  # 20% platform commission
        token_to_usd_rate=1.00,  # 1 token = $1 USD
        teacher_payout_percentage=0.80,  # 80% goes to teacher
        operational_cost_per_enrollment=2.50,  # $2.50 operational cost per enrollment
        marketing_cost_per_learner=15.00  # $15 to acquire a learner
    )


# Example usage and testing
if __name__ == "__main__":
    print("Testing Token Economy Cost Calculator...")

    # Create configuration
    config = create_default_token_economy_config()
    calculator = TokenEconomyCostCalculator(config)

    # Test course revenue calculation
    course_revenue = calculator.calculate_course_revenue(token_price=100, enrollments=200)
    print(f"✅ Course Revenue: ${course_revenue['total_usd_revenue']:,.2f}")
    print(f"   Platform Profit: ${course_revenue['net_profit']:,.2f}")
    print(f"   Teacher Earnings: ${course_revenue['teacher_earnings']:,.2f}")

    # Test learner LTV
    ltv = calculator.calculate_learner_lifetime_value(
        avg_tokens_spent_per_course=100,
        courses_per_year=3,
        years_active=3
    )
    print(f"✅ Learner LTV: ${ltv:,.2f}")

    # Test teacher ROI
    teacher_roi = calculator.calculate_teacher_roi(
        time_invested_hours=40,
        hourly_rate_alternative=50,
        total_earnings_from_course=3000
    )
    print(f"✅ Teacher ROI: {teacher_roi['roi_percentage']:.1f}%")

    # Test price elasticity impact
    price_impact = calculator.calculate_price_elasticity_impact(
        current_price=100,
        new_price=90,
        current_enrollments=200,
        price_elasticity=-1.2
    )
    print(f"✅ Price Impact: {price_impact['revenue_change_percentage']:+.1f}% revenue change")
    print(f"   {price_impact['recommendation']}")

    # Test portfolio metrics with sample data
    sample_courses = pd.DataFrame({
        'course_id': [f'C{i:05d}' for i in range(10)],
        'token_price': np.random.uniform(50, 150, 10),
        'total_enrollments': np.random.randint(50, 300, 10),
        'category': np.random.choice(['Programming', 'Business', 'Design'], 10)
    })

    portfolio_metrics = calculator.calculate_portfolio_metrics(sample_courses)
    print(f"✅ Portfolio Metrics: ${portfolio_metrics['total_usd_revenue']:,.2f} revenue")

    # Generate business report
    report = calculator.generate_business_report(portfolio_metrics, ltv, teacher_roi)
    print("\n" + "="*80)
    print("BUSINESS REPORT:")
    print("="*80)
    print(report)

    print("\nToken Economy Cost Calculator test completed successfully!")

