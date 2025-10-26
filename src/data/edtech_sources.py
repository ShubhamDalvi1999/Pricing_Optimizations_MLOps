"""
EdTech Token Economy Data Generation

This module creates demo data for the EdTech token economy platform with:
- Learners (token spenders)
- Teachers (token earners)
- Courses (products)
- Enrollments (transactions)
- Token Transactions
- Platform Metrics

Author: EdTech Token Economy Team
Date: October 2025
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import json
from typing import Dict, List, Tuple

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)


class EdTechTokenEconomyGenerator:
    """Generate demo EdTech token economy database with realistic data"""

    def __init__(self, db_path="edtech_token_economy.db"):
        self.db_path = db_path
        self.conn = None

    def connect(self):
        """Create database connection"""
        self.conn = sqlite3.connect(self.db_path)
        print(f"Connected to database: {self.db_path}")

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("Database connection closed")

    def create_tables(self):
        """Create the EdTech token economy tables"""
        
        # Learners Table
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS learners (
                learner_id TEXT PRIMARY KEY,
                account_created_date TEXT NOT NULL,
                account_age_days INTEGER,
                user_type TEXT,
                education_level TEXT,
                profession TEXT,
                age_group TEXT,
                location TEXT,
                total_courses_enrolled INTEGER DEFAULT 0,
                total_courses_completed INTEGER DEFAULT 0,
                completion_rate REAL DEFAULT 0.0,
                avg_course_rating_given REAL,
                total_learning_hours REAL DEFAULT 0.0,
                avg_daily_active_minutes REAL DEFAULT 0.0,
                last_activity_date TEXT,
                learning_streak_days INTEGER DEFAULT 0,
                total_tokens_spent REAL DEFAULT 0.0,
                avg_tokens_per_course REAL DEFAULT 0.0,
                token_balance REAL DEFAULT 100.0,
                tokens_purchased REAL DEFAULT 0.0,
                price_sensitivity_score REAL,
                preferred_price_range TEXT,
                favorite_categories TEXT,
                skill_level TEXT,
                preferred_course_length TEXT,
                preferred_learning_time TEXT,
                forum_participation_score REAL DEFAULT 0.0,
                assignment_submission_rate REAL DEFAULT 0.0,
                peer_interaction_score REAL DEFAULT 0.0,
                certificate_collection_rate REAL DEFAULT 0.0,
                churn_risk_score REAL DEFAULT 0.5,
                upsell_propensity REAL DEFAULT 0.5,
                course_completion_propensity REAL DEFAULT 0.5,
                created_date TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Teachers Table
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS teachers (
                teacher_id TEXT PRIMARY KEY,
                account_created_date TEXT NOT NULL,
                teaching_tenure_days INTEGER,
                verified_credentials TEXT,
                professional_experience_years INTEGER,
                specialization TEXT,
                language_proficiency TEXT,
                total_courses_created INTEGER DEFAULT 0,
                total_students_taught INTEGER DEFAULT 0,
                avg_course_rating REAL,
                avg_course_completion_rate REAL,
                total_teaching_hours REAL DEFAULT 0.0,
                response_time_hours REAL,
                student_satisfaction_score REAL,
                total_tokens_earned REAL DEFAULT 0.0,
                avg_tokens_per_student REAL DEFAULT 0.0,
                token_earning_velocity REAL DEFAULT 1.0,
                pricing_strategy TEXT,
                discount_frequency REAL DEFAULT 0.0,
                content_update_frequency REAL DEFAULT 0.0,
                video_quality_score REAL DEFAULT 0.0,
                exercise_quality_score REAL DEFAULT 0.0,
                curriculum_depth_score REAL DEFAULT 0.0,
                forum_response_rate REAL DEFAULT 0.0,
                live_session_frequency REAL DEFAULT 0.0,
                personalized_feedback_rate REAL DEFAULT 0.0,
                community_building_score REAL DEFAULT 0.0,
                competitive_pricing_index REAL DEFAULT 1.0,
                niche_dominance_score REAL DEFAULT 0.0,
                brand_recognition REAL DEFAULT 0.0,
                teacher_quality_score REAL DEFAULT 0.0,
                quality_tier TEXT,
                created_date TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Courses Table
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS courses (
                course_id TEXT PRIMARY KEY,
                teacher_id TEXT,
                course_title TEXT NOT NULL,
                category TEXT,
                subcategory TEXT,
                difficulty_level TEXT,
                duration_hours REAL,
                video_count INTEGER,
                assignment_count INTEGER,
                certificate_offered INTEGER,
                token_price REAL,
                original_token_price REAL,
                current_discount_pct REAL DEFAULT 0.0,
                price_elasticity_coefficient REAL,
                token_per_hour_ratio REAL,
                teacher_earnings_per_enrollment REAL,
                platform_fee_pct REAL DEFAULT 30.0,
                total_enrollments INTEGER DEFAULT 0,
                active_learners INTEGER DEFAULT 0,
                completion_rate REAL DEFAULT 0.0,
                enrollment_velocity REAL DEFAULT 0.0,
                waitlist_count INTEGER DEFAULT 0,
                competitive_courses_count INTEGER,
                avg_rating REAL,
                review_count INTEGER DEFAULT 0,
                content_quality_score REAL DEFAULT 0.0,
                updated_within_months INTEGER,
                plagiarism_check_score REAL DEFAULT 100.0,
                avg_completion_time_days REAL,
                avg_time_per_lesson_minutes REAL,
                forum_activity_score REAL DEFAULT 0.0,
                return_student_rate REAL DEFAULT 0.0,
                preview_to_enrollment_rate REAL DEFAULT 0.0,
                add_to_cart_rate REAL DEFAULT 0.0,
                cart_abandonment_rate REAL DEFAULT 0.0,
                refund_rate REAL DEFAULT 0.0,
                created_date TEXT,
                last_updated_date TEXT,
                FOREIGN KEY (teacher_id) REFERENCES teachers(teacher_id)
            )
        ''')

        # Enrollments Table
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS enrollments (
                enrollment_id TEXT PRIMARY KEY,
                learner_id TEXT,
                course_id TEXT,
                teacher_id TEXT,
                enrollment_date TEXT NOT NULL,
                transaction_type TEXT,
                tokens_spent REAL,
                tokens_to_teacher REAL,
                tokens_to_platform REAL,
                payment_method TEXT,
                device_type TEXT,
                referral_source TEXT,
                time_of_day INTEGER,
                day_of_week INTEGER,
                season TEXT,
                browse_to_enroll_minutes REAL,
                courses_viewed_before_purchase INTEGER,
                price_comparison_count INTEGER,
                session_duration_minutes REAL,
                pages_viewed INTEGER,
                promo_code_used INTEGER,
                discount_received_pct REAL DEFAULT 0.0,
                is_flash_sale INTEGER,
                is_bundle_purchase INTEGER,
                loyalty_points_redeemed INTEGER DEFAULT 0,
                completed INTEGER DEFAULT 0,
                completion_date TEXT,
                time_to_complete_days REAL,
                rating_given REAL,
                review_text TEXT,
                refunded INTEGER DEFAULT 0,
                refund_date TEXT,
                FOREIGN KEY (learner_id) REFERENCES learners(learner_id),
                FOREIGN KEY (course_id) REFERENCES courses(course_id),
                FOREIGN KEY (teacher_id) REFERENCES teachers(teacher_id)
            )
        ''')

        # Token Transactions Table
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS token_transactions (
                transaction_id TEXT PRIMARY KEY,
                user_id TEXT,
                user_type TEXT,
                transaction_date TEXT NOT NULL,
                transaction_type TEXT,
                tokens_amount REAL,
                token_balance_before REAL,
                token_balance_after REAL,
                related_enrollment_id TEXT,
                description TEXT
            )
        ''')

        # Platform Metrics Table (Time Series)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS platform_metrics (
                metric_date TEXT PRIMARY KEY,
                total_active_learners INTEGER,
                total_active_teachers INTEGER,
                courses_available INTEGER,
                avg_category_price REAL,
                total_tokens_in_circulation REAL,
                token_velocity REAL,
                token_burn_rate REAL,
                token_mint_rate REAL,
                token_liquidity_ratio REAL,
                platform_growth_rate REAL,
                viral_coefficient REAL,
                daily_active_users INTEGER,
                monthly_active_users INTEGER,
                avg_price_elasticity REAL,
                competitor_pricing_index REAL,
                seasonal_demand_index REAL
            )
        ''')

        print("Tables created successfully!")

    def generate_learners(self, n_learners=10000) -> pd.DataFrame:
        """Generate learners data"""
        print(f"Generating {n_learners} learners...")

        user_types = ["student", "professional", "hobbyist", "career_changer"]
        education_levels = ["high_school", "bachelors", "masters", "phd", "bootcamp"]
        professions = ["software_engineer", "data_analyst", "designer", "marketer", 
                      "student", "teacher", "consultant", "entrepreneur", "other"]
        age_groups = ["18-24", "25-34", "35-44", "45-54", "55+"]
        locations = ["US-West", "US-East", "EU-West", "EU-Central", "Asia-Pacific", 
                    "Latin-America", "Middle-East", "Africa"]
        price_ranges = ["low", "medium", "high"]
        skill_levels = ["beginner", "intermediate", "advanced", "expert"]
        course_lengths = ["short", "medium", "long"]
        learning_times = ["morning", "afternoon", "evening", "weekend", "flexible"]

        data = []
        for i in range(n_learners):
            account_date = datetime.now() - timedelta(days=random.randint(0, 1095))
            account_age = (datetime.now() - account_date).days
            
            # Last activity (more recent for active users)
            if random.random() > 0.2:  # 80% active
                last_activity = datetime.now() - timedelta(days=random.randint(0, 30))
            else:
                last_activity = datetime.now() - timedelta(days=random.randint(31, 180))

            # Generate realistic metrics
            total_enrolled = random.randint(0, 50)
            total_completed = int(total_enrolled * random.uniform(0.3, 0.9))
            completion_rate = total_completed / total_enrolled if total_enrolled > 0 else 0
            
            # Token economics
            avg_token_per_course = random.uniform(50, 200)
            total_spent = total_enrolled * avg_token_per_course * random.uniform(0.8, 1.2)
            tokens_purchased = total_spent * random.uniform(1.0, 1.3)
            token_balance = tokens_purchased - total_spent + random.uniform(0, 200)

            learner = {
                'learner_id': f"L{i+1:05d}",
                'account_created_date': account_date.date().isoformat(),
                'account_age_days': account_age,
                'user_type': random.choice(user_types),
                'education_level': random.choice(education_levels),
                'profession': random.choice(professions),
                'age_group': random.choice(age_groups),
                'location': random.choice(locations),
                'total_courses_enrolled': total_enrolled,
                'total_courses_completed': total_completed,
                'completion_rate': round(completion_rate, 2),
                'avg_course_rating_given': round(random.uniform(3.5, 5.0), 1),
                'total_learning_hours': round(total_completed * random.uniform(10, 40), 1),
                'avg_daily_active_minutes': round(random.uniform(5, 120), 1),
                'last_activity_date': last_activity.date().isoformat(),
                'learning_streak_days': random.randint(0, 60),
                'total_tokens_spent': round(total_spent, 2),
                'avg_tokens_per_course': round(avg_token_per_course, 2),
                'token_balance': round(token_balance, 2),
                'tokens_purchased': round(tokens_purchased, 2),
                'price_sensitivity_score': round(random.uniform(0.2, 0.9), 2),
                'preferred_price_range': random.choice(price_ranges),
                'favorite_categories': json.dumps(random.sample([
                    "Programming", "Data Science", "Design", "Business", "Marketing", 
                    "Personal Development", "Language", "Music"
                ], k=random.randint(1, 3))),
                'skill_level': random.choice(skill_levels),
                'preferred_course_length': random.choice(course_lengths),
                'preferred_learning_time': random.choice(learning_times),
                'forum_participation_score': round(random.uniform(0.0, 1.0), 2),
                'assignment_submission_rate': round(random.uniform(0.5, 1.0), 2),
                'peer_interaction_score': round(random.uniform(0.0, 1.0), 2),
                'certificate_collection_rate': round(random.uniform(0.4, 0.95), 2),
                'churn_risk_score': round(random.uniform(0.1, 0.9), 2),
                'upsell_propensity': round(random.uniform(0.2, 0.8), 2),
                'course_completion_propensity': round(random.uniform(0.3, 0.9), 2)
            }
            data.append(learner)

        df = pd.DataFrame(data)
        df.to_sql('learners', self.conn, if_exists='replace', index=False)
        print(f"Generated {len(df)} learners")
        return df

    def generate_teachers(self, n_teachers=500) -> pd.DataFrame:
        """Generate teachers data"""
        print(f"Generating {n_teachers} teachers...")

        specializations = [
            "Web Development", "Mobile Development", "Data Science", "Machine Learning",
            "Cloud Computing", "Cybersecurity", "UI/UX Design", "Graphic Design",
            "Digital Marketing", "Business Analytics", "Project Management", "Leadership",
            "Creative Writing", "Music Production", "Photography", "Language Teaching"
        ]
        
        languages = ["English", "Spanish", "French", "German", "Chinese", "Japanese", 
                    "Portuguese", "Arabic", "Hindi", "Russian"]
        
        pricing_strategies = ["low", "medium", "premium"]
        quality_tiers = ["Bronze", "Silver", "Gold", "Platinum"]

        data = []
        for i in range(n_teachers):
            account_date = datetime.now() - timedelta(days=random.randint(30, 1095))
            tenure_days = (datetime.now() - account_date).days
            
            # Generate realistic metrics
            total_courses = random.randint(1, 20)
            total_students = int(total_courses * random.uniform(50, 500))
            avg_rating = random.uniform(3.5, 5.0)
            avg_completion = random.uniform(0.5, 0.9)
            
            # Token economics
            avg_tokens_per_student = random.uniform(50, 150) * 0.7  # 70% teacher share
            total_earned = total_students * avg_tokens_per_student
            
            # Quality scores
            content_quality = random.uniform(60, 100)
            video_quality = random.uniform(60, 100)
            exercise_quality = random.uniform(50, 100)
            curriculum_depth = random.uniform(60, 100)
            
            # Calculate teacher quality score
            quality_score = (
                (avg_rating / 5 * 30) +
                (avg_completion * 25) +
                (min(total_students / 1000, 1) * 20) +
                (min(24 / random.uniform(6, 48), 1) * 15) +
                (random.uniform(0, 1) * 10)
            )
            
            # Assign quality tier
            if quality_score >= 86:
                tier = "Platinum"
            elif quality_score >= 71:
                tier = "Gold"
            elif quality_score >= 51:
                tier = "Silver"
            else:
                tier = "Bronze"

            teacher = {
                'teacher_id': f"T{i+1:05d}",
                'account_created_date': account_date.date().isoformat(),
                'teaching_tenure_days': tenure_days,
                'verified_credentials': json.dumps(random.sample([
                    "PhD", "Masters", "Professional Certificate", "Industry Expert", 
                    "Published Author", "Conference Speaker"
                ], k=random.randint(1, 3))),
                'professional_experience_years': random.randint(2, 25),
                'specialization': random.choice(specializations),
                'language_proficiency': json.dumps(random.sample(languages, k=random.randint(1, 3))),
                'total_courses_created': total_courses,
                'total_students_taught': total_students,
                'avg_course_rating': round(avg_rating, 1),
                'avg_course_completion_rate': round(avg_completion, 2),
                'total_teaching_hours': round(total_courses * random.uniform(10, 50), 1),
                'response_time_hours': round(random.uniform(4, 48), 1),
                'student_satisfaction_score': round(random.uniform(7.0, 10.0), 1),
                'total_tokens_earned': round(total_earned, 2),
                'avg_tokens_per_student': round(avg_tokens_per_student, 2),
                'token_earning_velocity': round(random.uniform(0.8, 1.5), 2),
                'pricing_strategy': random.choice(pricing_strategies),
                'discount_frequency': round(random.uniform(0.0, 0.5), 2),
                'content_update_frequency': round(random.uniform(0.5, 4.0), 1),
                'video_quality_score': round(video_quality, 1),
                'exercise_quality_score': round(exercise_quality, 1),
                'curriculum_depth_score': round(curriculum_depth, 1),
                'forum_response_rate': round(random.uniform(0.5, 1.0), 2),
                'live_session_frequency': round(random.uniform(0, 4), 1),
                'personalized_feedback_rate': round(random.uniform(0.3, 1.0), 2),
                'community_building_score': round(random.uniform(0.4, 1.0), 2),
                'competitive_pricing_index': round(random.uniform(0.8, 1.3), 2),
                'niche_dominance_score': round(random.uniform(0.0, 1.0), 2),
                'brand_recognition': round(random.uniform(0.0, 1.0), 2),
                'teacher_quality_score': round(quality_score, 1),
                'quality_tier': tier
            }
            data.append(teacher)

        df = pd.DataFrame(data)
        df.to_sql('teachers', self.conn, if_exists='replace', index=False)
        print(f"Generated {len(df)} teachers")
        return df

    def generate_courses(self, n_courses=2000, teachers_df=None) -> pd.DataFrame:
        """Generate courses data"""
        print(f"Generating {n_courses} courses...")

        if teachers_df is None:
            teachers_df = pd.read_sql("SELECT * FROM teachers", self.conn)
        
        categories = {
            "Programming": ["Python", "JavaScript", "Java", "C++", "Web Development", "Mobile Apps"],
            "Data Science": ["Machine Learning", "Statistics", "Data Visualization", "Big Data", "AI"],
            "Design": ["UI/UX", "Graphic Design", "3D Modeling", "Animation", "Product Design"],
            "Business": ["Entrepreneurship", "Marketing", "Finance", "Management", "Strategy"],
            "Personal Development": ["Leadership", "Communication", "Productivity", "Career", "Wellness"]
        }
        
        difficulty_levels = ["beginner", "intermediate", "advanced"]

        data = []
        for i in range(n_courses):
            # Select random teacher
            teacher = teachers_df.sample(1).iloc[0]
            
            # Select category and subcategory
            category = random.choice(list(categories.keys()))
            subcategory = random.choice(categories[category])
            
            # Course attributes
            difficulty = random.choice(difficulty_levels)
            duration = random.uniform(5, 60)
            video_count = int(duration * random.uniform(1.5, 3))
            assignment_count = int(video_count * random.uniform(0.2, 0.5))
            
            # More realistic pricing based on market factors
            base_price = 25  # Lower base price for better accessibility
            
            # Duration factor (more realistic scaling)
            duration_factor = 1 + (duration - 10) / 50  # Linear scaling from 10-60 hours
            
            # Difficulty factor (advanced courses cost more)
            difficulty_factor = {"beginner": 0.7, "intermediate": 1.0, "advanced": 1.4}[difficulty]
            
            # Teacher quality factor (premium teachers charge more)
            quality_factor = 0.5 + (teacher['teacher_quality_score'] / 100) * 1.5
            
            # Category premium (some categories are more valuable)
            category_premium = {
                "Programming": 1.2,
                "Data Science": 1.3, 
                "Design": 1.1,
                "Business": 1.0,
                "Personal Development": 0.9
            }[category]
            
            # Calculate realistic price
            original_price = base_price * duration_factor * difficulty_factor * quality_factor * category_premium
            
            # Apply realistic discounting (seasonal sales, launch discounts)
            discount_probability = 0.4  # 40% of courses have discounts
            if random.random() < discount_probability:
                # Different types of discounts
                discount_types = [
                    random.uniform(10, 20),  # Regular sale
                    random.uniform(25, 40),  # Black Friday style
                    random.uniform(15, 25),  # Launch discount
                ]
                current_discount = random.choice(discount_types)
            else:
                current_discount = 0
                
            token_price = max(10, original_price * (1 - current_discount / 100))  # Minimum 10 tokensI 
            
            # Platform split
            teacher_earnings = token_price * 0.7
            platform_fee = 30.0
            
            # REALISTIC enrollment prediction based on market dynamics
            # This will be calculated AFTER enrollments are generated to ensure consistency
            
            # For now, set a placeholder that will be updated later
            total_enrollments = 0  # Will be calculated after enrollments are generated
            
            # Quality metrics
            avg_rating = teacher['avg_course_rating'] * random.uniform(0.9, 1.1)
            avg_rating = min(5.0, max(1.0, avg_rating))
            
            completion_rate = teacher['avg_course_completion_rate'] * random.uniform(0.8, 1.2)
            completion_rate = min(1.0, max(0.0, completion_rate))
            
            created_date = teacher['account_created_date']
            last_updated = (datetime.fromisoformat(created_date) + 
                          timedelta(days=random.randint(0, 365))).date().isoformat()

            course = {
                'course_id': f"C{i+1:05d}",
                'teacher_id': teacher['teacher_id'],
                'course_title': f"{subcategory} {difficulty.title()} Course {i+1}",
                'category': category,
                'subcategory': subcategory,
                'difficulty_level': difficulty,
                'duration_hours': round(duration, 1),
                'video_count': video_count,
                'assignment_count': assignment_count,
                'certificate_offered': 1 if video_count >= 5 and assignment_count >= 3 else 0,
                'token_price': round(token_price, 2),
                'original_token_price': round(original_price, 2),
                'current_discount_pct': round(current_discount, 1),
                'price_elasticity_coefficient': round(random.uniform(-2.0, -0.5), 2),
                'token_per_hour_ratio': round(token_price / duration, 2),
                'teacher_earnings_per_enrollment': round(teacher_earnings, 2),
                'platform_fee_pct': platform_fee,
                'total_enrollments': total_enrollments,
                'active_learners': int(total_enrollments * random.uniform(0.2, 0.4)),
                'completion_rate': round(completion_rate, 2),
                'enrollment_velocity': round(random.uniform(0.5, 2.0), 2),
                'waitlist_count': random.randint(0, 50) if random.random() < 0.1 else 0,
                'competitive_courses_count': random.randint(5, 50),
                'avg_rating': round(avg_rating, 1),
                'review_count': int(total_enrollments * random.uniform(0.3, 0.7)),
                'content_quality_score': round(teacher['video_quality_score'] * random.uniform(0.9, 1.1), 1),
                'updated_within_months': (datetime.now().date() - 
                                        datetime.fromisoformat(last_updated).date()).days // 30,
                'plagiarism_check_score': round(random.uniform(85, 100), 1),
                'avg_completion_time_days': round(duration * random.uniform(2, 8), 1),
                'avg_time_per_lesson_minutes': round(random.uniform(15, 45), 1),
                'forum_activity_score': round(random.uniform(0.3, 1.0), 2),
                'return_student_rate': round(random.uniform(0.1, 0.4), 2),
                'preview_to_enrollment_rate': round(random.uniform(0.05, 0.3), 2),
                'add_to_cart_rate': round(random.uniform(0.1, 0.5), 2),
                'cart_abandonment_rate': round(random.uniform(0.2, 0.6), 2),
                'refund_rate': round(random.uniform(0.0, 0.1), 3),
                'created_date': created_date,
                'last_updated_date': last_updated
            }
            data.append(course)

        df = pd.DataFrame(data)
        df.to_sql('courses', self.conn, if_exists='replace', index=False)
        print(f"Generated {len(df)} courses")
        return df

    def generate_enrollments(self, n_enrollments=50000, learners_df=None, 
                           courses_df=None, teachers_df=None) -> pd.DataFrame:
        """Generate enrollments data"""
        print(f"Generating {n_enrollments} enrollments...")

        if learners_df is None:
            learners_df = pd.read_sql("SELECT * FROM learners", self.conn)
        if courses_df is None:
            courses_df = pd.read_sql("SELECT * FROM courses", self.conn)
        if teachers_df is None:
            teachers_df = pd.read_sql("SELECT * FROM teachers", self.conn)

        transaction_types = ["enrollment", "refund"]
        payment_methods = ["wallet", "credit_card", "paypal", "gift_card"]
        device_types = ["mobile", "desktop", "tablet"]
        referral_sources = ["organic", "social_media", "email", "advertisement", "referral"]
        seasons = ["Q1", "Q2", "Q3", "Q4"]

        # Create realistic enrollment patterns with price sensitivity
        # Courses with lower prices should get more enrollments
        courses_df['enrollment_probability'] = courses_df.apply(lambda row: 
            max(0.1, 1.0 - (row['token_price'] - 20) / 200), axis=1)
        
        # Normalize probabilities
        total_prob = courses_df['enrollment_probability'].sum()
        courses_df['enrollment_probability'] = courses_df['enrollment_probability'] / total_prob
        
        data = []
        for i in range(n_enrollments):
            # Select learner based on preferences and price sensitivity
            learner = learners_df.sample(1).iloc[0]
            
            # Select course with realistic price sensitivity
            # Learners prefer courses in their favorite categories AND consider price
            preferred_categories = learner['favorite_categories'].split(',') if pd.notna(learner['favorite_categories']) else []
            
            if preferred_categories and random.random() < 0.6:  # 60% choose preferred category
                available_courses = courses_df[courses_df['category'].isin(preferred_categories)]
                if not available_courses.empty:
                    # Weight by enrollment probability (price sensitivity)
                    course = available_courses.sample(1, weights=available_courses['enrollment_probability']).iloc[0]
                else:
                    course = courses_df.sample(1, weights=courses_df['enrollment_probability']).iloc[0]
            else:
                # Weight by enrollment probability (price sensitivity)
                course = courses_df.sample(1, weights=courses_df['enrollment_probability']).iloc[0]
            
            # Get teacher info
            teacher = teachers_df[teachers_df['teacher_id'] == course['teacher_id']].iloc[0]
            
            # Enrollment date (more realistic distribution)
            # Peak enrollment periods: Q1 (New Year), Q3 (Back to school)
            seasonal_weights = [1.3, 0.8, 1.4, 0.9]  # Q1, Q2, Q3, Q4
            quarter = random.choices([0, 1, 2, 3], weights=seasonal_weights)[0]
            days_in_quarter = random.randint(0, 90)
            enrollment_date = datetime.now() - timedelta(days=365 - (quarter * 90 + days_in_quarter))
            
            # Transaction details
            transaction_type = random.choices(transaction_types, weights=[0.95, 0.05])[0]
            
            # Price sensitivity based on learner characteristics
            base_price = course['token_price']
            learner_price_sensitivity = learner['price_sensitivity_score']
            
            # Apply realistic pricing based on learner sensitivity
            if learner_price_sensitivity > 0.7:  # Price-sensitive learners
                if random.random() < 0.4:  # 40% wait for discounts
                    discount = random.uniform(15, 35)
                    tokens_spent = base_price * (1 - discount / 100)
                else:
                    tokens_spent = base_price
                    discount = 0
            elif learner_price_sensitivity < 0.3:  # Price-insensitive learners
                tokens_spent = base_price
                discount = 0
            else:  # Moderate sensitivity
                if random.random() < 0.2:  # 20% get discounts
                    discount = random.uniform(10, 25)
                    tokens_spent = base_price * (1 - discount / 100)
                else:
                    tokens_spent = base_price
                    discount = 0
            
            tokens_to_teacher = tokens_spent * 0.7
            tokens_to_platform = tokens_spent * 0.3
            
            # Completion status
            completed = 1 if random.random() < course['completion_rate'] else 0
            if completed:
                completion_date = enrollment_date + timedelta(days=course['avg_completion_time_days'])
                time_to_complete = (completion_date - enrollment_date).days
                rating = random.uniform(3.0, 5.0)
            else:
                completion_date = None
                time_to_complete = None
                rating = None

            enrollment = {
                'enrollment_id': f"E{i+1:06d}",
                'learner_id': learner['learner_id'],
                'course_id': course['course_id'],
                'teacher_id': course['teacher_id'],
                'enrollment_date': enrollment_date.isoformat(),
                'transaction_type': transaction_type,
                'tokens_spent': round(tokens_spent, 2),
                'tokens_to_teacher': round(tokens_to_teacher, 2),
                'tokens_to_platform': round(tokens_to_platform, 2),
                'payment_method': random.choice(payment_methods),
                'device_type': random.choice(device_types),
                'referral_source': random.choice(referral_sources),
                'time_of_day': enrollment_date.hour,
                'day_of_week': enrollment_date.weekday(),
                'season': seasons[(enrollment_date.month - 1) // 3],
                'browse_to_enroll_minutes': round(random.uniform(5, 120), 1),
                'courses_viewed_before_purchase': random.randint(1, 10),
                'price_comparison_count': random.randint(0, 5),
                'session_duration_minutes': round(random.uniform(10, 60), 1),
                'pages_viewed': random.randint(5, 30),
                'promo_code_used': 1 if discount > 0 else 0,
                'discount_received_pct': round(discount, 1),
                'is_flash_sale': 1 if random.random() < 0.1 else 0,
                'is_bundle_purchase': 1 if random.random() < 0.15 else 0,
                'loyalty_points_redeemed': random.randint(0, 20) if random.random() < 0.2 else 0,
                'completed': completed,
                'completion_date': completion_date.isoformat() if completion_date else None,
                'time_to_complete_days': time_to_complete,
                'rating_given': round(rating, 1) if rating else None,
                'review_text': f"Great course! Learned a lot." if completed and random.random() < 0.5 else None,
                'refunded': 1 if transaction_type == "refund" else 0,
                'refund_date': (enrollment_date + timedelta(days=random.randint(1, 14))).isoformat() 
                              if transaction_type == "refund" else None
            }
            data.append(enrollment)

        df = pd.DataFrame(data)
        df.to_sql('enrollments', self.conn, if_exists='replace', index=False)
        print(f"Generated {len(df)} enrollments")
        
        # Update course enrollment counts based on actual enrollments
        self.update_course_enrollments()
        
        return df
    
    def update_course_enrollments(self):
        """Update course enrollment counts based on actual enrollments"""
        print("Updating course enrollment counts...")
        
        # Get actual enrollment counts per course
        enrollment_counts_query = """
            SELECT course_id, COUNT(*) as actual_enrollments
            FROM enrollments 
            GROUP BY course_id
        """
        enrollment_counts = pd.read_sql_query(enrollment_counts_query, self.conn)
        
        # Update courses table with actual enrollment counts
        for _, row in enrollment_counts.iterrows():
            update_query = """
                UPDATE courses 
                SET total_enrollments = ? 
                WHERE course_id = ?
            """
            self.conn.execute(update_query, (row['actual_enrollments'], row['course_id']))
        
        self.conn.commit()
        print(f"Updated enrollment counts for {len(enrollment_counts)} courses")
        
        # Update course ratings and reviews based on actual enrollments
        self.update_course_ratings()
    
    def update_course_ratings(self):
        """Update course ratings and review counts based on actual enrollments"""
        print("Updating course ratings and reviews...")
        
        # Get actual ratings and review counts per course
        ratings_query = """
            SELECT 
                course_id,
                AVG(rating_given) as avg_rating,
                COUNT(rating_given) as review_count
            FROM enrollments 
            WHERE rating_given IS NOT NULL
            GROUP BY course_id
        """
        ratings_data = pd.read_sql_query(ratings_query, self.conn)
        
        # Update courses table with actual ratings
        for _, row in ratings_data.iterrows():
            update_query = """
                UPDATE courses 
                SET avg_rating = ?, review_count = ? 
                WHERE course_id = ?
            """
            self.conn.execute(update_query, (round(row['avg_rating'], 1), int(row['review_count']), row['course_id']))
        
        self.conn.commit()
        print(f"Updated ratings for {len(ratings_data)} courses")

    def generate_platform_metrics(self, days=365) -> pd.DataFrame:
        """Generate platform metrics time series"""
        print(f"Generating {days} days of platform metrics...")

        data = []
        base_learners = 10000
        base_teachers = 500
        base_courses = 2000
        
        for i in range(days):
            metric_date = datetime.now() - timedelta(days=days-i-1)
            
            # Simulate growth
            growth_factor = 1 + (i / days) * 0.5  # 50% growth over period
            
            metrics = {
                'metric_date': metric_date.date().isoformat(),
                'total_active_learners': int(base_learners * growth_factor * random.uniform(0.95, 1.05)),
                'total_active_teachers': int(base_teachers * growth_factor * random.uniform(0.95, 1.05)),
                'courses_available': int(base_courses * growth_factor * random.uniform(0.95, 1.05)),
                'avg_category_price': round(random.uniform(80, 120), 2),
                'total_tokens_in_circulation': round(base_learners * 300 * growth_factor, 2),
                'token_velocity': round(random.uniform(0.7, 1.3), 2),
                'token_burn_rate': round(random.uniform(0.75, 0.95), 2),
                'token_mint_rate': round(random.uniform(0.8, 1.2), 2),
                'token_liquidity_ratio': round(random.uniform(0.4, 0.7), 2),
                'platform_growth_rate': round(random.uniform(0.03, 0.08), 3),
                'viral_coefficient': round(random.uniform(1.0, 1.5), 2),
                'daily_active_users': int(base_learners * growth_factor * random.uniform(0.15, 0.25)),
                'monthly_active_users': int(base_learners * growth_factor * random.uniform(0.6, 0.8)),
                'avg_price_elasticity': round(random.uniform(-1.5, -0.8), 2),
                'competitor_pricing_index': round(random.uniform(0.9, 1.1), 2),
                'seasonal_demand_index': round(random.uniform(0.8, 1.2), 2)
            }
            data.append(metrics)

        df = pd.DataFrame(data)
        df.to_sql('platform_metrics', self.conn, if_exists='replace', index=False)
        print(f"Generated {len(df)} platform metric records")
        return df

    def generate_all_data(self, n_learners=10000, n_teachers=500, 
                         n_courses=2000, n_enrollments=50000):
        """Generate all demo data"""
        print("\n" + "="*60)
        print("EDTECH TOKEN ECONOMY DATA GENERATION")
        print("="*60 + "\n")

        self.connect()
        self.create_tables()

        learners_df = self.generate_learners(n_learners)
        teachers_df = self.generate_teachers(n_teachers)
        courses_df = self.generate_courses(n_courses, teachers_df)
        enrollments_df = self.generate_enrollments(n_enrollments, learners_df, 
                                                   courses_df, teachers_df)
        self.generate_platform_metrics(days=365)

        print("\n" + "="*60)
        print("DATA GENERATION COMPLETED!")
        print("="*60)
        print(f"Database: {self.db_path}")
        print(f"Learners: {len(learners_df):,}")
        print(f"Teachers: {len(teachers_df):,}")
        print(f"Courses: {len(courses_df):,}")
        print(f"Enrollments: {len(enrollments_df):,}")
        print("="*60 + "\n")

    def export_to_csv(self, output_dir='data/raw'):
        """Export all tables to CSV files"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Exporting tables to CSV in {output_dir}/...")

        tables = ['learners', 'teachers', 'courses', 'enrollments', 
                 'token_transactions', 'platform_metrics']

        for table in tables:
            try:
                df = pd.read_sql(f"SELECT * FROM {table}", self.conn)
                csv_file = os.path.join(output_dir, f"{table}.csv")
                df.to_csv(csv_file, index=False)
                print(f"Exported {table} to {csv_file}")
            except Exception as e:
                print(f"Failed to export {table}: {e}")


if __name__ == "__main__":
    generator = EdTechTokenEconomyGenerator()
    generator.generate_all_data(
        n_learners=10000,
        n_teachers=500,
        n_courses=2000,
        n_enrollments=50000
    )
    generator.export_to_csv()
    generator.close()


