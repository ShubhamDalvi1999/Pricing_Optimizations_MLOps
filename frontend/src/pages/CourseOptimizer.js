import React, { useState, useEffect } from 'react';
import { Calculator, TrendingUp, DollarSign, Users, AlertCircle, CheckCircle } from 'lucide-react';
import { apiService } from '../services/api';

const CourseOptimizer = () => {
  const [formData, setFormData] = useState({
    course_id: '',
    current_price: '',
    current_enrollments: '',
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [courses, setCourses] = useState([]);

  useEffect(() => {
    fetchCourses();
  }, []);

  const fetchCourses = async () => {
    try {
      const response = await apiService.getCourses({ limit: 50 });
      setCourses(response.data);
    } catch (err) {
      console.error('Error fetching courses:', err);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleCourseChange = (e) => {
    const courseId = e.target.value;
    const selectedCourse = courses.find(course => course.course_id === courseId);
    
    setFormData(prev => ({
      ...prev,
      course_id: courseId,
      current_price: selectedCourse ? selectedCourse.token_price : '',
      current_enrollments: selectedCourse ? selectedCourse.total_enrollments : ''
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await apiService.optimizePrice({
        course_id: formData.course_id,
        current_price: parseFloat(formData.current_price),
        current_enrollments: parseInt(formData.current_enrollments),
      });
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred while optimizing price');
    } finally {
      setLoading(false);
    }
  };

  const getDemandTypeColor = (demandType) => {
    switch (demandType?.toLowerCase()) {
      case 'elastic':
        return 'text-green-600 bg-green-100';
      case 'inelastic':
        return 'text-red-600 bg-red-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const getRecommendationColor = (increasePct) => {
    if (increasePct > 5) return 'text-green-600';
    if (increasePct > 0) return 'text-yellow-600';
    return 'text-gray-600';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center">
          <Calculator className="h-8 w-8 text-blue-600" />
          <div className="ml-4">
            <h1 className="text-2xl font-bold text-gray-900">Course Price Optimizer</h1>
            <p className="text-gray-600">Get AI-powered recommendations for optimal token pricing</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input Form */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Course Information</h2>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label htmlFor="course_id" className="block text-sm font-medium text-gray-700 mb-2">
                Course ID
              </label>
              <select
                id="course_id"
                name="course_id"
                value={formData.course_id}
                onChange={handleCourseChange}
                className="select"
                required
              >
                <option value="">Select a course</option>
                {courses.map((course) => (
                  <option key={course.course_id} value={course.course_id}>
                    {course.course_id} - {course.course_title} ({course.category})
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label htmlFor="current_price" className="block text-sm font-medium text-gray-700 mb-2">
                Current Token Price
              </label>
              <input
                type="number"
                id="current_price"
                name="current_price"
                value={formData.current_price}
                onChange={handleInputChange}
                className="input"
                placeholder="e.g., 100"
                min="0"
                step="0.01"
                required
              />
            </div>

            <div>
              <label htmlFor="current_enrollments" className="block text-sm font-medium text-gray-700 mb-2">
                Current Enrollments
              </label>
              <input
                type="number"
                id="current_enrollments"
                name="current_enrollments"
                value={formData.current_enrollments}
                onChange={handleInputChange}
                className="input"
                placeholder="e.g., 200"
                min="0"
                required
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full btn btn-primary"
            >
              {loading ? (
                <>
                  <div className="loading mr-2"></div>
                  Optimizing...
                </>
              ) : (
                <>
                  <Calculator className="h-4 w-4 mr-2" />
                  Optimize Price
                </>
              )}
            </button>
          </form>
        </div>

        {/* Results */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Optimization Results</h2>
          
          {error && (
            <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg">
              <div className="flex">
                <AlertCircle className="h-5 w-5 text-red-400" />
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-red-800">Error</h3>
                  <p className="mt-1 text-sm text-red-700">{error}</p>
                </div>
              </div>
            </div>
          )}

          {result && (
            <div className="space-y-6">
              {/* Price Recommendation */}
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <div className="flex items-center mb-2">
                  <CheckCircle className="h-5 w-5 text-blue-600" />
                  <h3 className="ml-2 text-lg font-semibold text-blue-900">Price Recommendation</h3>
                </div>
                <div className="grid grid-cols-2 gap-4 mt-3">
                  <div>
                    <p className="text-sm text-blue-700">Current Price</p>
                    <p className="text-2xl font-bold text-blue-900">{result.current_price} tokens</p>
                  </div>
                  <div>
                    <p className="text-sm text-blue-700">Optimal Price</p>
                    <p className="text-2xl font-bold text-blue-900">{result.optimal_price} tokens</p>
                  </div>
                </div>
                <div className="mt-3">
                  <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                    result.price_change_pct > 0 ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'
                  }`}>
                    {result.price_change_pct > 0 ? '+' : ''}{result.price_change_pct?.toFixed(1) || '0'}% change
                  </span>
                </div>
              </div>

              {/* Revenue Impact */}
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-50 rounded-lg p-4">
                  <div className="flex items-center">
                    <DollarSign className="h-5 w-5 text-gray-600" />
                    <span className="ml-2 text-sm font-medium text-gray-700">Current Revenue</span>
                  </div>
                  <p className="mt-1 text-2xl font-bold text-gray-900">
                    ${result.current_revenue?.toLocaleString() || '0'}
                  </p>
                </div>
                <div className="bg-gray-50 rounded-lg p-4">
                  <div className="flex items-center">
                    <TrendingUp className="h-5 w-5 text-gray-600" />
                    <span className="ml-2 text-sm font-medium text-gray-700">Optimal Revenue</span>
                  </div>
                  <p className="mt-1 text-2xl font-bold text-gray-900">
                    ${result.optimal_revenue?.toLocaleString() || '0'}
                  </p>
                </div>
              </div>

              {/* Revenue Increase */}
              <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                <h4 className="text-sm font-medium text-green-800 mb-2">Revenue Impact</h4>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-green-700">Revenue Increase</span>
                  <span className={`text-lg font-bold ${getRecommendationColor(result.revenue_increase_pct)}`}>
                    +${result.revenue_increase?.toLocaleString() || '0'} ({result.revenue_increase_pct?.toFixed(1) || '0'}%)
                  </span>
                </div>
              </div>

              {/* Enrollment Prediction */}
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-50 rounded-lg p-4">
                  <div className="flex items-center">
                    <Users className="h-5 w-5 text-gray-600" />
                    <span className="ml-2 text-sm font-medium text-gray-700">Current Enrollments</span>
                  </div>
                  <p className="mt-1 text-2xl font-bold text-gray-900">
                    {result.current_enrollments?.toLocaleString() || '0'}
                  </p>
                </div>
                <div className="bg-gray-50 rounded-lg p-4">
                  <div className="flex items-center">
                    <TrendingUp className="h-5 w-5 text-gray-600" />
                    <span className="ml-2 text-sm font-medium text-gray-700">Predicted Enrollments</span>
                  </div>
                  <p className="mt-1 text-2xl font-bold text-gray-900">
                    {result.predicted_enrollments?.toLocaleString() || '0'}
                  </p>
                </div>
              </div>

              {/* Demand Analysis */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-700">Elasticity Coefficient</span>
                  <span className="text-sm font-bold text-gray-900">{result.elasticity_coefficient?.toFixed(2) || '0.00'}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-700">Demand Type</span>
                  <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getDemandTypeColor(result.demand_type)}`}>
                    {result.demand_type}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-700">Enrollment Change</span>
                  <span className={`text-sm font-bold ${result.enrollment_change_pct > 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {result.enrollment_change_pct > 0 ? '+' : ''}{result.enrollment_change_pct?.toFixed(1) || '0'}%
                  </span>
                </div>
              </div>

              {/* Recommendation */}
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                <h4 className="text-sm font-medium text-yellow-800 mb-2">AI Recommendation</h4>
                <p className="text-sm text-yellow-700">{result.recommendation}</p>
              </div>
            </div>
          )}

          {!result && !error && (
            <div className="text-center py-8">
              <Calculator className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500">Enter course details to get price optimization recommendations</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default CourseOptimizer;
