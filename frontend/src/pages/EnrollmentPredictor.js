import React, { useState } from 'react';
import { TrendingUp, Calculator } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { apiService } from '../services/api';

const EnrollmentPredictor = () => {
  const [formData, setFormData] = useState({
    category: 'Programming',
    subcategory: 'Python',
    difficulty_level: 'intermediate',
    duration_hours: 25.0,
    teacher_quality_score: 82.0,
    current_price: 100.0,
    current_enrollments: 200,
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const categories = [
    'Programming', 'Design', 'Business', 'Marketing', 'Data Science', 
    'Language', 'Photography', 'Music', 'Health', 'Fitness'
  ];

  const subcategories = {
    'Programming': ['Python', 'JavaScript', 'Java', 'C++', 'React', 'Node.js'],
    'Design': ['UI/UX', 'Graphic Design', 'Web Design', 'Logo Design', 'Illustration'],
    'Business': ['Entrepreneurship', 'Management', 'Finance', 'Strategy', 'Leadership'],
    'Marketing': ['Digital Marketing', 'Social Media', 'SEO', 'Content Marketing', 'Analytics'],
    'Data Science': ['Machine Learning', 'Statistics', 'Python', 'R', 'SQL'],
    'Language': ['English', 'Spanish', 'French', 'German', 'Chinese'],
    'Photography': ['Portrait', 'Landscape', 'Street', 'Wedding', 'Commercial'],
    'Music': ['Guitar', 'Piano', 'Singing', 'Production', 'Theory'],
    'Health': ['Nutrition', 'Mental Health', 'Fitness', 'Yoga', 'Meditation'],
    'Fitness': ['Weight Training', 'Cardio', 'Yoga', 'Pilates', 'CrossFit']
  };

  const difficultyLevels = ['beginner', 'intermediate', 'advanced'];

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await apiService.predictEnrollments({
        ...formData,
        duration_hours: parseFloat(formData.duration_hours),
        teacher_quality_score: parseFloat(formData.teacher_quality_score),
        current_price: parseFloat(formData.current_price),
        current_enrollments: parseInt(formData.current_enrollments),
      });
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred while predicting enrollments');
    } finally {
      setLoading(false);
    }
  };

  const chartData = result?.predictions?.map(pred => ({
    price: pred.token_price,
    enrollments: pred.predicted_enrollments,
    revenue: pred.predicted_revenue,
    priceChange: pred.price_change_pct
  })) || [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center">
          <TrendingUp className="h-8 w-8 text-purple-600" />
          <div className="ml-4">
            <h1 className="text-2xl font-bold text-gray-900">Enrollment Predictor</h1>
            <p className="text-gray-600">Predict enrollment demand at different price points</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input Form */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Course Features</h2>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label htmlFor="category" className="block text-sm font-medium text-gray-700 mb-2">
                Category
              </label>
              <select
                id="category"
                name="category"
                value={formData.category}
                onChange={handleInputChange}
                className="select"
                required
              >
                {categories.map(cat => (
                  <option key={cat} value={cat}>{cat}</option>
                ))}
              </select>
            </div>

            <div>
              <label htmlFor="subcategory" className="block text-sm font-medium text-gray-700 mb-2">
                Subcategory
              </label>
              <select
                id="subcategory"
                name="subcategory"
                value={formData.subcategory}
                onChange={handleInputChange}
                className="select"
                required
              >
                {subcategories[formData.category]?.map(sub => (
                  <option key={sub} value={sub}>{sub}</option>
                ))}
              </select>
            </div>

            <div>
              <label htmlFor="difficulty_level" className="block text-sm font-medium text-gray-700 mb-2">
                Difficulty Level
              </label>
              <select
                id="difficulty_level"
                name="difficulty_level"
                value={formData.difficulty_level}
                onChange={handleInputChange}
                className="select"
                required
              >
                {difficultyLevels.map(level => (
                  <option key={level} value={level}>{level.charAt(0).toUpperCase() + level.slice(1)}</option>
                ))}
              </select>
            </div>

            <div>
              <label htmlFor="duration_hours" className="block text-sm font-medium text-gray-700 mb-2">
                Duration (hours)
              </label>
              <input
                type="number"
                id="duration_hours"
                name="duration_hours"
                value={formData.duration_hours}
                onChange={handleInputChange}
                className="input"
                min="0"
                step="0.1"
                required
              />
            </div>

            <div>
              <label htmlFor="teacher_quality_score" className="block text-sm font-medium text-gray-700 mb-2">
                Teacher Quality Score (0-100)
              </label>
              <input
                type="number"
                id="teacher_quality_score"
                name="teacher_quality_score"
                value={formData.teacher_quality_score}
                onChange={handleInputChange}
                className="input"
                min="0"
                max="100"
                step="0.1"
                required
              />
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
                  Predicting...
                </>
              ) : (
                <>
                  <Calculator className="h-4 w-4 mr-2" />
                  Predict Enrollments
                </>
              )}
            </button>
          </form>
        </div>

        {/* Results */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Prediction Results</h2>
          
          {error && (
            <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-sm text-red-700">{error}</p>
            </div>
          )}

          {result && (
            <div className="space-y-4">
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <h3 className="text-sm font-medium text-blue-800 mb-2">Elasticity Analysis</h3>
                <p className="text-sm text-blue-700">
                  Elasticity Coefficient: <span className="font-semibold">{result.elasticity_coefficient.toFixed(2)}</span>
                </p>
              </div>

              {chartData.length > 0 && (
                <div>
                  <h3 className="text-sm font-medium text-gray-700 mb-3">Price vs Enrollments</h3>
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="price" />
                      <YAxis />
                      <Tooltip 
                        formatter={(value, name) => [
                          name === 'enrollments' ? value.toLocaleString() : `${value.toFixed(1)} tokens`,
                          name === 'enrollments' ? 'Enrollments' : 'Price'
                        ]}
                      />
                      <Line type="monotone" dataKey="enrollments" stroke="#3b82f6" strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}

              <div>
                <h3 className="text-sm font-medium text-gray-700 mb-3">Revenue Analysis</h3>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="price" />
                    <YAxis />
                    <Tooltip 
                      formatter={(value) => [`$${value.toLocaleString()}`, 'Revenue']}
                      labelFormatter={(label) => `Price: ${label} tokens`}
                    />
                    <Bar dataKey="revenue" fill="#10b981" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {!result && !error && (
            <div className="text-center py-8">
              <TrendingUp className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500">Enter course details to predict enrollment demand</p>
            </div>
          )}
        </div>
      </div>

      {/* Detailed Predictions Table */}
      {result && result.predictions && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Detailed Predictions</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Price
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Price Change
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Predicted Enrollments
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Predicted Revenue
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {result.predictions.map((prediction, index) => (
                  <tr key={index} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {prediction.token_price} tokens
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                        prediction.price_change_pct > 0 
                          ? 'bg-red-100 text-red-800' 
                          : prediction.price_change_pct < 0 
                          ? 'bg-green-100 text-green-800'
                          : 'bg-gray-100 text-gray-800'
                      }`}>
                        {prediction.price_change_pct > 0 ? '+' : ''}{prediction.price_change_pct}%
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {prediction.predicted_enrollments.toLocaleString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      ${prediction.predicted_revenue.toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default EnrollmentPredictor;
