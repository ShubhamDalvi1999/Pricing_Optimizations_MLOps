import React, { useState, useEffect } from 'react';
import { Brain, BarChart3, Settings, TrendingUp, AlertCircle, CheckCircle } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { apiService } from '../services/api';

const ModelInfo = () => {
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchModelInfo();
  }, []);

  const fetchModelInfo = async () => {
    try {
      setLoading(true);
      const response = await apiService.getModelInfo();
      setModelInfo(response.data);
      setError(null);
    } catch (err) {
      setError('Failed to fetch model information');
      console.error('Error fetching model info:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="loading"></div>
        <span className="ml-2">Loading model information...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-6">
        <div className="flex">
          <AlertCircle className="h-5 w-5 text-red-400" />
          <div className="ml-3">
            <h3 className="text-sm font-medium text-red-800">Error</h3>
            <p className="mt-1 text-sm text-red-700">{error}</p>
          </div>
        </div>
      </div>
    );
  }

  const featureImportanceData = modelInfo?.feature_importance ? 
    Object.entries(modelInfo.feature_importance)
      .map(([feature, importance]) => ({
        feature: feature.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
        importance: importance
      }))
      .sort((a, b) => b.importance - a.importance)
      .slice(0, 10) : [];

  const getPerformanceColor = (value, type) => {
    if (type === 'r2') {
      if (value >= 0.8) return 'text-green-600';
      if (value >= 0.6) return 'text-yellow-600';
      return 'text-red-600';
    }
    if (type === 'rmse') {
      if (value <= 0.4) return 'text-green-600';
      if (value <= 0.6) return 'text-yellow-600';
      return 'text-red-600';
    }
    return 'text-gray-600';
  };

  const getPerformanceLabel = (value, type) => {
    if (type === 'r2') {
      if (value >= 0.8) return 'Excellent';
      if (value >= 0.6) return 'Good';
      return 'Needs Improvement';
    }
    if (type === 'rmse') {
      if (value <= 0.4) return 'Excellent';
      if (value <= 0.6) return 'Good';
      return 'Needs Improvement';
    }
    return '';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center">
          <Brain className="h-8 w-8 text-indigo-600" />
          <div className="ml-4">
            <h1 className="text-2xl font-bold text-gray-900">Model Information</h1>
            <p className="text-gray-600">ML model performance metrics and feature importance</p>
          </div>
        </div>
      </div>

      {/* Model Performance */}
      {modelInfo?.metrics && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="flex items-center">
              <TrendingUp className="h-8 w-8 text-green-600" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">R² Score</p>
                <p className={`text-2xl font-bold ${getPerformanceColor(modelInfo.metrics.r2_score, 'r2')}`}>
                  {modelInfo.metrics.r2_score?.toFixed(3) || 'N/A'}
                </p>
                <p className="text-xs text-gray-500">
                  {getPerformanceLabel(modelInfo.metrics.r2_score, 'r2')}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="flex items-center">
              <BarChart3 className="h-8 w-8 text-blue-600" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">RMSE</p>
                <p className={`text-2xl font-bold ${getPerformanceColor(modelInfo.metrics.rmse, 'rmse')}`}>
                  {modelInfo.metrics.rmse?.toFixed(3) || 'N/A'}
                </p>
                <p className="text-xs text-gray-500">
                  {getPerformanceLabel(modelInfo.metrics.rmse, 'rmse')}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="flex items-center">
              <Settings className="h-8 w-8 text-purple-600" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Price Elasticity</p>
                <p className="text-2xl font-bold text-gray-900">
                  {modelInfo.metrics.price_elasticity?.toFixed(2) || 'N/A'}
                </p>
                <p className="text-xs text-gray-500">
                  {modelInfo.metrics.price_elasticity < -1 ? 'Elastic' : 'Inelastic'}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="flex items-center">
              <Brain className="h-8 w-8 text-indigo-600" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Model Type</p>
                <p className="text-lg font-bold text-gray-900">
                  {modelInfo.parameters?.model_type || 'Gradient Boosting'}
                </p>
                <p className="text-xs text-gray-500">Elasticity Model</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Feature Importance */}
      {featureImportanceData.length > 0 && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Feature Importance</h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-3">Top 10 Features</h4>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={featureImportanceData} layout="horizontal">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis dataKey="feature" type="category" width={120} />
                  <Tooltip 
                    formatter={(value) => [value.toFixed(3), 'Importance']}
                    labelFormatter={(label) => `Feature: ${label}`}
                  />
                  <Bar dataKey="importance" fill="#3b82f6" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-3">Feature Details</h4>
              <div className="space-y-2">
                {featureImportanceData.map((feature, index) => (
                  <div key={feature.feature} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                    <span className="text-sm font-medium text-gray-700">{feature.feature}</span>
                    <div className="flex items-center">
                      <div className="w-20 bg-gray-200 rounded-full h-2 mr-2">
                        <div 
                          className="bg-blue-500 h-2 rounded-full" 
                          style={{ width: `${(feature.importance / featureImportanceData[0].importance) * 100}%` }}
                        ></div>
                      </div>
                      <span className="text-sm text-gray-600">{feature.importance.toFixed(3)}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Model Parameters */}
      {modelInfo?.parameters && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Model Parameters</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {Object.entries(modelInfo.parameters).map(([key, value]) => (
              <div key={key} className="bg-gray-50 rounded-lg p-4">
                <h4 className="text-sm font-medium text-gray-700 mb-1">
                  {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </h4>
                <p className="text-lg font-semibold text-gray-900">
                  {typeof value === 'number' ? value.toFixed(3) : String(value)}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Model Metrics Details */}
      {modelInfo?.metrics && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Detailed Metrics</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Metric
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Value
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Description
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                    R² Score
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {modelInfo.metrics.r2_score?.toFixed(3) || 'N/A'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                      modelInfo.metrics.r2_score >= 0.8 ? 'bg-green-100 text-green-800' :
                      modelInfo.metrics.r2_score >= 0.6 ? 'bg-yellow-100 text-yellow-800' :
                      'bg-red-100 text-red-800'
                    }`}>
                      {getPerformanceLabel(modelInfo.metrics.r2_score, 'r2')}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    Proportion of variance explained by the model
                  </td>
                </tr>
                <tr className="bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                    RMSE
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {modelInfo.metrics.rmse?.toFixed(3) || 'N/A'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                      modelInfo.metrics.rmse <= 0.4 ? 'bg-green-100 text-green-800' :
                      modelInfo.metrics.rmse <= 0.6 ? 'bg-yellow-100 text-yellow-800' :
                      'bg-red-100 text-red-800'
                    }`}>
                      {getPerformanceLabel(modelInfo.metrics.rmse, 'rmse')}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    Root Mean Square Error of predictions
                  </td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                    Price Elasticity
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {modelInfo.metrics.price_elasticity?.toFixed(2) || 'N/A'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                      Math.abs(modelInfo.metrics.price_elasticity) > 1 ? 'bg-blue-100 text-blue-800' :
                      'bg-gray-100 text-gray-800'
                    }`}>
                      {Math.abs(modelInfo.metrics.price_elasticity) > 1 ? 'Elastic' : 'Inelastic'}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    Sensitivity of demand to price changes
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Model Health */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Model Health Status</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="mx-auto w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mb-3">
              <CheckCircle className="h-8 w-8 text-green-600" />
            </div>
            <h4 className="text-sm font-medium text-gray-900">Model Status</h4>
            <p className="text-xs text-gray-500">Active and Ready</p>
          </div>
          <div className="text-center">
            <div className="mx-auto w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mb-3">
              <Settings className="h-8 w-8 text-blue-600" />
            </div>
            <h4 className="text-sm font-medium text-gray-900">Last Updated</h4>
            <p className="text-xs text-gray-500">Pipeline Run</p>
          </div>
          <div className="text-center">
            <div className="mx-auto w-16 h-16 bg-purple-100 rounded-full flex items-center justify-center mb-3">
              <TrendingUp className="h-8 w-8 text-purple-600" />
            </div>
            <h4 className="text-sm font-medium text-gray-900">Performance</h4>
            <p className="text-xs text-gray-500">Optimal</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelInfo;
