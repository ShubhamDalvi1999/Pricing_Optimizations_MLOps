import React, { useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { 
  TrendingUp, 
  Calculator, 
  BarChart3, 
  Brain, 
  Coins, 
  Activity,
  ArrowRight,
  CheckCircle,
  AlertCircle,
  Zap
} from 'lucide-react';
import { apiService } from '../services/api';

const Dashboard = () => {
  const [healthStatus, setHealthStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const checkHealth = useCallback(async () => {
    try {
      console.log('Checking API health...');
      const response = await apiService.healthCheck();
      console.log('API Health Response:', response.data);
      setHealthStatus(response.data);
      setError(null);
    } catch (err) {
      console.error('API Health Check Failed:', err);
      setError(`Unable to connect to API server at ${process.env.REACT_APP_API_URL || 'http://localhost:8000'}. Please ensure the backend is running.`);
      setHealthStatus({ 
        status: 'unhealthy',
        database: 'disconnected',
        model: 'not loaded'
      });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    checkHealth();
  }, [checkHealth]);

  const features = [
    {
      title: 'Course Price Optimizer',
      description: 'Get AI-powered recommendations for optimal token pricing',
      icon: Calculator,
      href: '/optimize',
      color: 'bg-blue-500',
    },
    {
      title: 'Category Analysis',
      description: 'Analyze performance metrics across course categories',
      icon: BarChart3,
      href: '/categories',
      color: 'bg-green-500',
    },
    {
      title: 'Enrollment Predictor',
      description: 'Predict enrollment demand at different price points',
      icon: TrendingUp,
      href: '/predict',
      color: 'bg-purple-500',
    },
    {
      title: 'Token Economy',
      description: 'Monitor token flow and platform health metrics',
      icon: Coins,
      href: '/token-economy',
      color: 'bg-yellow-500',
    },
    {
      title: 'Model Information',
      description: 'View ML model performance and feature importance',
      icon: Brain,
      href: '/model-info',
      color: 'bg-indigo-500',
    },
  ];

  const stats = [
    {
      name: 'API Status',
      value: healthStatus?.status === 'healthy' ? 'Online' : 'Offline',
      icon: Activity,
      color: healthStatus?.status === 'healthy' ? 'text-green-600' : 'text-red-600',
    },
    {
      name: 'Database',
      value: healthStatus?.database === 'connected' ? 'Connected' : 'Disconnected',
      icon: CheckCircle,
      color: healthStatus?.database === 'connected' ? 'text-green-600' : 'text-red-600',
    },
    {
      name: 'ML Model',
      value: healthStatus?.model === 'loaded' ? 'Loaded' : 'Not Loaded',
      icon: Brain,
      color: healthStatus?.model === 'loaded' ? 'text-green-600' : 'text-red-600',
    },
  ];

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="loading"></div>
        <span className="ml-2">Checking system status...</span>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="text-center">
            <div className="flex justify-center mb-6">
              <div className="p-3 bg-primary-100 rounded-full">
                <Zap className="h-8 w-8 text-primary-600" />
              </div>
            </div>
            <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
              EdTech Token Economy
            </h1>
            <p className="text-xl md:text-2xl text-gray-600 mb-8 max-w-3xl mx-auto">
              ML-powered platform for optimizing token-based pricing in educational technology
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link to="/optimize" className="btn btn-primary text-lg px-8 py-3">
                <Calculator className="h-5 w-5 mr-2" />
                Start Optimizing
              </Link>
              <Link to="/categories" className="btn btn-secondary text-lg px-8 py-3">
                <BarChart3 className="h-5 w-5 mr-2" />
                View Analytics
              </Link>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-12">

        {/* Error Alert */}
        {error && (
          <div className="mb-8 bg-red-50 border-l-4 border-red-400 p-4 rounded-r-lg">
            <div className="flex">
              <AlertCircle className="h-5 w-5 text-red-400" />
              <div className="ml-3 flex-1">
                <h3 className="text-sm font-medium text-red-800">Connection Error</h3>
                <div className="mt-2 text-sm text-red-700">
                  <p>{error}</p>
                  <p className="mt-1">
                    Make sure to start the API server: <code className="bg-red-100 px-1 rounded">cd api && uvicorn main:app --reload</code>
                  </p>
                </div>
              </div>
              <button
                onClick={checkHealth}
                className="ml-4 btn btn-secondary text-sm"
              >
                Retry
              </button>
            </div>
          </div>
        )}

        {/* System Status */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-gray-900">System Status</h2>
            <button
              onClick={checkHealth}
              disabled={loading}
              className="btn btn-secondary text-sm"
            >
              {loading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-600 mr-2"></div>
                  Checking...
                </>
              ) : (
                <>
                  <Activity className="h-4 w-4 mr-2" />
                  Refresh Status
                </>
              )}
            </button>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {stats.map((stat) => {
            const Icon = stat.icon;
            return (
              <div key={stat.name} className="card hover:shadow-md transition-shadow duration-200">
                <div className="flex items-center">
                  <div className={`p-3 rounded-full ${stat.value === 'Online' || stat.value === 'Connected' || stat.value === 'Loaded' ? 'bg-green-100' : 'bg-red-100'}`}>
                    <Icon className={`h-6 w-6 ${stat.value === 'Online' || stat.value === 'Connected' || stat.value === 'Loaded' ? 'text-green-600' : 'text-red-600'}`} />
                  </div>
                  <div className="ml-4">
                    <p className="text-sm font-medium text-gray-600">{stat.name}</p>
                    <p className={`text-xl font-semibold ${stat.value === 'Online' || stat.value === 'Connected' || stat.value === 'Loaded' ? 'text-green-600' : 'text-red-600'}`}>
                      {stat.value}
                    </p>
                  </div>
                </div>
              </div>
            );
          })}
          </div>
        </div>

        {/* Features Grid */}
        <div className="mb-12">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Platform Features</h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Comprehensive tools for optimizing your EdTech token economy
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature) => {
              const Icon = feature.icon;
              return (
                <Link
                  key={feature.title}
                  to={feature.href}
                  className="group card hover:shadow-lg transition-all duration-200 hover:-translate-y-1"
                >
                  <div className="flex items-start">
                    <div className={`flex-shrink-0 p-3 rounded-lg ${feature.color} text-white group-hover:scale-110 transition-transform duration-200`}>
                      <Icon className="h-6 w-6" />
                    </div>
                    <div className="ml-4 flex-1">
                      <h3 className="text-lg font-semibold text-gray-900 group-hover:text-primary-600 transition-colors">
                        {feature.title}
                      </h3>
                      <p className="mt-2 text-sm text-gray-600">{feature.description}</p>
                      <div className="mt-4 flex items-center text-sm font-medium text-primary-600 group-hover:text-primary-700">
                        Explore feature
                        <ArrowRight className="ml-1 h-4 w-4 group-hover:translate-x-1 transition-transform" />
                      </div>
                    </div>
                  </div>
                </Link>
              );
            })}
          </div>
        </div>

        {/* Quick Actions */}
        <div className="card">
          <div className="text-center mb-6">
            <h3 className="text-2xl font-bold text-gray-900 mb-2">Quick Actions</h3>
            <p className="text-gray-600">Get started with these essential tools</p>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <Link
              to="/optimize"
              className="group flex items-center justify-center px-6 py-4 border border-gray-300 rounded-lg text-sm font-medium text-gray-700 hover:bg-primary-50 hover:border-primary-300 hover:text-primary-700 transition-all duration-200"
            >
              <Calculator className="h-5 w-5 mr-2 group-hover:scale-110 transition-transform" />
              Optimize Price
            </Link>
            <Link
              to="/predict"
              className="group flex items-center justify-center px-6 py-4 border border-gray-300 rounded-lg text-sm font-medium text-gray-700 hover:bg-primary-50 hover:border-primary-300 hover:text-primary-700 transition-all duration-200"
            >
              <TrendingUp className="h-5 w-5 mr-2 group-hover:scale-110 transition-transform" />
              Predict Demand
            </Link>
            <Link
              to="/categories"
              className="group flex items-center justify-center px-6 py-4 border border-gray-300 rounded-lg text-sm font-medium text-gray-700 hover:bg-primary-50 hover:border-primary-300 hover:text-primary-700 transition-all duration-200"
            >
              <BarChart3 className="h-5 w-5 mr-2 group-hover:scale-110 transition-transform" />
              View Analytics
            </Link>
            <Link
              to="/model-info"
              className="group flex items-center justify-center px-6 py-4 border border-gray-300 rounded-lg text-sm font-medium text-gray-700 hover:bg-primary-50 hover:border-primary-300 hover:text-primary-700 transition-all duration-200"
            >
              <Brain className="h-5 w-5 mr-2 group-hover:scale-110 transition-transform" />
              Model Details
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
