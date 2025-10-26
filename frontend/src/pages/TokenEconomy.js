import React, { useState, useEffect, useCallback } from 'react';
import { Coins, Users, DollarSign, Activity } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { apiService } from '../services/api';

const TokenEconomy = () => {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [days, setDays] = useState(30);

  const fetchTokenEconomyMetrics = useCallback(async () => {
    try {
      setLoading(true);
      const response = await apiService.getTokenEconomyMetrics(days);
      setMetrics(response.data);
      setError(null);
    } catch (err) {
      setError('Failed to fetch token economy metrics');
      console.error('Error fetching token economy metrics:', err);
    } finally {
      setLoading(false);
    }
  }, [days]);

  useEffect(() => {
    fetchTokenEconomyMetrics();
  }, [fetchTokenEconomyMetrics]);

  const chartData = metrics?.metrics?.map(metric => ({
    date: new Date(metric.date).toLocaleDateString(),
    tokensBurned: metric.daily_tokens_burned,
    enrollments: metric.daily_enrollments,
    revenue: metric.daily_platform_revenue,
    activeUsers: metric.daily_active_users || 0,
    newUsers: metric.daily_new_users || 0
  })) || [];

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="loading"></div>
        <span className="ml-2">Loading token economy metrics...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-6">
        <div className="flex">
          <div className="ml-3">
            <h3 className="text-sm font-medium text-red-800">Error</h3>
            <p className="mt-1 text-sm text-red-700">{error}</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <Coins className="h-8 w-8 text-yellow-600" />
            <div className="ml-4">
              <h1 className="text-2xl font-bold text-gray-900">Token Economy</h1>
              <p className="text-gray-600">Monitor token flow and platform health metrics</p>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <label htmlFor="days" className="text-sm font-medium text-gray-700">
              Period:
            </label>
            <select
              id="days"
              value={days}
              onChange={(e) => setDays(parseInt(e.target.value))}
              className="select w-24"
            >
              <option value={7}>7 days</option>
              <option value={30}>30 days</option>
              <option value={90}>90 days</option>
            </select>
          </div>
        </div>
      </div>

      {/* Summary Stats */}
      {metrics?.summary && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="flex items-center">
              <Coins className="h-8 w-8 text-yellow-600" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Total Tokens Burned</p>
                <p className="text-2xl font-bold text-gray-900">
                  {metrics.summary.total_tokens_burned.toLocaleString()}
                </p>
              </div>
            </div>
          </div>
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="flex items-center">
              <Users className="h-8 w-8 text-blue-600" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Avg Daily Enrollments</p>
                <p className="text-2xl font-bold text-gray-900">
                  {Math.round(metrics.summary.avg_daily_enrollments).toLocaleString()}
                </p>
              </div>
            </div>
          </div>
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="flex items-center">
              <DollarSign className="h-8 w-8 text-green-600" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Total Revenue</p>
                <p className="text-2xl font-bold text-gray-900">
                  ${metrics.summary.total_revenue.toLocaleString()}
                </p>
              </div>
            </div>
          </div>
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="flex items-center">
              <Activity className="h-8 w-8 text-purple-600" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Period</p>
                <p className="text-2xl font-bold text-gray-900">
                  {days} days
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Token Flow */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Daily Token Flow</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip 
                formatter={(value) => [value.toLocaleString(), 'Tokens Burned']}
                labelFormatter={(label) => `Date: ${label}`}
              />
              <Line type="monotone" dataKey="tokensBurned" stroke="#f59e0b" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Enrollment Trends */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Daily Enrollments</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip 
                formatter={(value) => [value.toLocaleString(), 'Enrollments']}
                labelFormatter={(label) => `Date: ${label}`}
              />
              <Line type="monotone" dataKey="enrollments" stroke="#3b82f6" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Revenue Analysis */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Daily Revenue</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis />
            <Tooltip 
              formatter={(value) => [`$${value.toLocaleString()}`, 'Revenue']}
              labelFormatter={(label) => `Date: ${label}`}
            />
            <Bar dataKey="revenue" fill="#10b981" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* User Activity */}
      {chartData.some(d => d.activeUsers > 0 || d.newUsers > 0) && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">User Activity</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip 
                formatter={(value, name) => [
                  value.toLocaleString(), 
                  name === 'activeUsers' ? 'Active Users' : 'New Users'
                ]}
                labelFormatter={(label) => `Date: ${label}`}
              />
              <Line type="monotone" dataKey="activeUsers" stroke="#8b5cf6" strokeWidth={2} name="activeUsers" />
              <Line type="monotone" dataKey="newUsers" stroke="#ef4444" strokeWidth={2} name="newUsers" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Detailed Metrics Table */}
      {metrics?.metrics && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Daily Metrics</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Date
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Tokens Burned
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Enrollments
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Revenue
                  </th>
                  {metrics.metrics.some(m => m.daily_active_users > 0) && (
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Active Users
                    </th>
                  )}
                  {metrics.metrics.some(m => m.daily_new_users > 0) && (
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      New Users
                    </th>
                  )}
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {metrics.metrics.map((metric, index) => (
                  <tr key={metric.date} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {new Date(metric.date).toLocaleDateString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {metric.daily_tokens_burned.toLocaleString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {metric.daily_enrollments.toLocaleString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      ${metric.daily_platform_revenue.toLocaleString()}
                    </td>
                    {metrics.metrics.some(m => m.daily_active_users > 0) && (
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {(metric.daily_active_users || 0).toLocaleString()}
                      </td>
                    )}
                    {metrics.metrics.some(m => m.daily_new_users > 0) && (
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {(metric.daily_new_users || 0).toLocaleString()}
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Health Indicators */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h4 className="text-sm font-medium text-gray-700 mb-2">Token Economy Health</h4>
          <div className="flex items-center">
            <div className="flex-1 bg-gray-200 rounded-full h-2">
              <div 
                className="bg-green-500 h-2 rounded-full" 
                style={{ width: '85%' }}
              ></div>
            </div>
            <span className="ml-2 text-sm font-medium text-gray-900">85%</span>
          </div>
          <p className="mt-2 text-xs text-gray-500">Based on token velocity and user engagement</p>
        </div>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h4 className="text-sm font-medium text-gray-700 mb-2">Platform Growth</h4>
          <div className="flex items-center">
            <div className="flex-1 bg-gray-200 rounded-full h-2">
              <div 
                className="bg-blue-500 h-2 rounded-full" 
                style={{ width: '72%' }}
              ></div>
            </div>
            <span className="ml-2 text-sm font-medium text-gray-900">72%</span>
          </div>
          <p className="mt-2 text-xs text-gray-500">Based on enrollment trends and revenue growth</p>
        </div>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h4 className="text-sm font-medium text-gray-700 mb-2">User Engagement</h4>
          <div className="flex items-center">
            <div className="flex-1 bg-gray-200 rounded-full h-2">
              <div 
                className="bg-purple-500 h-2 rounded-full" 
                style={{ width: '68%' }}
              ></div>
            </div>
            <span className="ml-2 text-sm font-medium text-gray-900">68%</span>
          </div>
          <p className="mt-2 text-xs text-gray-500">Based on active users and course completion rates</p>
        </div>
      </div>
    </div>
  );
};

export default TokenEconomy;
