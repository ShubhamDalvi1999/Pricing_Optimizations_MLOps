import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    console.log(`Making ${config.method?.toUpperCase()} request to ${config.url}`);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// API endpoints
export const apiService = {
  // Health check
  healthCheck: () => api.get('/health'),

  // Get courses with filtering
  getCourses: (params = {}) => {
    const queryParams = new URLSearchParams();
    Object.keys(params).forEach(key => {
      if (params[key] !== undefined && params[key] !== null && params[key] !== '') {
        queryParams.append(key, params[key]);
      }
    });
    return api.get(`/courses?${queryParams.toString()}`);
  },

  // Get category analysis
  getCategoryAnalysis: () => api.get('/categories'),

  // Optimize course price
  optimizePrice: (data) => api.post('/optimize_price', data),

  // Predict enrollments
  predictEnrollments: (data) => api.post('/predict_enrollments', data),

  // Get model info
  getModelInfo: () => api.get('/model_info'),

  // Get token economy metrics
  getTokenEconomyMetrics: (days = 30) => api.get(`/token_economy_metrics?days=${days}`),
};

export default api;
