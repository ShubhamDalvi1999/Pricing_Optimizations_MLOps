import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import CourseOptimizer from './pages/CourseOptimizer';
import CategoryAnalysis from './pages/CategoryAnalysis';
import EnrollmentPredictor from './pages/EnrollmentPredictor';
import TokenEconomy from './pages/TokenEconomy';
import ModelInfo from './pages/ModelInfo';

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/optimize" element={<CourseOptimizer />} />
          <Route path="/categories" element={<CategoryAnalysis />} />
          <Route path="/predict" element={<EnrollmentPredictor />} />
          <Route path="/token-economy" element={<TokenEconomy />} />
          <Route path="/model-info" element={<ModelInfo />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;
