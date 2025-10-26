# EdTech Token Economy - React Frontend

A modern, responsive React frontend for the EdTech Token Economy ML platform. This frontend provides an intuitive interface for interacting with all the FastAPI endpoints and visualizing ML model results.

## 🚀 Features

- **Modern UI/UX**: Clean, professional design with responsive layout
- **Complete API Integration**: All FastAPI endpoints are integrated and functional
- **Interactive Charts**: Data visualization using Recharts library
- **Real-time Data**: Live updates from the ML models
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Error Handling**: Comprehensive error handling and user feedback

## 📱 Pages & Features

### 1. Dashboard
- System health monitoring
- Quick access to all features
- API status indicators
- Feature overview cards

### 2. Course Price Optimizer
- Interactive form for course details
- Real-time price optimization recommendations
- Revenue impact analysis
- Elasticity coefficient display

### 3. Category Analysis
- Performance metrics by course category
- Interactive charts and graphs
- Revenue distribution analysis
- Detailed category comparison table

### 4. Enrollment Predictor
- Course feature input form
- Price sensitivity analysis
- Enrollment predictions at different price points
- Revenue optimization charts

### 5. Token Economy
- Token flow monitoring
- Platform health metrics
- User activity tracking
- Revenue trends analysis

### 6. Model Information
- ML model performance metrics
- Feature importance visualization
- Model parameters display
- Health status indicators

## 🛠️ Technology Stack

- **React 18**: Modern React with hooks
- **React Router**: Client-side routing
- **Axios**: HTTP client for API calls
- **Recharts**: Data visualization library
- **Lucide React**: Modern icon library
- **CSS3**: Custom styling with utility classes

## 📦 Installation

### Prerequisites

- Node.js (v14 or higher)
- npm or yarn
- Running EdTech Token Economy API server

### Setup

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm start
   # or
   npm run dev
   ```

4. **Open your browser:**
   Navigate to `http://localhost:3000`

## 🔧 Configuration

### API Configuration

The frontend is configured to connect to the FastAPI backend. You can modify the API URL by:

1. **Environment Variables:**
   Create a `.env` file in the frontend directory:
   ```env
   REACT_APP_API_URL=http://localhost:8000
   ```

2. **Default Configuration:**
   If no environment variable is set, it defaults to `http://localhost:8000`

### Backend Requirements

Make sure the FastAPI backend is running:

```bash
# In the main project directory
cd api
uvicorn main:app --reload
```

The API should be accessible at `http://localhost:8000`

## 📊 Available Scripts

- `npm start`: Start development server
- `npm run dev`: Start development server (alias for start)
- `npm build`: Build for production
- `npm test`: Run tests
- `npm eject`: Eject from Create React App

## 🎨 UI Components

### Layout
- Responsive sidebar navigation
- Mobile-friendly hamburger menu
- Clean header with page titles
- Consistent spacing and typography

### Forms
- Input validation
- Loading states
- Error handling
- Success feedback

### Charts
- Interactive bar charts
- Line charts for trends
- Pie charts for distributions
- Responsive design

### Tables
- Sortable columns
- Responsive design
- Hover effects
- Clean typography

## 🔌 API Integration

All FastAPI endpoints are integrated:

- `GET /health` - System health check
- `GET /courses` - Course listing with filters
- `GET /categories` - Category analysis
- `POST /optimize_price` - Price optimization
- `POST /predict_enrollments` - Enrollment prediction
- `GET /model_info` - Model information
- `GET /token_economy_metrics` - Token economy metrics

## 📱 Responsive Design

The frontend is fully responsive and works on:

- **Desktop**: Full sidebar navigation
- **Tablet**: Collapsible sidebar
- **Mobile**: Hamburger menu navigation

## 🎯 Key Features

### Real-time Updates
- Live API status monitoring
- Dynamic data loading
- Error state handling

### Interactive Visualizations
- Hover tooltips on charts
- Clickable elements
- Responsive chart sizing

### User Experience
- Loading indicators
- Error messages
- Success feedback
- Intuitive navigation

## 🚀 Deployment

### Production Build

```bash
npm run build
```

This creates a `build` folder with optimized production files.

### Deployment Options

- **Static Hosting**: Netlify, Vercel, GitHub Pages
- **CDN**: CloudFront, CloudFlare
- **Server**: Nginx, Apache

### Environment Variables

For production, set:
```env
REACT_APP_API_URL=https://your-api-domain.com
```

## 🔧 Development

### Project Structure

```
frontend/
├── public/
│   ├── index.html
│   └── manifest.json
├── src/
│   ├── components/
│   │   └── Layout.js
│   ├── pages/
│   │   ├── Dashboard.js
│   │   ├── CourseOptimizer.js
│   │   ├── CategoryAnalysis.js
│   │   ├── EnrollmentPredictor.js
│   │   ├── TokenEconomy.js
│   │   └── ModelInfo.js
│   ├── services/
│   │   └── api.js
│   ├── App.js
│   ├── index.js
│   └── index.css
├── package.json
└── README.md
```

### Adding New Features

1. Create new page components in `src/pages/`
2. Add API endpoints in `src/services/api.js`
3. Update routing in `src/App.js`
4. Add navigation items in `src/components/Layout.js`

## 🐛 Troubleshooting

### Common Issues

1. **API Connection Error**
   - Ensure the FastAPI server is running
   - Check the API URL configuration
   - Verify CORS settings

2. **Charts Not Rendering**
   - Check browser console for errors
   - Ensure data is properly formatted
   - Verify Recharts installation

3. **Build Errors**
   - Clear node_modules and reinstall
   - Check for TypeScript errors
   - Verify all dependencies are installed

### Debug Mode

Enable debug logging by adding to `src/services/api.js`:

```javascript
// Add this to see API calls in console
console.log('API Request:', config);
```

## 📄 License

This project is part of the EdTech Token Economy ML Platform.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues or questions:
- Check the main project README
- Review API documentation at `http://localhost:8000/docs`
- Check browser console for errors
