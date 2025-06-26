# 🌾 Agrosarthi Backend API

## Overview

**Agrosarthi** is a comprehensive agricultural assistance platform that leverages AI and machine learning to provide farmers with intelligent crop recommendations, disease detection, weather forecasting, and agricultural guidance. The backend is built with FastAPI and deployed on Google Cloud Run.



## 🚀 Features

### 🤖 AI-Powered Services
- **Crop Prediction**: ML-based crop recommendations using soil and weather parameters
- **Disease Detection**: Image-based plant disease identification using Google Gemini Vision API
- **Price Estimation**: Market price prediction for agricultural commodities
- **Yield Prediction**: Crop yield forecasting based on historical data
- **Agricultural Chatbot**: Multi-language AI assistant for farming queries

### 🌤️ Weather & Environmental
- **7-Day Weather Forecast**: Location-based weather predictions
- **Soil Parameter Estimation**: NPK and pH estimation using geographical data
- **Climate Analysis**: Weather pattern analysis for agricultural planning

### 🌍 Multi-Language Support
- English
- Hindi (हिंदी)
- Marathi (मराठी)

### 📊 Data Analytics
- Historical weather data integration using Meteostat
- Market price analysis for Maharashtra region
- Comprehensive crop planning with detailed agricultural guidance

## 🛠️ Technology Stack

- **Framework**: FastAPI
- **AI/ML**: TensorFlow, Google Gemini API
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Weather Data**: WeatherAPI, Meteostat
- **Image Processing**: PIL (Pillow)
- **Deployment**: Google Cloud Run (Dockerized)
- **CORS**: Configured for cross-origin requests

## 📋 API Endpoints

### Core Prediction Services

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict/` | POST | Crop prediction based on soil parameters |
| `/predict-price/` | POST | Agricultural commodity price estimation |
| `/predict-yield/` | POST | Crop yield prediction |
| `/predict-disease/` | POST | Plant disease detection from images |

### Information Services

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query/` | POST | AI agricultural chatbot queries |
| `/weather-forecast` | GET | 7-day weather forecast |
| `/soil-estimate` | GET | Soil parameter estimation |
| `/generate-crop-plan` | POST | Detailed crop cultivation plan |

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- Google Cloud SDK (for deployment)
- Docker (for containerization)

### Local Development

1. **Clone the repository**
```bash
git clone <repository-url>
cd agrosarthi-backend
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
export WEATHER_API_KEY="your_weather_api_key"
export GEMINI_API_KEY="your_gemini_api_key"
```

4. **Run the application**
```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

5. **Access the API documentation**
```
http://localhost:8080/docs
```

### Docker Deployment

1. **Build the Docker image**
```bash
docker build -t agrosarthi-backend .
```

2. **Run the container**
```bash
docker run -p 8080:8080 -e WEATHER_API_KEY="your_key" -e GEMINI_API_KEY="your_key" agrosarthi-backend
```

### Google Cloud Run Deployment

1. **Build and push to Google Container Registry**
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/agrosarthi-backend
```

2. **Deploy to Cloud Run**
```bash
gcloud run deploy agrosarthi-backend \
  --image gcr.io/PROJECT_ID/agrosarthi-backend \
  --platform managed \
  --region asia-south1 \
  --allow-unauthenticated
```

## 📊 Machine Learning Models

### Crop Prediction Model
- **Type**: Neural Network (TensorFlow/Keras)
- **Input Features**: N, P, K, Temperature, Humidity, pH, Rainfall
- **Output**: Top 3 recommended crops with confidence scores
- **Model File**: `model/crop_suggestion_model.h5`

### Price Estimation Model
- **Type**: Regression Model (Scikit-learn)
- **Features**: District, Month, Market, Commodity, Variety, Season
- **Coverage**: Maharashtra state markets
- **Model File**: `model/crop_price_model.pkl`

### Yield Prediction Model
- **Type**: Regression Model (Scikit-learn)
- **Features**: State, District, Commodity, Season, Area
- **Model File**: `model/yield_prediction_model.pkl`

## 🌐 API Usage Examples

### Crop Prediction
```python
import requests

url = "https://agrosarthi-backend-885337506715.asia-south1.run.app/predict/"
data = {
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 20.8,
    "humidity": 82,
    "ph": 6.5,
    "rainfall": 202.9,
    "language": "English"
}

response = requests.post(url, json=data)
print(response.json())
```

### Disease Detection
```python
import requests

url = "https://agrosarthi-backend-885337506715.asia-south1.run.app/predict-disease/"
files = {"file": open("plant_image.jpg", "rb")}
data = {"crop_name": "tomato", "language": "English"}

response = requests.post(url, files=files, data=data)
print(response.json())
```

### Weather Forecast
```python
import requests

url = "https://agrosarthi-backend-885337506715.asia-south1.run.app/weather-forecast"
params = {"lat": 18.5204, "lon": 73.8567, "language": "English"}

response = requests.get(url, params=params)
print(response.json())
```

## 🌍 Supported Regions

- **Primary Focus**: Maharashtra, India
- **Weather Data**: Global coverage
- **Soil Data**: Comprehensive village-level data for Maharashtra
- **Market Data**: Maharashtra agricultural markets

## 🔒 Security Features

- **CORS Configuration**: Secure cross-origin resource sharing
- **Input Validation**: Pydantic models for request validation
- **Error Handling**: Comprehensive exception handling
- **API Key Management**: Secure environment variable handling

## 📈 Performance Optimization

- **Async Operations**: FastAPI async support for better performance
- **Model Caching**: Pre-loaded ML models for faster predictions
- **Efficient Data Processing**: Optimized pandas operations
- **Response Compression**: Automatic response compression

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google Gemini API** for AI-powered agricultural assistance
- **WeatherAPI** for weather data services
- **Meteostat** for historical weather data
- **TensorFlow** for machine learning capabilities
- **FastAPI** for the robust web framework

## 📞 Support

For support, email support@agrosarthi.com or create an issue on GitHub.

---

**Made with ❤️ for farmers and agricultural communities**
