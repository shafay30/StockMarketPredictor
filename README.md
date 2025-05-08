# 📈 Stock Market Predictor

*A machine learning web app that forecasts stock prices and analyzes market sentiment*

Successful Prediction (AAPL):
<img width="681" alt="Screenshot 2025-05-06 at 4 57 12 PM" src="https://github.com/user-attachments/assets/5d4e83eb-ae65-4ac5-aabb-b101d27f1458" />

AAPL Prediction
*The model successfully predicted AAPL's next-day closing price at $199.82*

Model Training Prompt (TSLA)
<img width="702" alt="Screenshot 2025-05-06 at 4 57 22 PM" src="https://github.com/user-attachments/assets/46fe3229-faeb-482e-a878-eada76d2c396" />

When no model exists (like for TSLA here), the system prompts to train one

---

## ✨ Features

### 📊 Prediction Engine
- Next-day price forecasts using Random Forest ML model
- Automatic model training for new stocks
- 15+ technical indicators (SMA, RSI, MACD, Bollinger Bands)

### 📰 Sentiment Analysis
- Real-time news headline processing
- VADER sentiment scoring (Positive/Neutral/Negative)
- Yahoo Finance integration

### 🖥️ Web Interface
- Clean Bootstrap frontend
- Interactive dashboard
- Model training prompts

---

## 🛠️ How It Works

1. **Data Pipeline**  
   - Fetches historical data via Marketstack API
   - Calculates technical indicators
   - Caches results for performance

2. **Machine Learning**  
   - Random Forest regression model
   - Automated training for new tickers
   - StandardScaler for feature normalization

3. **Web App**  
   - Flask backend API
   - Responsive Bootstrap UI
   - Error handling and user prompts

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Marketstack API key

### Installation
```bash
git clone https://github.com/shafay30/StockMarketPredictor.git
cd StockMarketPredictor
pip install -r requirements.txt
