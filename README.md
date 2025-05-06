# StockMarketPredictor

A machine learning-powered web app that predicts stock prices and analyzes market sentiment.

📊 Key Features
Stock Price Prediction: Get next-day price forecasts for any publicly traded company

Sentiment Analysis: Evaluates market sentiment from recent news headlines

Automatic Model Training: Self-learning system trains new models when needed

Interactive Dashboard: Visualize predictions and model performance

🛠 How It Works
Data Collection: Uses Marketstack API to fetch historical stock data

Feature Engineering: Calculates 15+ technical indicators (SMA, EMA, RSI, MACD, etc.)

Machine Learning: Random Forest model trained on historical patterns

Sentiment Analysis: NLTK's VADER analyzes news headlines from Yahoo Finance

Web Interface: Flask backend with Bootstrap frontend for easy interaction

Successful Prediction (AAPL):
<img width="681" alt="Screenshot 2025-05-06 at 4 57 12 PM" src="https://github.com/user-attachments/assets/5d4e83eb-ae65-4ac5-aabb-b101d27f1458" />

AAPL Prediction
*The model successfully predicted AAPL's next-day closing price at $199.82*

Model Training Prompt (TSLA)
<img width="702" alt="Screenshot 2025-05-06 at 4 57 22 PM" src="https://github.com/user-attachments/assets/46fe3229-faeb-482e-a878-eada76d2c396" />

When no model exists (like for TSLA here), the system prompts to train one

⚙️ Setup Instructions
Clone this repository

Install requirements: pip install -r requirements.txt

Add your API keys to .env:

MARKETSTACK_API_KEY=your_key_here
Run: python app.py

🌟 Why This Project?
Combines fundamental and technical analysis with sentiment data

Production-ready with error handling and caching

Clean, responsive interface

Modular architecture for easy extension
