import joblib
import numpy as np
import pandas as pd
import requests
import os
import logging
from datetime import datetime, timedelta
import time
import random
from dotenv import load_dotenv
import warnings
from sklearn.exceptions import DataConversionWarning

# Suppress warnings
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketstackProvider:
    def __init__(self):
        self.api_key = os.getenv('MARKETSTACK_API_KEY')
        if not self.api_key:
            raise ValueError("MARKETSTACK_API_KEY environment variable not set")
        self.base_url = "http://api.marketstack.com/v1/"
        self.max_retries = 3
        self.min_delay = 1
        self.max_delay = 5
        self.min_data_points = 50  # Minimum data points needed for indicators
        self.cache_dir = "data_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

    def _make_request(self, endpoint, params=None):
        params = params or {}
        params.update({"access_key": self.api_key})

        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    f"{self.base_url}{endpoint}",
                    params=params,
                    timeout=10
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = random.uniform(self.min_delay, self.max_delay)
                    logger.warning(f"Attempt {attempt + 1} failed. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    raise Exception(f"API request failed: {str(e)}")

    def get_stock_data(self, ticker):
        cache_file = os.path.join(self.cache_dir, f"{ticker}.pkl")
        
        # Try cache first
        try:
            if os.path.exists(cache_file):
                cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                if datetime.now() - cache_time < timedelta(hours=1):  # Cache for 1 hour
                    with open(cache_file, "rb") as f:
                        return pd.read_pickle(f)
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")

        try:
            # Get more data points for technical indicators
            data = self._make_request("eod", {
                "symbols": ticker,
                "limit": self.min_data_points,
                "sort": "DESC"
            })
            
            if not data.get("data"):
                raise ValueError("No data returned from Marketstack")
            
            df = pd.DataFrame(data["data"])
            if len(df) < 20:  # Minimum needed for some indicators
                raise ValueError(f"Insufficient data points ({len(df)}). Need at least 20.")
            
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            df = df.sort_index()  # Ensure chronological order
            
            df = df.rename(columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume"
            })
            
            # Save to cache
            try:
                df.to_pickle(cache_file)
            except Exception as e:
                logger.warning(f"Failed to cache data: {e}")
            
            return df[["Open", "High", "Low", "Close", "Volume"]]
        except Exception as e:
            raise ValueError(f"Failed to fetch data: {str(e)}")

class StockPredictor:
    def __init__(self):
        self.model_dir = "model"
        os.makedirs(self.model_dir, exist_ok=True)
        self.max_retries = 3
        self.initial_delay = 5
        self.data_provider = MarketstackProvider()

    def get_model_files(self, ticker):
        model_file = f"{self.model_dir}/{ticker}_model.pkl"
        scaler_file = f"{self.model_dir}/{ticker}_scaler.pkl"
        
        if os.path.exists(model_file) and os.path.exists(scaler_file):
            return model_file, scaler_file
            
        model_files = sorted(
            [f for f in os.listdir(self.model_dir) 
             if f.startswith(f"{ticker}_model_")],
            reverse=True
        )
        
        if model_files:
            latest_model = os.path.join(self.model_dir, model_files[0])
            latest_scaler = latest_model.replace("_model_", "_scaler_")
            return latest_model, latest_scaler
            
        return None, None

    def predict_stock_price(self, ticker, auto_train=False):
        try:
            logger.info(f"Starting prediction for {ticker}")
            
            model_file, scaler_file = self.get_model_files(ticker)
            if not model_file or not scaler_file:
                if auto_train:
                    logger.info(f"No model found for {ticker}, attempting to train...")
                    from train_model import train_model as train_stock_model
                    if train_stock_model(ticker):
                        model_file, scaler_file = self.get_model_files(ticker)
                    else:
                        raise ValueError("Auto-training failed")
                else:
                    raise FileNotFoundError(f"No model available for {ticker}. Please train first.")
            
            model = self._load_with_retries(model_file, "model")
            scaler = self._load_with_retries(scaler_file, "scaler")
            
            # Get and prepare data with all features
            df = self.data_provider.get_stock_data(ticker)
            df = self._calculate_technical_indicators(df)
            
            # Prepare input with ALL 15 features (same order as training)
            latest_data = np.array([[
                df["Open"].iloc[-1],
                df["High"].iloc[-1],
                df["Low"].iloc[-1],
                df["Volume"].iloc[-1],
                df["SMA_20"].iloc[-1],
                df["SMA_50"].iloc[-1],
                df["EMA_20"].iloc[-1],
                df["RSI"].iloc[-1],
                df["MACD"].iloc[-1],
                df["MACD_Signal"].iloc[-1],
                df["Upper_Band"].iloc[-1],
                df["Middle_Band"].iloc[-1],
                df["Lower_Band"].iloc[-1],
                df["ATR"].iloc[-1],
                df["OBV"].iloc[-1]
            ]])
            
            scaled_data = scaler.transform(latest_data)
            prediction = model.predict(scaled_data)[0]
            
            logger.info(f"Successful prediction for {ticker}: {prediction:.2f}")
            return round(prediction, 2)
            
        except FileNotFoundError as e:
            logger.error(str(e))
            raise ValueError(f"No model available for {ticker}. Please train first.")
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise ValueError(f"Prediction error: {str(e)}")

    def _calculate_technical_indicators(self, df):
        """Calculate all technical indicators needed for prediction"""
        # Replicate the same indicator calculations as in train_model.py
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        
        # RSI calculation
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain/loss)))
        
        # MACD calculation
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['Middle_Band'] = df['Close'].rolling(window=20).mean()
        rolling_std = df['Close'].rolling(window=20).std()
        df['Upper_Band'] = df['Middle_Band'] + (rolling_std * 2)
        df['Lower_Band'] = df['Middle_Band'] - (rolling_std * 2)
        
        # ATR
        tr = pd.DataFrame({
            'hl': df['High'] - df['Low'],
            'hc': abs(df['High'] - df['Close'].shift()),
            'lc': abs(df['Low'] - df['Close'].shift())
        }).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()
        
        # OBV
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
        
        return df.dropna()

    def _load_with_retries(self, file_path, file_type):
        for attempt in range(self.max_retries):
            try:
                return joblib.load(file_path)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = random.uniform(1, self.initial_delay)
                    logger.warning(f"{file_type} load attempt {attempt+1} failed. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    raise