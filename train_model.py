import pandas as pd
import numpy as np
import joblib
import os
import sys
import time
import logging
from datetime import datetime, timedelta
import pickle
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Technical Indicators Calculator
class TechnicalIndicators:
    @staticmethod
    def calculate_sma(series, window):
        return series.rolling(window=window).mean()

    @staticmethod
    def calculate_ema(series, window):
        return series.ewm(span=window, adjust=False).mean()

    @staticmethod
    def calculate_rsi(series, window=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(series, fast=12, slow=26, signal=9):
        ema_fast = TechnicalIndicators.calculate_ema(series, fast)
        ema_slow = TechnicalIndicators.calculate_ema(series, slow)
        macd = ema_fast - ema_slow
        signal_line = TechnicalIndicators.calculate_ema(macd, signal)
        return macd, signal_line

    @staticmethod
    def calculate_atr(high, low, close, window=14):
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close.shift()),
            'lc': abs(low - close.shift())
        }).max(axis=1)
        return tr.rolling(window=window).mean()

    @staticmethod
    def calculate_bollinger_bands(series, window=20, num_std=2):
        sma = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        upper = sma + (rolling_std * num_std)
        lower = sma - (rolling_std * num_std)
        return upper, sma, lower

class MarketstackProvider:
    def __init__(self):
        self.api_key = os.getenv('MARKETSTACK_API_KEY')
        if not self.api_key:
            raise ValueError("MARKETSTACK_API_KEY environment variable not set")
        self.base_url = "http://api.marketstack.com/v1/"
        self.cache_dir = "data_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.max_retries = 3
        self.min_delay = 2  # Minimum delay between retries (seconds)
        self.max_delay = 5   # Maximum delay between retries (seconds)

    def _make_request(self, endpoint, params=None):
        """Make API request with retries and delays"""
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
        """Get stock data with caching"""
        cache_file = os.path.join(self.cache_dir, f"{ticker}.pkl")
        
        # Try cache first
        try:
            if os.path.exists(cache_file):
                cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                if datetime.now() - cache_time < timedelta(days=1):
                    with open(cache_file, "rb") as f:
                        return pickle.load(f)
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")

        # Fetch fresh data
        try:
            data = self._make_request("eod", {
                "symbols": ticker,
                "interval": "1day",  # Correct Marketstack parameter
                "sort": "ASC",
                "limit": 1000  # Max allowed by Marketstack
            })
            
            if not data.get("data"):
                raise ValueError("No data returned from Marketstack")

            # Convert to pandas DataFrame
            df = pd.DataFrame(data["data"])
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            
            # Rename columns to match expected format
            df = df.rename(columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume"
            })
            
            # Save to cache
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(df, f)
            except Exception as e:
                logger.warning(f"Cache write failed: {e}")
            
            return df[["Open", "High", "Low", "Close", "Volume"]]
        except Exception as e:
            logger.error(f"Failed to fetch data: {str(e)}")
            raise ValueError(f"Failed to fetch data: {str(e)}")

def add_technical_indicators(df):
    """Add technical indicators to dataframe"""
    ti = TechnicalIndicators()
    
    # Moving Averages
    df['SMA_20'] = ti.calculate_sma(df['Close'], 20)
    df['SMA_50'] = ti.calculate_sma(df['Close'], 50)
    df['EMA_20'] = ti.calculate_ema(df['Close'], 20)
    
    # Momentum Indicators
    df['RSI'] = ti.calculate_rsi(df['Close'])
    df['MACD'], df['MACD_Signal'] = ti.calculate_macd(df['Close'])
    
    # Volatility
    df['Upper_Band'], df['Middle_Band'], df['Lower_Band'] = ti.calculate_bollinger_bands(df['Close'])
    df['ATR'] = ti.calculate_atr(df['High'], df['Low'], df['Close'])
    
    # Volume Indicators
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
    
    return df.dropna()

def prepare_data(df):
    """Prepare features and target"""
    df["Date"] = df.index
    df["Target"] = df["Close"].shift(-1)
    return df.dropna()

def train_model(ticker):
    """Main training function"""
    try:
        logger.info(f"\n{'='*50}\nStarting training for {ticker}\n{'='*50}")
        
        provider = MarketstackProvider()
        df = provider.get_stock_data(ticker)
        df = add_technical_indicators(df)
        df = prepare_data(df)
        
        # Select features including technical indicators
        feature_columns = [
            'Open', 'High', 'Low', 'Volume',
            'SMA_20', 'SMA_50', 'EMA_20',
            'RSI', 'MACD', 'MACD_Signal',
            'Upper_Band', 'Middle_Band', 'Lower_Band',
            'ATR', 'OBV'
        ]
        
        X = df[feature_columns]
        y = df["Target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        save_artifacts(ticker, model, scaler)
        
        score = model.score(X_test, y_test)
        logger.info(f"Training successful. R² score: {score:.4f}")
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False

def save_artifacts(ticker, model, scaler):
    """Save model and scaler with versioning"""
    os.makedirs("model", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_file = f"model/{ticker}_model_{timestamp}.pkl"
    scaler_file = f"model/{ticker}_scaler_{timestamp}.pkl"
    
    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)
    
    # Create symlinks
    for src, dest in [
        (model_file, f"model/{ticker}_model.pkl"),
        (scaler_file, f"model/{ticker}_scaler.pkl")
    ]:
        if os.path.exists(dest):
            os.remove(dest)
        os.symlink(os.path.basename(src), dest)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_model.py TICKER")
        sys.exit(1)
        
    ticker = sys.argv[1].upper()
    
    # Random initial delay
    delay = random.uniform(0, 30)
    logger.info(f"Initial random delay: {delay:.1f}s")
    time.sleep(delay)
    
    success = train_model(ticker)
    sys.exit(0 if success else 1)