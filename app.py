from flask import Flask, render_template, request, redirect, url_for
from predict import StockPredictor
from sentiment import analyze_sentiment
import logging
import nltk
import os
from dotenv import load_dotenv
import warnings
from sklearn.exceptions import DataConversionWarning

# Suppress warnings
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Load environment variables
load_dotenv()  

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# NLTK Setup with persistent download
def initialize_nltk():
    nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
    os.makedirs(nltk_data_path, exist_ok=True)
    
    # Add to NLTK path if not already present
    if nltk_data_path not in nltk.data.path:
        nltk.data.path.append(nltk_data_path)
    
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')  # Check for compressed version
    except LookupError:
        try:
            nltk.data.find('sentiment/vader_lexicon')  # Check for uncompressed
        except LookupError:
            logger.info("Downloading VADER lexicon...")
            nltk.download('vader_lexicon', download_dir=nltk_data_path)
            logger.info("VADER lexicon download complete")

initialize_nltk()

# Initialize predictor
predictor = StockPredictor()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    sentiment = None
    error = None
    ticker = ""
    show_train_button = False
    training_success = request.args.get('trained')

    if request.method == "POST":
        try:
            ticker = request.form["ticker"].strip().upper()
            if not ticker:
                raise ValueError("Please enter a stock symbol")
            
            try:
                prediction = predictor.predict_stock_price(ticker)
                sentiment = analyze_sentiment(ticker)
            except ValueError as e:
                if "No model available" in str(e):
                    show_train_button = True
                    error = str(e)
                else:
                    raise
                
            if prediction is None:
                raise ValueError("Prediction generation failed")
                
        except ValueError as e:
            error = str(e)
            logger.error(f"Client error for {ticker}: {error}")
        except Exception as e:
            error = "System error. Please try again later."
            logger.error(f"System error for {ticker}: {str(e)}")

    return render_template(
        "index.html",
        prediction=prediction,
        sentiment=sentiment,
        ticker=ticker,
        error=error,
        show_train_button=show_train_button,
        training_success=training_success
    )

@app.route("/train/<ticker>", methods=["POST"])
def train_model(ticker):
    try:
        from train_model import train_model as train_stock_model
        logger.info(f"Starting training process for {ticker}")
        success = train_stock_model(ticker)
        if success:
            logger.info(f"Successfully trained model for {ticker}")
            return redirect(url_for('index', trained=ticker))
        else:
            logger.error(f"Training failed for {ticker}")
            return render_template(
                "error.html",
                message=f"Training failed for {ticker}",
                details="Please check the API key and try again later"
            ), 500
    except Exception as e:
        logger.error(f"Training error for {ticker}: {str(e)}")
        return render_template(
            "error.html",
            message=f"Training error for {ticker}",
            details=str(e)
        ), 500

if __name__ == "__main__":
    app.run(debug=True)