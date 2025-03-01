import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend, must be set before importing plt
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import sys
import json
import logging
import yfinance as yf
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import platform

# Import the improved stock predictor class
from stock_predictor import StockPricePredictor

# Detect operating system
IS_MACOS = platform.system() == 'Darwin'

# Configure for different environments
if os.environ.get('FLASK_ENV') == 'production':
    # Production settings
    DEBUG = False
    MODEL_PATH = os.environ.get('STOCK_MODEL_PATH', '/app/models')
    LOG_FILE = os.environ.get('LOG_FILE', '/app/logs/chatbot.log')
else:
    # Development settings
    DEBUG = True
    MODEL_PATH = os.environ.get('STOCK_MODEL_PATH', 'models')
    LOG_FILE = os.environ.get('LOG_FILE', 'chatbot.log')

# Make sure log directory exists
os.makedirs(os.path.dirname(LOG_FILE) if '/' in LOG_FILE else '.', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger("StockChatbot")

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # Enable CORS for all routes

# Initialize the improved stock predictor with optimal parameters
predictor = StockPricePredictor(
    base_path=MODEL_PATH,
    sequence_length=30,             # Reduced to 30 days to solve insufficient data points issue
    feature_engineering_level='advanced',  # Use advanced feature engineering
    risk_assessment=True,           # Enable risk assessment
    validation_method='time_series_split',  # Use time series cross validation
    confidence_interval=0.95        # Use 95% confidence interval
)

@app.route('/')
def home():
    """Render the home page"""
    # Check if index.html exists in templates folder, if not serve index_english.txt
    if os.path.exists(os.path.join('templates', 'index.html')):
        return render_template('index.html')
    elif os.path.exists('index_english.txt'):
        with open('index_english.txt', 'r') as f:
            content = f.read()
        return content, 200, {'Content-Type': 'text/html'}
    else:
        return "Error: No index template found", 404

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint to process stock prediction requests"""
    data = request.json
    ticker = data.get('ticker')
    
    if not ticker:
        return jsonify({'error': 'No stock ticker provided'}), 400
    
    try:
        # Get current stock price
        logger.info(f"Getting current price for {ticker}")
        current_data = yf.Ticker(ticker).history(period='1d')
        if current_data.empty:
            return jsonify({'error': f'Unable to get data for {ticker}'}), 404
        
        current_price = current_data['Close'].iloc[-1]
        logger.info(f"Current price for {ticker}: ${current_price:.2f}")
        
        # Check if model exists - now using model_manager
        logger.info(f"Checking if model exists for {ticker}")
        model, scaler = predictor.model_manager.load_model_files(ticker)
        
        if model is None:
            # Train new model if none exists
            logger.info(f"No model found for {ticker}, training new model...")
            # Decide whether to visualize based on operating system
            visualize = not IS_MACOS
            success = predictor.process_ticker(ticker, visualize=visualize)
            
            if not success:
                return jsonify({'error': f'Unable to train model for {ticker}'}), 500
            
            model, scaler = predictor.model_manager.load_model_files(ticker)
            if model is None:
                return jsonify({'error': f'Still unable to load model for {ticker} after training'}), 500
        
        # Get future predictions with confidence intervals - now using model_manager
        logger.info(f"Predicting future prices for {ticker}")
        future_df = predictor.model_manager.predict_future(ticker, days=30, include_confidence_intervals=True)
        
        if future_df is None:
            return jsonify({'error': f'Unable to predict future prices for {ticker}'}), 500
        
        # Convert DataFrame to dictionary suitable for JSON serialization
        prediction_dict = {}
        confidence_lower = {}
        confidence_upper = {}
        
        for date, row in future_df.iterrows():
            # Convert Timestamp to string
            date_str = date.strftime('%Y-%m-%d')
            prediction_dict[date_str] = row['Predicted_Close']
            
            # Check if confidence interval data exists
            if 'Lower_Bound' in future_df.columns and 'Upper_Bound' in future_df.columns:
                confidence_lower[date_str] = row['Lower_Bound']
                confidence_upper[date_str] = row['Upper_Bound']
        
        # Get metrics
        metrics_path = os.path.join(MODEL_PATH, ticker, f"{ticker}_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            logger.info(f"Loaded metrics for {ticker}: {metrics}")
        else:
            logger.warning(f"Metrics file not found for {ticker}")
            metrics = None
        
        # Directly use the risk_level from metrics (same as visualization.py)
        risk_level_english = metrics.get('risk_level', 'Unknown Risk') if metrics else 'Unknown Risk'
        
        # Define risk level mapping for advice generation
        risk_level_map = {
            "Low Risk": "Low Risk",
            "Medium-Low Risk": "Medium-Low Risk",
            "Medium Risk": "Medium Risk",
            "Medium-High Risk": "Medium-High Risk",
            "High Risk": "High Risk",
            "Unknown Risk": "Unknown Risk",
            "N/A": "Unknown Risk"
        }
        
        # Process chart image
        plot_img = None
        # Get plot image
        plot_path = os.path.join(MODEL_PATH, ticker, f"{ticker}_prediction_analysis.png")
        if not os.path.exists(plot_path):
            # Try old filename format
            plot_path = os.path.join(MODEL_PATH, ticker, f"{ticker}_prediction.png")
        
        if os.path.exists(plot_path):
            logger.info(f"Using existing prediction chart for {ticker}")
            try:
                with open(plot_path, 'rb') as f:
                    plot_img = base64.b64encode(f.read()).decode('utf-8')
            except Exception as e:
                logger.error(f"Error reading prediction chart for {ticker}: {e}")
                # Will continue with plot_img = None
        
        # If no existing chart found and not on macOS, try to generate a simple chart
        if plot_img is None and not IS_MACOS:
            logger.info(f"Generating simple prediction chart for {ticker}")
            try:
                buffer = BytesIO()
                plt.figure(figsize=(10, 6))
                
                # Get recent historical data for plotting
                hist_data = yf.Ticker(ticker).history(period="3mo")
                if not hist_data.empty:
                    # Plot historical prices
                    plt.plot(hist_data.index, hist_data['Close'], label='Historical Price', color='blue')
                
                # Plot predicted prices
                future_dates = [pd.Timestamp(date) for date in prediction_dict.keys()]
                future_prices = list(prediction_dict.values())
                plt.plot(future_dates, future_prices, label='Predicted Price', color='red', linestyle='--')
                
                # Plot confidence interval (if available)
                if confidence_lower and confidence_upper:
                    lower_bounds = list(confidence_lower.values())
                    upper_bounds = list(confidence_upper.values())
                    plt.fill_between(future_dates, lower_bounds, upper_bounds, 
                                    color='red', alpha=0.2, label='Prediction Confidence Interval')
                
                plt.title(f"{ticker} Stock Price Prediction")
                plt.xlabel('Date')
                plt.ylabel('Price ($)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                plot_img = base64.b64encode(buffer.read()).decode('utf-8')
                plt.close()
                
                # Also save to file
                plt.figure(figsize=(10, 6))
                if not hist_data.empty:
                    plt.plot(hist_data.index, hist_data['Close'], label='Historical Price', color='blue')
                plt.plot(future_dates, future_prices, label='Predicted Price', color='red', linestyle='--')
                
                if confidence_lower and confidence_upper:
                    plt.fill_between(future_dates, lower_bounds, upper_bounds, 
                                    color='red', alpha=0.2, label='Prediction Confidence Interval')
                
                plt.title(f"{ticker} Stock Price Prediction")
                plt.xlabel('Date')
                plt.ylabel('Price ($)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Ensure directory exists
                ticker_dir = os.path.join(MODEL_PATH, ticker)
                os.makedirs(ticker_dir, exist_ok=True)
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                logger.error(f"Error generating prediction chart for {ticker}: {e}")
                # Will continue with plot_img = None
                
        # Generate investment advice
        advice = generate_advice(ticker, risk_level_english, future_df)
        logger.info(f"Generated advice for {ticker}: {advice[:50]}...")
        
        # Get additional performance metrics
        additional_metrics = {}
        if metrics:
            if 'direction_accuracy' in metrics:
                additional_metrics['direction_accuracy'] = metrics['direction_accuracy']
            if 'relative_volatility' in metrics:
                additional_metrics['relative_volatility'] = metrics['relative_volatility']
            if 'risk_adjusted_return' in metrics:
                additional_metrics['risk_adjusted_return'] = metrics['risk_adjusted_return']
            if 'max_drawdown' in metrics:
                additional_metrics['max_drawdown'] = metrics['max_drawdown']
        
        # Prepare response
        response = {
            'ticker': ticker,
            'current_price': current_price,
            'prediction': prediction_dict,
            'confidence_lower': confidence_lower,
            'confidence_upper': confidence_upper,
            'risk_level': risk_level_english,  # English version directly from metrics
            'metrics': metrics,
            'additional_metrics': additional_metrics,
            'plot_img': plot_img,
            'advice': advice
        }
        
        logger.info(f"Successfully processed request for {ticker}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing request for {ticker}: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error processing request: {str(e)}'}), 500

def generate_advice(ticker, risk_level, future_df):
    """
    Generate investment advice
    Based on risk level, predicted trend, and confidence intervals
    
    Parameters:
        ticker (str): Stock ticker symbol
        risk_level (str): Risk level
        future_df (DataFrame): Prediction data
    """
    if future_df is None or future_df.empty:
        return "Unable to generate advice: Prediction data is empty"
    
    # Get prediction trend
    first_price = future_df['Predicted_Close'].iloc[0]
    last_price = future_df['Predicted_Close'].iloc[-1]
    price_change = (last_price - first_price) / first_price * 100
    
    # Calculate prediction uncertainty (if confidence intervals exist)
    uncertainty = "medium"
    if 'Lower_Bound' in future_df.columns and 'Upper_Bound' in future_df.columns:
        # Calculate confidence interval width on the last day as percentage of predicted price
        last_width = (future_df['Upper_Bound'].iloc[-1] - future_df['Lower_Bound'].iloc[-1]) / future_df['Predicted_Close'].iloc[-1] * 100
        if last_width < 5:
            uncertainty = "low"
        elif last_width > 15:
            uncertainty = "high"
    
    # Generate advice based on risk level and trend
    if risk_level == "Low Risk":
        if price_change > 5:
            return f"{ticker} has low risk and is predicted to increase by approximately {price_change:.2f}% over the next 30 days. Prediction uncertainty is {uncertainty}. Suitable for conservative investors to consider purchasing."
        elif price_change < -5:
            return f"{ticker} has low risk but is predicted to decrease by approximately {abs(price_change):.2f}% over the next 30 days. Prediction uncertainty is {uncertainty}. Recommend watching or considering a small investment."
        else:
            return f"{ticker} has low risk and its price is predicted to remain stable. Prediction uncertainty is {uncertainty}. Suitable for investors seeking stability."
    
    elif risk_level == "Medium-Low Risk":
        if price_change > 5:
            return f"{ticker} has medium-low risk and is predicted to increase by approximately {price_change:.2f}% over the next 30 days. Prediction uncertainty is {uncertainty}. Suitable for moderately conservative investors to allocate appropriately."
        elif price_change < -5:
            return f"{ticker} has medium-low risk but is predicted to decrease by approximately {abs(price_change):.2f}% over the next 30 days. Prediction uncertainty is {uncertainty}. Recommend watching or considering a smaller position."
        else:
            return f"{ticker} has medium-low risk and price fluctuations are expected to be minimal. Prediction uncertainty is {uncertainty}. Suitable for medium to long-term investors to consider."
    
    elif risk_level == "Medium Risk":
        if price_change > 8:
            return f"{ticker} has medium risk but is predicted to increase by approximately {price_change:.2f}% over the next 30 days. Prediction uncertainty is {uncertainty}. Suitable for balanced investors to consider purchasing with appropriate position control."
        elif price_change < -8:
            return f"{ticker} has medium risk and is predicted to decrease by approximately {abs(price_change):.2f}% over the next 30 days. Prediction uncertainty is {uncertainty}. Recommend careful consideration or watching."
        else:
            return f"{ticker} has medium risk and price fluctuations are expected to be minimal. Prediction uncertainty is {uncertainty}. Can be allocated moderately in a portfolio to diversify risk."
    
    elif risk_level == "Medium-High Risk":
        if price_change > 10:
            return f"{ticker} has medium-high risk and is predicted to increase by approximately {price_change:.2f}% over the next 30 days. Prediction uncertainty is {uncertainty}. Suitable for investors with strong risk tolerance to participate with small positions, paying attention to stop-loss."
        elif price_change < -10:
            return f"{ticker} has medium-high risk and is predicted to decrease by approximately {abs(price_change):.2f}% over the next 30 days. Prediction uncertainty is {uncertainty}. Recommend avoiding investment or considering hedging strategies."
        else:
            return f"{ticker} has medium-high risk and price changes are not expected to be significant. Prediction uncertainty is {uncertainty}. Recommend cautious participation with controlled position size."
    
    elif risk_level == "High Risk":
        if price_change > 10:
            return f"{ticker} has high risk but is predicted to increase significantly by approximately {price_change:.2f}% over the next 30 days. Prediction uncertainty is {uncertainty}. Suitable for investors with strong risk tolerance to try in small amounts, but please control risk."
        elif price_change < -10:
            return f"{ticker} has high risk and is predicted to decrease significantly by approximately {abs(price_change):.2f}% over the next 30 days. Prediction uncertainty is {uncertainty}. Recommend avoiding investment or considering hedging strategies."
        else:
            return f"{ticker} has high risk, and even though price changes are not expected to be significant, prediction uncertainty is {uncertainty}. Still recommend cautious operation, only suitable for investors with high risk tolerance to participate in small amounts."
    
    else:
        return f"Unable to determine the risk level for {ticker}, predicted to change by approximately {price_change:.2f}% over the next 30 days. Prediction uncertainty is {uncertainty}. Recommend conducting more research and consulting professional advice before investing."

@app.route('/api/tickers', methods=['GET'])
def get_available_tickers():
    """Return a list of tickers that have trained models"""
    try:
        tickers = []
        # Check if model directory exists
        if os.path.exists(MODEL_PATH):
            # Get all subdirectories (each should be a ticker)
            for item in os.listdir(MODEL_PATH):
                if os.path.isdir(os.path.join(MODEL_PATH, item)) and not item.startswith('.') and item != 'data_cache':
                    # Check if model file exists
                    if os.path.exists(os.path.join(MODEL_PATH, item, f"{item}_lstm_model.h5")):
                        tickers.append(item)
        
        return jsonify({'tickers': tickers})
    except Exception as e:
        logger.error(f"Error getting ticker list: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error getting ticker list: {str(e)}'}), 500

@app.route('/api/backtest', methods=['POST'])
def run_backtest():
    """API endpoint to run backtest on a stock"""
    data = request.json
    ticker = data.get('ticker')
    
    if not ticker:
        return jsonify({'error': 'No stock ticker provided'}), 400
    
    try:
        # Run backtest
        logger.info(f"Running backtest analysis for {ticker}")
        backtest_results = predictor.run_backtest(ticker, prediction_window=30)
        
        if backtest_results is None:
            return jsonify({'error': f'Unable to run backtest for {ticker}'}), 500
        
        # Convert DataFrame to dictionary
        backtest_dict = backtest_results.to_dict(orient='records')
        
        # Get backtest image (if exists)
        backtest_img = None
        backtest_path = os.path.join(MODEL_PATH, 'backtests', f'{ticker}_backtest_latest.png')
        if os.path.exists(backtest_path):
            logger.info(f"Using existing backtest chart for {ticker}")
            try:
                with open(backtest_path, 'rb') as f:
                    backtest_img = base64.b64encode(f.read()).decode('utf-8')
            except Exception as e:
                logger.error(f"Error reading backtest chart for {ticker}: {e}")
        
        # Calculate backtest performance metrics
        direction_accuracy = backtest_results['Direction Correct'].mean() * 100
        mean_error = backtest_results['Prediction Error (%)'].mean()
        mean_abs_error = backtest_results['Prediction Error (%)'].abs().mean()
        
        # Prepare backtest summary
        backtest_summary = {
            'direction_accuracy': float(direction_accuracy),
            'mean_error': float(mean_error),
            'mean_abs_error': float(mean_abs_error),
            'total_tests': len(backtest_results)
        }
        
        response = {
            'ticker': ticker,
            'backtest_results': backtest_dict,
            'backtest_summary': backtest_summary,
            'backtest_img': backtest_img
        }
        
        logger.info(f"Successfully completed backtest analysis for {ticker}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error running backtest for {ticker}: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error running backtest: {str(e)}'}), 500

@app.route('/api/compare', methods=['POST'])
def compare_stocks():
    """API endpoint to compare multiple stocks"""
    data = request.json
    tickers = data.get('tickers', [])
    
    if not tickers or not isinstance(tickers, list) or len(tickers) == 0:
        return jsonify({'error': 'No valid stock ticker list provided'}), 400
    
    try:
        # Limit number of stocks to compare at once
        if len(tickers) > 10:
            tickers = tickers[:10]
            logger.warning(f"Number of stocks exceeds 10, limited to first 10: {tickers}")
        
        # Compare stocks
        logger.info(f"Comparing stocks: {tickers}")
        comparison_results = predictor.compare_stocks(tickers, days_to_predict=30, visualize=True)
        
        if comparison_results is None or len(comparison_results) == 0:
            return jsonify({'error': 'Unable to compare specified stocks'}), 500
        
        # Convert DataFrame to dictionary
        comparison_dict = comparison_results.to_dict(orient='records')
        
        # Get comparison image (if exists)
        comparison_img = None
        # Get most recent comparison image
        comparison_dir = os.path.join(MODEL_PATH, 'comparisons')
        if os.path.exists(comparison_dir):
            # Get the most recent image
            image_files = [f for f in os.listdir(comparison_dir) if f.startswith('stock_comparison_')]
            if image_files:
                # Sort by modification time
                latest_image = max(image_files, key=lambda x: os.path.getmtime(os.path.join(comparison_dir, x)))
                try:
                    with open(os.path.join(comparison_dir, latest_image), 'rb') as f:
                        comparison_img = base64.b64encode(f.read()).decode('utf-8')
                except Exception as e:
                    logger.error(f"Error reading comparison chart: {e}")
        
        # Calculate average expected return
        avg_return = comparison_results['Expected Return (%)'].mean()
        
        response = {
            'tickers': tickers,
            'comparison_results': comparison_dict,
            'average_return': float(avg_return),
            'comparison_img': comparison_img
        }
        
        logger.info(f"Successfully completed stock comparison analysis")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error comparing stocks: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error comparing stocks: {str(e)}'}), 500

if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5006))
    app.run(host='0.0.0.0', port=port, debug=DEBUG)
