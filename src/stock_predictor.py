#from stock_predictor import StockPricePredictor, visualize_stock_prediction, analyze_portfolio
import numpy as np
import pandas as pd
import os
import logging
import datetime
import warnings
import time
import pickle
from tqdm import tqdm

# Import the component modules
from data_utils import DataProcessor
from model_utils import ModelManager
from visualization import Visualizer

# Set up Seaborn style to improve chart quality
import seaborn as sns
sns.set(style="whitegrid")

# Suppress YFinance warning messages
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging system
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("stock_prediction.log"), logging.StreamHandler()]
)
logger = logging.getLogger("StockPredictor")

class StockPricePredictor:
    """
    Enhanced stock price prediction system using deep learning models
    
    Main features:
    - Multi-layer deep learning architecture (LSTM + GRU) for capturing time series patterns
    - Adaptive data preparation and feature engineering
    - Robust model training and evaluation mechanisms
    - Risk assessment and prediction confidence intervals
    - Detailed visualization and results analysis
    """
    
    def __init__(self, base_path=None, sequence_length=30, train_split=0.8, 
                 feature_engineering_level='advanced', risk_assessment=True,
                 ensemble_models=False, validation_method='time_series_split', 
                 confidence_interval=0.95):
        """
        Initialize stock price predictor
        
        Parameters:
            base_path (str, optional): Model and data storage path
            sequence_length (int, default=30): 
                Historical sequence length for prediction
            train_split (float, default=0.8): 
                Train/test split ratio
            feature_engineering_level (str, default='advanced'): 
                Feature engineering complexity ('basic', 'intermediate', 'advanced')
            risk_assessment (bool, default=True): 
                Whether to perform risk assessment and volatility analysis
            ensemble_models (bool, default=False): 
                Whether to use model ensemble for improved prediction stability
            validation_method (str, default='time_series_split'): 
                Validation method ('simple_split', 'time_series_split', 'walk_forward')
            confidence_interval (float, default=0.95): 
                Prediction confidence interval
        """
        # Data storage paths setup
        if base_path is None:
            self.base_path = os.environ.get('STOCK_MODEL_PATH', 'models')
        else:
            self.base_path = base_path
        
        os.makedirs(self.base_path, exist_ok=True)
        self.data_cache_dir = os.path.join(self.base_path, 'data_cache')
        os.makedirs(self.data_cache_dir, exist_ok=True)
        
        # Core parameters setup
        self.sequence_length = sequence_length
        self.train_split = train_split
        self.feature_engineering_level = feature_engineering_level
        self.risk_assessment = risk_assessment
        self.ensemble_models = ensemble_models
        self.validation_method = validation_method
        self.confidence_interval = confidence_interval
        
        # Model and scaler storage
        self.scaler_dict = {}
        self.models_dict = {}
        self.feature_importance = {}
        
        # Initialize component modules
        self.data_processor = DataProcessor(self)
        self.model_manager = ModelManager(self)
        self.visualizer = Visualizer(self)
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        logger.info(f"Initialized stock prediction system, models will be saved to: {os.path.abspath(self.base_path)}")
        logger.info(f"Model configuration: sequence_length={sequence_length}, train_split={train_split}, " 
                   f"feature_engineering_level={feature_engineering_level}, risk_assessment={risk_assessment}")
    
    def process_ticker(self, ticker, visualize=True):
        """
        Process complete workflow for a single stock
        
        Parameters:
            ticker (str): Stock ticker symbol
            visualize (bool): Whether to generate visualization results
            
        Returns:
            bool: Whether processing was successful
        """
        logger.info(f"====== Starting processing for {ticker} ======")
        
        try:
            # 1. Check for cached data
            cache_file = os.path.join(self.data_cache_dir, f"{ticker}_processed_data.pkl")
            df = None
            
            if os.path.exists(cache_file):
                logger.info(f"Found cached data for {ticker}")
                try:
                    with open(cache_file, 'rb') as f:
                        df = pickle.load(f)
                    logger.info(f"Loaded {ticker} data from cache, {len(df)} data points")
                except Exception as e:
                    logger.warning(f"Error reading cached data for {ticker}: {e}")
            
            # 2. If no valid cached data, get and process new data
            if df is None or len(df) < self.sequence_length + 20:
                # Get stock data
                df = self.data_processor.get_stock_data(ticker, period="1y")
                if df is None or df.empty:
                    logger.error(f"Cannot process {ticker}: Unable to get stock data")
                    return False
                
                # Add technical indicators
                df = self.data_processor.add_technical_indicators(df)
                if df is None or df.empty:
                    logger.error(f"Cannot process {ticker}: Failed to add technical indicators")
                    return False
                
                # Save processed data
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(df, f)
                    logger.info(f"Cached processed data for {ticker} to: {cache_file}")
                except Exception as e:
                    logger.warning(f"Error caching processed data for {ticker}: {e}")
            
            # Ensure enough data
            if len(df) < self.sequence_length + 20:
                logger.error(f"Cannot process {ticker}: Too few data points ({len(df)})")
                return False
            
            # 3. Analyze feature importance
            importance_df = self.data_processor.analyze_feature_importance(df)
            
            # 4. Prepare data
            prepare_result = self.data_processor.prepare_data(df)
            if prepare_result[0] is None:
                logger.error(f"Cannot process {ticker}: Failed to prepare data")
                return False
                
            X_train, y_train, X_test, y_test, scaler, (train_dates, test_dates) = prepare_result
            
            # 5. Train model
            train_result = self.model_manager.train_model(ticker, X_train, y_train, X_test, y_test)
            if train_result[0] is None:
                logger.error(f"Cannot process {ticker}: Failed to train model")
                return False
                
            model, history = train_result
            
            # 6. Evaluate model
            eval_result = self.model_manager.evaluate_model(model, X_test, y_test, scaler, df)
            if eval_result[0] is None:
                logger.error(f"Cannot process {ticker}: Failed to evaluate model")
                return False
                
            y_pred, y_test_inv, metrics = eval_result
            
            # 7. Save model files
            saved = self.model_manager.save_model_files(ticker, model, scaler, metrics, importance_df)
            if not saved:
                logger.warning(f"{ticker}: Failed to save model files, but continuing processing")
            
            # 8. Predict future prices
            try:
                future_df = self.model_manager.predict_future(ticker, days=30)
            except Exception as e:
                logger.warning(f"Error predicting future prices for {ticker}: {e}, but continuing processing")
                future_df = None
            
            # 9. Visualize results
            if visualize:
                try:
                    self.visualizer.visualize_results(ticker, test_dates, y_test_inv, y_pred, metrics, history, future_df)
                except Exception as e:
                    logger.warning(f"Error generating visualization for {ticker}: {e}")
            
            # Add model and scaler to dictionaries
            self.models_dict[ticker] = model
            self.scaler_dict[ticker] = scaler
            
            logger.info(f"====== Processing complete for {ticker} ======")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def process_multiple_tickers(self, tickers=None, limit=5, visualize=False, parallel=False):
        """
        Process multiple stocks
        
        Parameters:
            tickers (list, optional): List of stock ticker symbols
            limit (int): If tickers is None, limit the number of stocks to process
            visualize (bool): Whether to generate visualization results
            parallel (bool): Whether to process in parallel
            
        Returns:
            tuple: (successful_tickers, failed_tickers) Lists of successful and failed tickers
        """
        if tickers is None:
            # If no ticker list provided, get S&P 500 stocks
            tickers = self.data_processor.get_sp500_tickers(limit=limit)
        
        successful_tickers = []
        failed_tickers = []
        
        logger.info(f"Starting processing for {len(tickers)} stocks...")
        
        # Use tqdm to create progress bar
        for i, ticker in enumerate(tqdm(tickers, desc="Processing stocks")):
            logger.info(f"Processing stock {i+1}/{len(tickers)}: {ticker}")
            
            # Add delay to avoid API rate limits
            if i > 0 and not parallel:
                time.sleep(1)
            
            # Process stock
            success = self.process_ticker(ticker, visualize=visualize)
            
            if success:
                successful_tickers.append(ticker)
            else:
                failed_tickers.append(ticker)
        
        logger.info(f"Stock processing complete. Success: {len(successful_tickers)}, Failed: {len(failed_tickers)}")
        if failed_tickers:
            logger.info(f"Failed stocks: {failed_tickers}")
        
        return successful_tickers, failed_tickers
    
    def get_model_summary(self):
        """
        Get summary of all trained models
        
        Returns:
            DataFrame: Model summary
        """
        model_dirs = [d for d in os.listdir(self.base_path) 
                     if os.path.isdir(os.path.join(self.base_path, d)) and 
                     not d.startswith('.') and d != 'data_cache']
        
        summary = []
        for ticker in model_dirs:
            metrics_path = os.path.join(self.base_path, ticker, f"{ticker}_metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    import json
                    metrics = json.load(f)
                    
                    # Get last modified time
                    last_modified = datetime.datetime.fromtimestamp(
                        os.path.getmtime(metrics_path)).strftime('%Y-%m-%d')
                    
                    summary.append({
                        'Ticker': ticker,
                        'RMSE': round(metrics.get('rmse', 0), 4),
                        'MAPE (%)': round(metrics.get('mape', 0), 2),
                        'Direction Accuracy (%)': round(metrics.get('direction_accuracy', 0), 2),
                        'Risk Level': metrics.get('risk_level', 'N/A'),
                        'Relative Volatility': round(metrics.get('relative_volatility', 0), 4),
                        'Risk Adjusted Return': round(metrics.get('risk_adjusted_return', 0), 4),
                        'Max Drawdown (%)': round(metrics.get('max_drawdown', 0), 2),
                        'Last Updated': last_modified
                    })
        
        if not summary:
            logger.warning("No model metrics found")
            return pd.DataFrame()
            
        # Create DataFrame and sort
        summary_df = pd.DataFrame(summary)
        summary_df = summary_df.sort_values('MAPE (%)')
        
        return summary_df
    
    def compare_stocks(self, tickers, days_to_predict=30, visualize=True):
        """
        Compare prediction results for multiple stocks
        
        Parameters:
            tickers (list): List of stock ticker symbols
            days_to_predict (int): Number of days to predict
            visualize (bool): Whether to generate visualization results
            
        Returns:
            DataFrame: Comparison results
        """
        logger.info(f"Comparing prediction results for {len(tickers)} stocks...")
        
        results = []
        future_predictions = {}
        
        for ticker in tickers:
            try:
                # Check if model exists
                model_dir = os.path.join(self.base_path, ticker)
                if not os.path.exists(model_dir):
                    logger.warning(f"Model for {ticker} does not exist, attempting to train...")
                    success = self.process_ticker(ticker, visualize=False)
                    if not success:
                        logger.error(f"Cannot process {ticker}, skipping")
                        continue
                
                # Get model metrics
                metrics_path = os.path.join(model_dir, f"{ticker}_metrics.json")
                metrics = {}
                
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        import json
                        metrics = json.load(f)
                
                # Predict future prices
                future_df = self.model_manager.predict_future(ticker, days=days_to_predict)
                if future_df is None:
                    logger.warning(f"Cannot predict future prices for {ticker}, skipping")
                    continue
                
                # Calculate expected return
                last_price = future_df.index[0] - datetime.timedelta(days=1)  # Assumed last actual price date
                current_price = self.data_processor.get_stock_data(ticker, period="5d")['Close'].iloc[-1]
                
                future_price = future_df['Predicted_Close'].iloc[-1]
                expected_return = (future_price / current_price - 1) * 100
                
                # Get standard deviation of future prices as uncertainty indicator
                uncertainty = future_df['Predicted_Close'].std() / future_df['Predicted_Close'].mean() * 100
                
                # Store results
                results.append({
                    'Ticker': ticker,
                    'Current Price': round(current_price, 2),
                    f'Predicted Price ({days_to_predict} days)': round(future_price, 2),
                    'Expected Return (%)': round(expected_return, 2),
                    'Prediction Uncertainty (%)': round(uncertainty, 2),
                    'MAPE (%)': round(metrics.get('mape', 0), 2),
                    'Direction Accuracy (%)': round(metrics.get('direction_accuracy', 0), 2),
                    'Risk Level': metrics.get('risk_level', 'N/A')
                })
                
                # Store future prediction DataFrame
                future_predictions[ticker] = future_df
                
            except Exception as e:
                logger.error(f"Error comparing {ticker}: {e}")
                continue
        
        if not results:
            logger.warning("No stocks successfully compared")
            return pd.DataFrame()
        
        # Create DataFrame and sort by expected return
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values('Expected Return (%)', ascending=False)
        
        # Visualize comparison results
        if visualize and future_predictions:
            self.visualizer.visualize_stock_comparison(comparison_df, future_predictions, days_to_predict)
        
        logger.info(f"Stock comparison complete, {len(comparison_df)} stocks compared")
        return comparison_df
    
    def run_backtest(self, ticker, start_date=None, end_date=None, prediction_window=30):
        """
        Run backtest for specific stock
        
        Parameters:
            ticker (str): Stock ticker symbol
            start_date (str): Backtest start date (YYYY-MM-DD)
            end_date (str): Backtest end date (YYYY-MM-DD)
            prediction_window (int): Number of days for each prediction
            
        Returns:
            DataFrame: Backtest results
        """
        logger.info(f"Running backtest analysis for {ticker}...")
        
        try:
            # Get backtest period stock data
            if start_date is None:
                # Default to using past year's data
                df = self.data_processor.get_stock_data(ticker, period="1y")
            else:
                # Use specified date range
                if end_date is None:
                    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
                import yfinance as yf
                df = yf.download(ticker, start=start_date, end=end_date)
            
            if df is None or df.empty:
                logger.error(f"Cannot get backtest data for {ticker}")
                return None
            
            logger.info(f"Got backtest data from {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
            
            # Ensure model exists
            model_dir = os.path.join(self.base_path, ticker)
            if not os.path.exists(model_dir):
                logger.warning(f"Model for {ticker} does not exist, attempting to train...")
                success = self.process_ticker(ticker, visualize=False)
                if not success:
                    logger.error(f"Cannot process {ticker}, backtest failed")
                    return None
            
            # Create backtest points - approximately monthly
            test_points = pd.date_range(start=df.index[self.sequence_length + 50], 
                                       end=df.index[-prediction_window-1], 
                                       freq='30D')
            
            if len(test_points) < 3:
                logger.warning(f"Too few backtest points ({len(test_points)}), trying weekly frequency")
                test_points = pd.date_range(start=df.index[self.sequence_length + 50], 
                                          end=df.index[-prediction_window-1], 
                                          freq='7D')
            
            logger.info(f"Will run backtest at {len(test_points)} time points")
            
            # Start backtest
            backtest_results = []
            
            for test_date in test_points:
                # Find closest trading day
                closest_date_idx = df.index.get_indexer([test_date], method='nearest')[0]
                test_date_actual = df.index[closest_date_idx]
                
                # Get historical data up to this date
                historical_df = df.loc[:test_date_actual].copy()
                
                # Add technical indicators
                historical_df = self.data_processor.add_technical_indicators(historical_df)
                if historical_df is None or len(historical_df) < self.sequence_length + 20:
                    logger.warning(f"Insufficient data points at {test_date_actual.strftime('%Y-%m-%d')}, skipping")
                    continue
                
                # For simplicity in backtesting, use original model
                model, scaler = self.model_manager.load_model_files(ticker)
                if model is None or scaler is None:
                    logger.error(f"Cannot load model or scaler for {ticker}, backtest failed")
                    return None
                
                # Use model to predict next N days
                # Get latest sequence
                historical_scaled = scaler.transform(historical_df.values)
                last_sequence = historical_scaled[-self.sequence_length:]
                
                # Predict next N days
                future_prices = []
                current_sequence = last_sequence.copy()
                
                for _ in range(prediction_window):
                    # Reshape sequence for model input
                    current_sequence_reshaped = current_sequence.reshape(1, self.sequence_length, historical_df.shape[1])
                    
                    # Use model to predict next value
                    next_pred = model.predict(current_sequence_reshaped, verbose=0)[0, 0]
                    
                    # Prepare array for inverse scaling
                    next_pred_for_inverse = np.zeros((1, historical_df.shape[1]))
                    next_pred_for_inverse[0, 0] = next_pred  # Assume first column is Close price
                    
                    # Inverse scale to get actual price
                    next_price = scaler.inverse_transform(next_pred_for_inverse)[0, 0]
                    future_prices.append(next_price)
                    
                    # Update sequence
                    new_values = current_sequence[-1].copy()
                    new_values[0] = next_pred  # Only update Close price
                    current_sequence = np.vstack([current_sequence[1:], [new_values]])
                
                # Get actual future prices (if available)
                future_dates = pd.date_range(start=test_date_actual + pd.Timedelta(days=1), periods=prediction_window)
                
                # Find closest actual future dates
                actual_future_prices = []
                for future_date in future_dates:
                    future_idx = df.index.get_indexer([future_date], method='nearest')[0]
                    if future_idx < len(df):
                        actual_future_prices.append(df['Close'].iloc[future_idx])
                    else:
                        actual_future_prices.append(None)
                
                # Find actual price at end of prediction window
                if len(actual_future_prices) == prediction_window and actual_future_prices[-1] is not None:
                    actual_end_price = actual_future_prices[-1]
                    predicted_end_price = future_prices[-1]
                    
                    # Calculate prediction error percentage
                    prediction_error_pct = (predicted_end_price / actual_end_price - 1) * 100
                    
                    # Calculate actual return
                    actual_return_pct = (actual_end_price / historical_df['Close'].iloc[-1] - 1) * 100
                    
                    # Calculate predicted return
                    predicted_return_pct = (predicted_end_price / historical_df['Close'].iloc[-1] - 1) * 100
                    
                    # Calculate if direction is correct (up or down)
                    actual_direction = actual_return_pct > 0
                    predicted_direction = predicted_return_pct > 0
                    direction_correct = actual_direction == predicted_direction
                    
                    # Add to backtest results
                    backtest_results.append({
                        'Backtest Date': test_date_actual.strftime('%Y-%m-%d'),
                        'Initial Price': historical_df['Close'].iloc[-1],
                        f'Predicted Price ({prediction_window} days)': predicted_end_price,
                        f'Actual Price ({prediction_window} days)': actual_end_price,
                        'Prediction Error (%)': prediction_error_pct,
                        'Actual Return (%)': actual_return_pct,
                        'Predicted Return (%)': predicted_return_pct,
                        'Direction Correct': direction_correct
                    })
                else:
                    logger.warning(f"Insufficient future data at {test_date_actual.strftime('%Y-%m-%d')}, skipping")
            
            if not backtest_results:
                logger.error(f"Backtest for {ticker} failed: No valid backtest results")
                return None
            
            # Create backtest results DataFrame
            backtest_df = pd.DataFrame(backtest_results)
            
            # Calculate backtest performance metrics
            direction_accuracy = backtest_df['Direction Correct'].mean() * 100
            mean_error = backtest_df['Prediction Error (%)'].mean()
            mean_abs_error = backtest_df['Prediction Error (%)'].abs().mean()
            
            logger.info(f"{ticker} backtest complete. Direction accuracy: {direction_accuracy:.2f}%, "
                       f"Mean error: {mean_error:.2f}%, Mean absolute error: {mean_abs_error:.2f}%")
            
            # Visualize backtest results
            self.visualizer.visualize_backtest(ticker, backtest_df, prediction_window)
            
            return backtest_df
            
        except Exception as e:
            logger.error(f"Error running backtest for {ticker}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

# Utility functions for direct use
def visualize_stock_prediction(ticker, days_to_predict=30, force_retrain=False):
    """
    Convenience function to visualize prediction results for specific stock
    
    Parameters:
        ticker (str): Stock ticker symbol
        days_to_predict (int): Number of days to predict
        force_retrain (bool): Whether to force model retraining
        
    Returns:
        bool: Whether successful
    """
    predictor = StockPricePredictor(sequence_length=30, 
                                   feature_engineering_level='advanced',
                                   risk_assessment=True)
    
    # Try to load existing model (unless force retraining)
    if not force_retrain:
        model, scaler = predictor.model_manager.load_model_files(ticker)
    else:
        model, scaler = None, None
    
    # If no existing model found or force retraining, train new model
    if model is None or scaler is None:
        print(f"Training new model for {ticker}...")
        return predictor.process_ticker(ticker, visualize=True)
    
    # Get latest data
    print(f"Getting latest data for {ticker}...")
    df = predictor.data_processor.get_stock_data(ticker, period="1y")
    if df is None or df.empty:
        print(f"Failed to get data for {ticker}")
        return False
    
    # Add technical indicators
    df = predictor.data_processor.add_technical_indicators(df)
    if df is None or df.empty:
        print(f"Failed to add technical indicators for {ticker}")
        return False
    
    # Prepare data
    print("Preparing data...")
    prepare_result = predictor.data_processor.prepare_data(df)
    if prepare_result[0] is None:
        print(f"Failed to prepare data for {ticker}")
        return False
    
    X_train, y_train, X_test, y_test, _, (_, test_dates) = prepare_result
    
    # Evaluate model
    print("Evaluating model...")
    eval_result = predictor.model_manager.evaluate_model(model, X_test, y_test, scaler, df)
    if eval_result[0] is None:
        print(f"Failed to evaluate model for {ticker}")
        return False
    
    y_pred, y_test_inv, metrics = eval_result
    
    # Predict future prices
    print(f"Predicting future {days_to_predict} days for {ticker}...")
    future_df = predictor.model_manager.predict_future(ticker, days=days_to_predict)
    if future_df is None:
        print(f"Failed to predict future prices for {ticker}")
        return False
    
    # Visualize results
    print("Generating visualization results...")
    predictor.visualizer.visualize_results(ticker, test_dates, y_test_inv, y_pred, metrics, None, future_df)
    print(f"Prediction analysis for {ticker} complete!")
    return True

def analyze_portfolio(tickers, days_to_predict=30):
    """
    Analyze predictions for a portfolio of stocks
    
    Parameters:
        tickers (list): List of stock ticker symbols
        days_to_predict (int): Number of days to predict
        
    Returns:
        DataFrame: Portfolio analysis results
    """
    predictor = StockPricePredictor(sequence_length=30, 
                                   feature_engineering_level='advanced',
                                   risk_assessment=True)
    
    print(f"Analyzing portfolio of {len(tickers)} stocks...")
    
    # Compare stocks
    results = predictor.compare_stocks(tickers, days_to_predict=days_to_predict, visualize=True)
    
    if results is None or len(results) == 0:
        print("Portfolio analysis failed")
        return None
    
    # Print results
    print("\n=============== Portfolio Analysis Results ===============")
    print(f"Analyzed {len(results)} stocks, prediction period: {days_to_predict} days")
    print("\nStock ranking (by expected return):")
    print(results[['Ticker', 'Current Price', f'Predicted Price ({days_to_predict} days)', 'Expected Return (%)', 'Risk Level']].to_string(index=False))
    
    # Calculate portfolio statistics
    avg_return = results['Expected Return (%)'].mean()
    avg_uncertainty = results['Prediction Uncertainty (%)'].mean()
    avg_mape = results['MAPE (%)'].mean()
    
    print(f"\nPortfolio average expected return: {avg_return:.2f}%")
    print(f"Portfolio average prediction uncertainty: {avg_uncertainty:.2f}%")
    print(f"Portfolio average MAPE: {avg_mape:.2f}%")
    
    # Risk distribution
    risk_counts = results['Risk Level'].value_counts()
    print("\nRisk distribution:")
    for risk, count in risk_counts.items():
        print(f"  {risk}: {count} stocks ({count/len(results)*100:.1f}%)")
    
    return results

# Main program entry
if __name__ == "__main__":
    # Create predictor instance
    predictor = StockPricePredictor(
        sequence_length=30,             # 30 days historical sequence
        feature_engineering_level='advanced',  # Use advanced feature engineering
        risk_assessment=True,           # Enable risk assessment
        validation_method='time_series_split',  # Use time series cross-validation
        confidence_interval=0.95        # Use 95% confidence interval
    )
    
    # Method 1: Process single stock and generate visualization
    # predictor.process_ticker('AAPL', visualize=True)
    
    # Method 2: Use convenience function for visualization
    # visualize_stock_prediction('MSFT', days_to_predict=30)
    
    # Method 3: Process multiple stocks
    custom_tickers = ['AAPL', 'NVDA']#,'TSLA', 'GOOGL', 'AMZN', 'NVDA']
    successful, failed = predictor.process_multiple_tickers(tickers=custom_tickers, visualize=True)
    print(f"Successfully processed stocks: {successful}")
    print(f"Failed stocks: {failed}")
    
    # Method 4: Get summary of all trained models
    # summary = predictor.get_model_summary()
    # print("Model performance summary:")
    # print(summary)
    
    # Method 5: Compare prediction results for multiple stocks
    # comparison = predictor.compare_stocks(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'], days_to_predict=30)
    # print(comparison)
    
    # Method 6: Run backtest
    # backtest = predictor.run_backtest('AAPL', prediction_window=30)
    # print(backtest)
    
    # Method 7: Analyze portfolio
    # portfolio = analyze_portfolio(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'], days_to_predict=30)
