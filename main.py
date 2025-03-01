from stock_predictor import StockPricePredictor, visualize_stock_prediction, analyze_portfolio
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
    custom_tickers = ['MA']#,'TSLA', 'GOOGL', 'AMZN', 'NVDA']
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