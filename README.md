# Stock Price Prediction System

A machine learning system for stock price prediction with LSTM/GRU models and a Flask API.

## Overview

This project uses deep learning (LSTM and GRU) to predict stock prices with features including:

- Multiple stock analysis and comparison
- Technical indicator features
- Risk assessment
- Backtesting capabilities
- Confidence intervals
- Interactive web interface

## Project Structure

- `app/` - Flask web application
  - `app.py` - Main Flask application
  - `templates/` - HTML templates
- `src/` - Core modules
  - `stock_predictor.py` - Main StockPricePredictor class
  - `data_utils.py` - Data processing
  - `model_utils.py` - Model management
  - `visualization.py` - Visualization utilities
- `models/` - Trained model files (not included in repo, download separately)
- `main.py` - Script to run predictor directly(For training stocks only) 



## Usage

### Running the Web Application

```bash
python app/app.py
```

The web interface will be available at http://localhost:5006

### Running Stock Prediction from CLI

```bash
python main.py
```

You can modify `main.py` to analyze different stocks by changing the `custom_tickers` list.

### Using the StockPricePredictor Library

```python
from src.stock_predictor import StockPricePredictor

# Initialize predictor
predictor = StockPricePredictor(
    sequence_length=30,
    feature_engineering_level='advanced',
    risk_assessment=True
)

# Process a single stock
predictor.process_ticker('AAPL', visualize=True)

# Compare multiple stocks
comparison = predictor.compare_stocks(['AAPL', 'MSFT', 'GOOGL'], days_to_predict=30)
print(comparison)
```

## API Endpoints

The Flask app provides the following API endpoints:

- `GET /` - Home page
- `POST /api/predict` - Get prediction for a specific stock
- `GET /api/tickers` - Get list of available pre-trained models
- `POST /api/backtest` - Run backtest for a stock
- `POST /api/compare` - Compare multiple stocks
