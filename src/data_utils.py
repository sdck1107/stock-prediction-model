import numpy as np
import pandas as pd
import os
import datetime
import logging
import pickle
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger("StockPredictor.DataProcessor")

class DataProcessor:
    def __init__(self, predictor):
        """
        Initialize with reference to parent predictor
        
        Parameters:
            predictor: Parent StockPricePredictor instance
        """
        self.predictor = predictor
    
    def get_sp500_tickers(self, limit=None):
        """
        Get S&P 500 stock tickers
        
        Parameters:
            limit (int, optional): Limit number of tickers returned
            
        Returns:
            list: List of stock tickers
        """
        try:
            logger.info("Getting S&P 500 stock tickers...")
            # Try to read cached tickers
            cache_file = os.path.join(self.predictor.data_cache_dir, 'sp500_tickers.pkl')
            
            if os.path.exists(cache_file) and (datetime.datetime.now() - 
                                              datetime.datetime.fromtimestamp(os.path.getmtime(cache_file))).days < 7:
                with open(cache_file, 'rb') as f:
                    tickers = pickle.load(f)
                logger.info(f"Loaded {len(tickers)} S&P 500 tickers from cache")
            else:
                # If cache doesn't exist or expired, get latest data from Wikipedia
                sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
                tickers = sp500['Symbol'].tolist()
                
                # Clean ticker symbols (remove special characters)
                tickers = [ticker.replace('.', '-') for ticker in tickers]
                
                # Save to cache
                with open(cache_file, 'wb') as f:
                    pickle.dump(tickers, f)
                
                logger.info(f"Retrieved and cached {len(tickers)} S&P 500 tickers from web")
            
            if limit:
                tickers = tickers[:limit]
            
            return tickers
            
        except Exception as e:
            logger.error(f"Error getting S&P 500 tickers: {e}")
            # If error, return default stock list
            return ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']
    
    def get_stock_data(self, ticker, period="5y", interval="1d", provider="yfinance", cache=True):
        """
        Get stock data from multiple data sources with caching for consistency
        
        Parameters:
            ticker (str): Stock ticker symbol
            period (str, default="5y"): Time period for data
            interval (str, default="1d"): Data interval
            provider (str, default="yfinance"): Data provider
            cache (bool, default=True): Whether to use cached data
            
        Returns:
            DataFrame: Stock data
        """
        logger.info(f"Getting stock data for {ticker}...")
        
        # Check cache
        cache_file = os.path.join(self.predictor.data_cache_dir, f"{ticker}_raw_data.pkl")
        if cache and os.path.exists(cache_file):
            cache_time = datetime.datetime.fromtimestamp(os.path.getmtime(cache_file))
            current_time = datetime.datetime.now()
            
            # If cache is less than 24 hours old, use cached data
            if (current_time - cache_time).days < 1:
                try:
                    with open(cache_file, 'rb') as f:
                        df = pickle.load(f)
                    logger.info(f"Loaded {ticker} data from cache, {len(df)} data points")
                    return df
                except Exception as e:
                    logger.warning(f"Error reading cached data for {ticker}: {e}")
        
        # Get data from data source
        if provider == "yfinance":
            try:
                df = yf.download(ticker, period=period, interval=interval, progress=False)
                
                # Check if data is empty
                if df.empty:
                    raise Exception("Retrieved data is empty")
                    
                # Output detailed data info for debugging
                logger.info(f"Shape of {ticker} data from yfinance: {df.shape}")
                logger.info(f"Retrieved raw data columns: {df.columns.tolist()}")
                logger.info(f"Data range: {df.index.min()} to {df.index.max()}")
                
                # Handle MultiIndex columns (if exist)
                if isinstance(df.columns, pd.MultiIndex):
                    logger.info("Detected MultiIndex columns, flattening...")
                    # Only keep first level column names
                    df.columns = df.columns.get_level_values(0)
                    logger.info(f"Flattened column names: {df.columns.tolist()}")
                
                # Ensure all column names are strings
                df.columns = [str(col) for col in df.columns]
                
                # Ensure 'Close' column exists
                if 'Close' not in df.columns and 'Adj Close' in df.columns:
                    df['Close'] = df['Adj Close']
                    logger.info("Created 'Close' column from 'Adj Close'")
                
                # Cache data
                if cache:
                    try:
                        with open(cache_file, 'wb') as f:
                            pickle.dump(df, f)
                        logger.info(f"{ticker} data cached to: {cache_file}")
                    except Exception as e:
                        logger.warning(f"Error caching {ticker} data: {e}")
                
                # Check data validity again
                if len(df) < self.predictor.sequence_length + 20:
                    logger.warning(f"Number of data points for {ticker} ({len(df)}) is insufficient. Need at least {self.predictor.sequence_length + 20} points.")
                    
                return df
                
            except Exception as e:
                logger.warning(f"Error getting {ticker} data using yfinance: {e}")
                logger.warning("Trying to use sample data instead...")
                
                # Create a simulated data as fallback
                dates = pd.date_range(start='2020-01-01', periods=800, freq='D')
                df = pd.DataFrame({
                    'Open': np.random.normal(100, 10, 800),
                    'High': np.random.normal(105, 10, 800),
                    'Low': np.random.normal(95, 10, 800),
                    'Close': np.random.normal(100, 10, 800),
                    'Adj Close': np.random.normal(100, 10, 800),
                    'Volume': np.random.normal(1000000, 200000, 800)
                }, index=dates)
                
                # Ensure simulated data is closer to real data distribution
                for i in range(1, len(df)):
                    # Add some trend and autocorrelation
                    df.iloc[i, df.columns.get_indexer(['Close'])] = (
                        df.iloc[i-1, df.columns.get_indexer(['Close'])][0] * 0.99 + 
                        np.random.normal(0, 3)
                    )
                    # Ensure High >= Open >= Low and High >= Close >= Low
                    df.iloc[i, df.columns.get_indexer(['Open'])] = df.iloc[i, df.columns.get_indexer(['Close'])][0] + np.random.normal(0, 2)
                    df.iloc[i, df.columns.get_indexer(['High'])] = max(df.iloc[i, df.columns.get_indexer(['Open'])][0], df.iloc[i, df.columns.get_indexer(['Close'])][0]) + abs(np.random.normal(0, 2))
                    df.iloc[i, df.columns.get_indexer(['Low'])] = min(df.iloc[i, df.columns.get_indexer(['Open'])][0], df.iloc[i, df.columns.get_indexer(['Close'])][0]) - abs(np.random.normal(0, 2))
                
                logger.info(f"Created simulated data for {ticker}, {len(df)} data points")
                
                return df
        
        return None
    
    def add_technical_indicators(self, df):
        """
        Add technical indicators as prediction features
        
        Based on different feature engineering levels, builds feature sets from basic to advanced
        to enhance prediction accuracy. Financial research shows technical indicators combination
        can effectively capture market trends, momentum, volatility and reversal patterns.
        
        Parameters:
            df (DataFrame): Raw stock data
            
        Returns:
            DataFrame: Data with technical indicators added
        """
        logger.info("Adding technical indicators...")
        if df is None or df.empty:
            logger.error("Cannot add technical indicators: Data is empty")
            return None
            
        try:
            # Ensure all column names are strings
            df.columns = [str(col) for col in df.columns]
            
            # Check if necessary columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.warning(f"Missing required columns: {missing_columns}")
                
                # Special handling for 'Close' column
                if 'Close' in missing_columns and 'Adj Close' in df.columns:
                    df['Close'] = df['Adj Close']
                    missing_columns.remove('Close')
                    logger.info("Created 'Close' column from 'Adj Close'")
                
                # If still missing key columns, cannot continue
                if 'Close' in missing_columns:
                    logger.error("Missing 'Close' column, cannot continue")
                    return None
            
            # Copy dataframe to avoid modifying original data
            df_with_indicators = df.copy()
            
            # Basic feature set (for all levels)
            # ===========================
            
            # 1. Basic price indicators
            df_with_indicators['Price_Change'] = df['Close'].pct_change() * 100  # Daily price change percentage
            df_with_indicators['Price_Change_1d'] = df['Close'].diff()  # Absolute price change
            
            # 2. Moving averages (MA)
            df_with_indicators['MA5'] = df['Close'].rolling(window=5).mean()
            df_with_indicators['MA10'] = df['Close'].rolling(window=10).mean()
            df_with_indicators['MA20'] = df['Close'].rolling(window=20).mean()
            df_with_indicators['MA50'] = df['Close'].rolling(window=50).mean()
            
            # 3. Price to moving average relationships
            df_with_indicators['Close_MA5_Ratio'] = df['Close'] / df_with_indicators['MA5']
            df_with_indicators['Close_MA20_Ratio'] = df['Close'] / df_with_indicators['MA20']
            
            # 4. Volatility indicators
            df_with_indicators['Daily_Range'] = (df['High'] - df['Low']) / df['Close'] * 100  # Daily volatility
            df_with_indicators['Daily_Return'] = (df['Close'] / df['Open'] - 1) * 100  # Daily return
            
            # Intermediate feature set
            # =======================
            if self.predictor.feature_engineering_level in ['intermediate', 'advanced']:
                # 5. Relative Strength Index (RSI)
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                df_with_indicators['RSI14'] = 100 - (100 / (1 + rs))
                
                # 6. Moving Average Convergence/Divergence (MACD)
                ema12 = df['Close'].ewm(span=12, adjust=False).mean()
                ema26 = df['Close'].ewm(span=26, adjust=False).mean()
                macd = ema12 - ema26
                signal = macd.ewm(span=9, adjust=False).mean()
                df_with_indicators['MACD'] = macd
                df_with_indicators['MACD_Signal'] = signal
                df_with_indicators['MACD_Histogram'] = macd - signal
                
                # 7. Bollinger Bands
                df_with_indicators['BB_Middle'] = df['Close'].rolling(window=20).mean()
                bb_std = df['Close'].rolling(window=20).std()
                df_with_indicators['BB_Upper'] = df_with_indicators['BB_Middle'] + (bb_std * 2)
                df_with_indicators['BB_Lower'] = df_with_indicators['BB_Middle'] - (bb_std * 2)
                df_with_indicators['BB_Width'] = (df_with_indicators['BB_Upper'] - df_with_indicators['BB_Lower']) / df_with_indicators['BB_Middle'] * 100
                df_with_indicators['BB_Position'] = (df['Close'] - df_with_indicators['BB_Lower']) / (df_with_indicators['BB_Upper'] - df_with_indicators['BB_Lower'])
                
                # 8. Momentum indicators
                df_with_indicators['Momentum_5d'] = df['Close'] / df['Close'].shift(5) - 1
                df_with_indicators['Momentum_10d'] = df['Close'] / df['Close'].shift(10) - 1
                
                # 9. Volume indicators
                df_with_indicators['Volume_Change'] = df['Volume'].pct_change() * 100
                df_with_indicators['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
                df_with_indicators['Volume_Ratio'] = df['Volume'] / df_with_indicators['Volume_MA5']
            
            # Advanced feature set
            # ==================
            if self.predictor.feature_engineering_level == 'advanced':
                # 10. Average True Range (ATR) - volatility indicator
                high_low = df['High'] - df['Low']
                high_close = (df['High'] - df['Close'].shift()).abs()
                low_close = (df['Low'] - df['Close'].shift()).abs()
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                df_with_indicators['ATR14'] = tr.rolling(window=14).mean()
                df_with_indicators['ATR_Ratio'] = df_with_indicators['ATR14'] / df['Close'] * 100
                
                # 11. Stochastic Oscillator
                low_min = df['Low'].rolling(window=14).min()
                high_max = df['High'].rolling(window=14).max()
                df_with_indicators['SlowK'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
                df_with_indicators['SlowD'] = df_with_indicators['SlowK'].rolling(window=3).mean()
                
                # 12. On-Balance Volume (OBV)
                obv = pd.Series(0, index=df.index)
                for i in range(1, len(df)):
                    if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                        obv.iloc[i] = obv.iloc[i-1] + df['Volume'].iloc[i]
                    elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                        obv.iloc[i] = obv.iloc[i-1] - df['Volume'].iloc[i]
                    else:
                        obv.iloc[i] = obv.iloc[i-1]
                df_with_indicators['OBV'] = obv
                df_with_indicators['OBV_MA10'] = obv.rolling(window=10).mean()
                
                # 13. Money Flow Index (MFI)
                typical_price = (df['High'] + df['Low'] + df['Close']) / 3
                money_flow = typical_price * df['Volume']
                positive_flow = pd.Series(0, index=df.index)
                negative_flow = pd.Series(0, index=df.index)
                
                for i in range(1, len(df)):
                    if typical_price.iloc[i] > typical_price.iloc[i-1]:
                        positive_flow.iloc[i] = money_flow.iloc[i]
                    elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                        negative_flow.iloc[i] = money_flow.iloc[i]
                
                positive_mf = positive_flow.rolling(window=14).sum()
                negative_mf = negative_flow.rolling(window=14).sum()
                
                # Prevent division by zero
                mfi = 100 - (100 / (1 + positive_mf / negative_mf.replace(0, 1e-10)))
                df_with_indicators['MFI'] = mfi
                
                # 14. Volume Weighted Average Price (VWAP)
                df_with_indicators['VWAP'] = (df['Close'] * df['Volume']).rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()
                df_with_indicators['VWAP_Ratio'] = df['Close'] / df_with_indicators['VWAP']
                
                # 15. Trend strength indicator
                # Calculate trend strength - difference between current price and price N days ago as a proportion of ATR
                df_with_indicators['Trend_20d'] = (df['Close'] - df['Close'].shift(20)) / df_with_indicators['ATR14']
                
                # 16. Volatility analysis (using historical volatility)
                df_with_indicators['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
                df_with_indicators['Volatility_20d'] = df_with_indicators['Returns'].rolling(window=20).std() * np.sqrt(252) * 100
                
                # 17. Seasonality features
                # Add date-related features
                df_with_indicators['Day_of_Week'] = df.index.dayofweek
                df_with_indicators['Month'] = df.index.month
                df_with_indicators['Quarter'] = df.index.quarter
                
                # 18. Price cyclicality analysis
                # Get 50-day and 200-day high and low points
                df_with_indicators['High_50d'] = df['High'].rolling(window=50).max()
                df_with_indicators['Low_50d'] = df['Low'].rolling(window=50).min()
                df_with_indicators['High_200d'] = df['High'].rolling(window=200).max()
                df_with_indicators['Low_200d'] = df['Low'].rolling(window=200).min()
                
                # Distance to high and low points as percentage
                df_with_indicators['Dist_to_High_50d'] = (df_with_indicators['High_50d'] - df['Close']) / df['Close'] * 100
                df_with_indicators['Dist_to_Low_50d'] = (df['Close'] - df_with_indicators['Low_50d']) / df['Close'] * 100
            
            # Remove NaN values
            df_with_indicators = df_with_indicators.dropna()
            
            logger.info(f"Successfully added technical indicators, feature count: {df_with_indicators.shape[1]}")
            
            # Remove infinite values and NaN values
            df_with_indicators = df_with_indicators.replace([np.inf, -np.inf], np.nan).dropna()
            
            return df_with_indicators
        
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def analyze_feature_importance(self, df, target_col='Close'):
        """
        Analyze feature importance based on correlation and optionally ML model
        
        Parameters:
            df (DataFrame): DataFrame with features
            target_col (str): Target column name
            
        Returns:
            DataFrame: Features sorted by importance
        """
        logger.info("Analyzing feature importance...")
        try:
            # Calculate correlation with target
            correlations = df.corr()[target_col].abs().sort_values(ascending=False)
            
            # Remove target column's self-correlation
            correlations = correlations.drop(target_col)
            
            # Create feature importance dataframe
            importance_df = pd.DataFrame({
                'Feature': correlations.index,
                'Correlation': correlations.values,
                'Rank': range(1, len(correlations) + 1)
            })
            
            logger.info(f"Feature importance analysis complete, top 5 important features:\n{importance_df.head(5)}")
            return importance_df
            
        except Exception as e:
            logger.error(f"Error analyzing feature importance: {e}")
            return None
    
    def prepare_data(self, df, target_col='Close', feature_selection=True, max_features=30):
        """
        Prepare data for model training
        
        Parameters:
            df (DataFrame): Stock data
            target_col (str, default='Close'): Target column name
            feature_selection (bool, default=True): Whether to perform feature selection
            max_features (int, default=30): Maximum number of features
            
        Returns:
            tuple: (X_train, y_train, X_test, y_test, scaler, dates)
        """
        logger.info("Preparing training data...")
        if df is None or df.empty:
            logger.error("Cannot prepare data: Data is empty")
            return None, None, None, None, None, None
            
        try:
            # Ensure all column names are strings
            df.columns = [str(col) for col in df.columns]
            
            # Check if target column exists
            if target_col not in df.columns:
                logger.warning(f"Target column '{target_col}' not found")
                
                # Try to find alternative column (ignore case)
                for col in df.columns:
                    if col.lower() == target_col.lower():
                        target_col = col
                        logger.info(f"Using alternative column (ignore case): {target_col}")
                        break
                else:
                    # If not found through case-insensitive match, try using 'Adj Close'
                    if 'Adj Close' in df.columns:
                        target_col = 'Adj Close'
                        logger.info(f"Using 'Adj Close' as alternative target column")
                    else:
                        # Last resort: use first column as target
                        target_col = df.columns[0]
                        logger.warning(f"Cannot find suitable target column, using first column: {target_col}")
            
            # Feature selection - select most correlated features based on importance
            if feature_selection and df.shape[1] > max_features + 1:  # +1 for target column
                logger.info(f"Performing feature selection, selecting most important {max_features} from {df.shape[1]} features")
                importance_df = self.analyze_feature_importance(df, target_col)
                
                if importance_df is not None:
                    # Keep target column and most important features
                    top_features = importance_df['Feature'].head(max_features).tolist()
                    
                    if target_col not in top_features:
                        top_features = [target_col] + top_features
                    
                    df = df[top_features]
                    logger.info(f"Shape after feature selection: {df.shape}")
            
            # Select features
            features = df.columns.tolist()
            
            # Get target column index
            target_idx = features.index(target_col)
            logger.info(f"Target column index: {target_idx}")
            
            # Create feature scaler
            scaler = StandardScaler()  # Use StandardScaler instead of MinMaxScaler for better handling of outliers
            df_scaled = scaler.fit_transform(df)
            
            # Prepare data based on different validation methods
            if self.predictor.validation_method == 'simple_split':
                # Simple split
                X, y = self.create_sequences(df_scaled, self.predictor.sequence_length, target_idx)
                
                # Split into training and test sets
                split = int(self.predictor.train_split * len(X))
                if split <= 0:
                    logger.warning("Too few data points to properly split training and test sets")
                    split = max(1, int(0.5 * len(X)))  # Ensure at least one sample
                    
                X_train, X_test = X[:split], X[split:]
                y_train, y_test = y[:split], y[split:]
                
                # Get corresponding dates
                dates = df.index[self.predictor.sequence_length:].to_numpy()
                train_dates = dates[:split]
                test_dates = dates[split:]
                
            elif self.predictor.validation_method == 'time_series_split':
                # Time series cross-validation
                X, y = self.create_sequences(df_scaled, self.predictor.sequence_length, target_idx)
                tscv = TimeSeriesSplit(n_splits=5)
                
                # Use last split as final train/test sets
                split_idx = 0
                for train_idx, test_idx in tscv.split(X):
                    split_idx += 1
                    if split_idx == 5:  # Use last split
                        X_train, X_test = X[train_idx], X[test_idx]
                        y_train, y_test = y[train_idx], y[test_idx]
                
                # Get corresponding dates
                dates = df.index[self.predictor.sequence_length:].to_numpy()
                date_indices = np.arange(len(dates))
                
                # Use same indices as time series split to get dates
                split_idx = 0
                for train_idx, test_idx in tscv.split(date_indices):
                    split_idx += 1
                    if split_idx == 5:  # Use last split
                        train_dates = dates[train_idx]
                        test_dates = dates[test_idx]
                
            else:  # 'walk_forward' or other
                logger.warning(f"Unsupported validation method: {self.predictor.validation_method}, using simple split")
                # Default to simple split
                X, y = self.create_sequences(df_scaled, self.predictor.sequence_length, target_idx)
                
                split = int(self.predictor.train_split * len(X))
                X_train, X_test = X[:split], X[split:]
                y_train, y_test = y[:split], y[split:]
                
                dates = df.index[self.predictor.sequence_length:].to_numpy()
                train_dates = dates[:split]
                test_dates = dates[split:]
            
            logger.info(f"Data preparation complete. Training samples: {len(X_train)}, Test samples: {len(X_test)}, Features: {X.shape[2]}")
            
            return X_train, y_train, X_test, y_test, scaler, (train_dates, test_dates)
        
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None, None, None, None, None
    
    def create_sequences(self, data, seq_length, target_col_idx):
        """
        Create sequences for LSTM training
        
        Parameters:
            data (array): Scaled data
            seq_length (int): Sequence length
            target_col_idx (int): Index of target column
            
        Returns:
            tuple: (X, y) input sequences and target values
        """
        X, y = [], []
        for i in range(len(data) - seq_length):
            # Use past seq_length days of all features as input
            X.append(data[i:i+seq_length])
            # Use day i+seq_length's target column value as prediction target
            y.append(data[i+seq_length, target_col_idx])
        return np.array(X), np.array(y)
