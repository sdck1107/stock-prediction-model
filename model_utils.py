import numpy as np
import pandas as pd
import os
import joblib
import time
import datetime
import logging
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2

logger = logging.getLogger("StockPredictor.ModelManager")

class ModelManager:
    def __init__(self, predictor):
        """
        Initialize with reference to parent predictor
        
        Parameters:
            predictor: Parent StockPricePredictor instance
        """
        self.predictor = predictor
    
    def build_model(self, input_shape, model_type='lstm_gru'):
        """
        Build neural network model
        
        Parameters:
            input_shape (tuple): Input shape (sequence_length, n_features)
            model_type (str): Model type
            
        Returns:
            Sequential: Built model
        """
        logger.info(f"Building deep learning model, input shape: {input_shape}...")
        
        if model_type == 'lstm_gru':
            # Advanced LSTM+GRU model, providing stronger sequence learning capability and gradient flow
            model = Sequential([
                # First layer: Bidirectional LSTM, enhances long-term dependency capturing
                Bidirectional(LSTM(128, return_sequences=True, 
                             kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)), 
                             input_shape=input_shape),
                BatchNormalization(),  # Accelerate training and reduce overfitting
                Dropout(0.3),  # Prevent overfitting
                
                # Second layer: LSTM, captures medium-term patterns
                LSTM(96, return_sequences=True,
                     kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
                BatchNormalization(),
                Dropout(0.3),
                
                # Third layer: GRU, better handles short-term dependencies and changing trends
                GRU(64, return_sequences=False, 
                    kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
                BatchNormalization(),
                Dropout(0.3),
                
                # Fully connected layer, enhances feature representation
                Dense(48, activation='relu', 
                      kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
                BatchNormalization(),
                Dropout(0.2),
                
                # Output layer
                Dense(1)
            ])
            
            # Use Adam optimizer with initial learning rate and decay
            optimizer = Adam(learning_rate=0.001, decay=1e-6)
            model.compile(optimizer=optimizer, loss='mean_squared_error')
            
        elif model_type == 'simple_lstm':
            # Simplified LSTM model, suitable for less data or simpler patterns
            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(32, return_sequences=False),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1)
            ])
            optimizer = Adam(learning_rate=0.001)
            model.compile(optimizer=optimizer, loss='mean_squared_error')
            
        else:
            logger.warning(f"Unsupported model type: {model_type}, using default LSTM+GRU model")
            # Default to LSTM+GRU model
            model = Sequential([
                Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
                Dropout(0.3),
                LSTM(96, return_sequences=True),
                Dropout(0.3),
                GRU(64, return_sequences=False),
                Dropout(0.3),
                Dense(48, activation='relu'),
                Dropout(0.2),
                Dense(1)
            ])
            optimizer = Adam(learning_rate=0.001)
            model.compile(optimizer=optimizer, loss='mean_squared_error')
        
        logger.info(f"Model built: {model_type}")
        return model
    
    def train_model(self, ticker, X_train, y_train, X_test, y_test, model_type='lstm_gru', epochs=100, batch_size=32):
        """
        Train model and save
        
        Parameters:
            ticker (str): Stock ticker symbol
            X_train (array): Training features
            y_train (array): Training targets
            X_test (array): Testing features
            y_test (array): Testing targets
            model_type (str): Model type
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            
        Returns:
            tuple: (model, history) Trained model and training history
        """
        if X_train is None or y_train is None:
            logger.error(f"Cannot train model for {ticker}: Training data is empty")
            return None, None
            
        logger.info(f"Starting model training for {ticker}...")
        try:
            # Build model
            model = self.build_model((X_train.shape[1], X_train.shape[2]), model_type)
            
            # Create directory
            ticker_dir = os.path.join(self.predictor.base_path, ticker)
            os.makedirs(ticker_dir, exist_ok=True)
            
            # Set up callbacks
            checkpoint_path = os.path.join(ticker_dir, f"{ticker}_best_model.h5")
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
                ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', verbose=0),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=0.0001, verbose=1)
            ]
            
            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=1
            )
            
            # Load best model
            if os.path.exists(checkpoint_path):
                model = load_model(checkpoint_path)
            
            logger.info(f"{ticker} model training complete, training loss: {history.history['loss'][-1]:.4f}, validation loss: {history.history['val_loss'][-1]:.4f}")
            return model, history
            
        except Exception as e:
            logger.error(f"Error training model for {ticker}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None
    
    def evaluate_model(self, model, X_test, y_test, scaler, df_original=None, feature_index=0):
        """
        Evaluate model performance
        
        Parameters:
            model: Trained model
            X_test (array): Test features
            y_test (array): Test targets
            scaler: Scaler for inverse scaling
            df_original (DataFrame, optional): Original dataframe for risk assessment
            feature_index (int): Index of target feature
            
        Returns:
            tuple: (y_pred_inv, y_test_inv, metrics) Predictions, actual values, and evaluation metrics
        """
        if model is None or X_test is None or y_test is None:
            logger.error("Cannot evaluate model: Model or test data is empty")
            return None, None, None
            
        logger.info("Evaluating model performance...")
        try:
            # Predict
            y_pred = model.predict(X_test, verbose=0)
            
            # Ensure y_pred is a 1D array
            y_pred = y_pred.flatten()
            
            # Prepare arrays for inverse scaling
            y_test_for_inverse = np.zeros((len(y_test), scaler.scale_.shape[0]))
            y_test_for_inverse[:, feature_index] = y_test
            
            y_pred_for_inverse = np.zeros((len(y_pred), scaler.scale_.shape[0]))
            y_pred_for_inverse[:, feature_index] = y_pred
            
            # Inverse scale
            y_test_inv = scaler.inverse_transform(y_test_for_inverse)[:, feature_index]
            y_pred_inv = scaler.inverse_transform(y_pred_for_inverse)[:, feature_index]
            
            # Calculate basic metrics
            mse = mean_squared_error(y_test_inv, y_pred_inv)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test_inv, y_pred_inv)
            r2 = r2_score(y_test_inv, y_pred_inv)
            
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                mape = np.mean(np.abs((y_test_inv - y_pred_inv) / np.where(y_test_inv != 0, y_test_inv, 1e-10))) * 100
            
            logger.info(f"Basic evaluation metrics: MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}, MAPE: {mape:.2f}%")
            
            # Risk and volatility assessment
            errors = y_pred_inv - y_test_inv
            volatility = np.std(errors)
            
            # Calculate confidence interval
            confidence_level = self.predictor.confidence_interval
            lower_quantile = (1 - confidence_level) / 2
            upper_quantile = 1 - lower_quantile
            
            error_lower_quantile = np.quantile(errors, lower_quantile)
            error_upper_quantile = np.quantile(errors, upper_quantile)
            
            # Risk assessment
            thresholds = {
                "Low Risk": 0.02,  # 2%
                "Medium-Low Risk": 0.04,  # 4%
                "Medium Risk": 0.06,  # 6%
                "Medium-High Risk": 0.08,  # 8%
                "High Risk": float('inf')  # Greater than 8%
            }

            # Calculate relative volatility (volatility/average price)
            avg_price = np.mean(y_test_inv)
            relative_volatility = volatility / avg_price
            
            # Determine risk level
            risk_level = "Unknown"
            for level, threshold in thresholds.items():
                if relative_volatility <= threshold:
                    risk_level = level
                    break
            
            # Calculate direction accuracy (percentage of prediction direction matching actual direction)
            actual_direction = np.diff(y_test_inv) > 0
            predicted_direction = np.diff(y_pred_inv) > 0
            direction_accuracy = np.mean(actual_direction == predicted_direction) * 100
            
            # Calculate prediction bias (positive means predictions average higher than actual, negative means lower)
            bias = np.mean(y_pred_inv - y_test_inv)
            
            # Calculate maximum drawdown (maximum price drop percentage during test period)
            if df_original is not None:
                max_drawdown = self.calculate_max_drawdown(df_original['Close'])
            else:
                # Use test set to calculate
                max_drawdown = self.calculate_max_drawdown(y_test_inv)
            
            # Calculate risk-adjusted return (return/volatility)
            # Assume return is price change percentage during test period
            returns = (y_test_inv[-1] / y_test_inv[0] - 1) * 100
            risk_adjusted_return = returns / (np.std(y_test_inv) / np.mean(y_test_inv) * 100)
            
            logger.info(f"Risk assessment: Relative volatility: {relative_volatility:.4f}, Risk level: {risk_level}, "
                       f"Direction accuracy: {direction_accuracy:.2f}%, Bias: {bias:.4f}")
            
            # Build complete evaluation metrics
            metrics = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
                'mape': float(mape),
                'volatility': float(volatility),
                'relative_volatility': float(relative_volatility),
                'risk_level': risk_level,
                'direction_accuracy': float(direction_accuracy),
                'prediction_bias': float(bias),
                'max_drawdown': float(max_drawdown) if isinstance(max_drawdown, (int, float)) else float(0),
                'risk_adjusted_return': float(risk_adjusted_return),
                'confidence_interval': [float(error_lower_quantile), float(error_upper_quantile)]
            }
            
            return y_pred_inv, y_test_inv, metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None, None
    
    def calculate_max_drawdown(self, prices):
        """
        Calculate maximum drawdown
        
        Parameters:
            prices (array): Price series
            
        Returns:
            float: Maximum drawdown percentage
        """
        try:
            # Ensure input is numpy array
            prices = np.array(prices)
            
            # Calculate cumulative maximum
            max_prices = np.maximum.accumulate(prices)
            
            # Calculate drawdown at each point
            drawdowns = (max_prices - prices) / max_prices
            
            # Return maximum drawdown
            max_drawdown = np.max(drawdowns) * 100  # Convert to percentage
            return max_drawdown
        except Exception as e:
            logger.warning(f"Error calculating maximum drawdown: {e}")
            return 0.0
    
    def save_model_files(self, ticker, model, scaler, metrics=None, importance_df=None):
        """
        Save model, scaler and evaluation metrics
        
        Parameters:
            ticker (str): Stock ticker symbol
            model: Trained model
            scaler: Data scaler
            metrics (dict, optional): Evaluation metrics
            importance_df (DataFrame, optional): Feature importance
            
        Returns:
            bool: Whether successfully saved
        """
        if model is None or scaler is None:
            logger.error(f"Cannot save model files for {ticker}: Model or scaler is empty")
            return False
            
        logger.info(f"Saving model files for {ticker}...")
        try:
            # Create stock-specific directory
            ticker_dir = os.path.join(self.predictor.base_path, ticker)
            os.makedirs(ticker_dir, exist_ok=True)
            
            # Save model
            model_path = os.path.join(ticker_dir, f"{ticker}_lstm_model.h5")
            model.save(model_path)
            logger.info(f"Model saved to: {model_path}")
            
            # Save model summary
            with open(os.path.join(ticker_dir, f"{ticker}_model_summary.txt"), 'w') as f:
                model.summary(print_fn=lambda x: f.write(x + '\n'))
            
            # Save scaler
            scaler_path = os.path.join(ticker_dir, f"{ticker}_scaler.joblib")
            joblib.dump(scaler, scaler_path)
            logger.info(f"Scaler saved to: {scaler_path}")
            
            # Save evaluation metrics
            if metrics:
                import json
                metrics_path = os.path.join(ticker_dir, f"{ticker}_metrics.json")
                
                # Ensure all values are JSON serializable
                for key, value in metrics.items():
                    if isinstance(value, np.ndarray):
                        metrics[key] = value.tolist()
                    elif isinstance(value, np.float32) or isinstance(value, np.float64):
                        metrics[key] = float(value)
                
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f)
                logger.info(f"Metrics saved to: {metrics_path}")
            
            # Save feature importance
            if importance_df is not None:
                importance_path = os.path.join(ticker_dir, f"{ticker}_feature_importance.csv")
                importance_df.to_csv(importance_path, index=False)
                logger.info(f"Feature importance saved to: {importance_path}")
                
                # Store feature importance in instance variable
                self.predictor.feature_importance[ticker] = importance_df
            
            logger.info(f"✅ Model files for {ticker} successfully saved")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model files for {ticker}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def load_model_files(self, ticker):
        """
        Load model and scaler for specific stock
        
        Parameters:
            ticker (str): Stock ticker symbol
            
        Returns:
            tuple: (model, scaler) Loaded model and scaler
        """
        logger.info(f"Loading model files for {ticker}...")
        ticker_dir = os.path.join(self.predictor.base_path, ticker)
        
        try:
            # Check if directory exists
            if not os.path.exists(ticker_dir):
                logger.info(f"Model directory for {ticker} does not exist: {ticker_dir}")
                return None, None
            
            # Load model
            model_path = os.path.join(ticker_dir, f"{ticker}_lstm_model.h5")
            if not os.path.exists(model_path):
                logger.warning(f"Model file for {ticker} does not exist: {model_path}")
                return None, None
                
            # Try to load model
            for _ in range(3):  # Try up to 3 times
                try:
                    model = load_model(model_path)
                    logger.info(f"Successfully loaded model: {model_path}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load model, retrying: {e}")
                    time.sleep(1)
            else:
                logger.error(f"Could not load model after multiple attempts: {model_path}")
                return None, None
            
            # Load scaler
            scaler_path = os.path.join(ticker_dir, f"{ticker}_scaler.joblib")
            if not os.path.exists(scaler_path):
                logger.warning(f"Scaler file for {ticker} does not exist: {scaler_path}")
                return model, None
                
            # Try to load scaler
            for _ in range(3):  # Try up to 3 times
                try:
                    scaler = joblib.load(scaler_path)
                    logger.info(f"Successfully loaded scaler: {scaler_path}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load scaler, retrying: {e}")
                    time.sleep(1)
            else:
                logger.error(f"Could not load scaler after multiple attempts: {scaler_path}")
                return model, None
            
            logger.info(f"✅ Model files for {ticker} successfully loaded")
            
            # Store model and scaler in instance variables
            self.predictor.models_dict[ticker] = model
            self.predictor.scaler_dict[ticker] = scaler
            
            return model, scaler
            
        except Exception as e:
            logger.error(f"Error loading model files for {ticker}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None

    def predict_future(self, ticker, days=30, include_confidence_intervals=True):
        """
        Predict future N days of stock prices
        
        Parameters:
            ticker (str): Stock ticker symbol
            days (int): Number of days to predict
            include_confidence_intervals (bool): Whether to include confidence intervals
            
        Returns:
            DataFrame: Prediction results
        """
        logger.info(f"Predicting future {days} days for {ticker}...")
        try:
            # Load model and scaler
            model, scaler = self.load_model_files(ticker)
            if model is None or scaler is None:
                logger.error(f"Cannot predict future prices for {ticker}: Failed to load model or scaler")
                return None
                
            # Get latest data (use longer time period to ensure enough data)
            df = self.predictor.data_processor.get_stock_data(ticker, period="1y")
            if df is None or df.empty:
                logger.error(f"Cannot predict future prices for {ticker}: Unable to get latest data")
                return None
                
            # Get last actual close price for later comparison
            last_actual_close = df['Close'].iloc[-1]
            logger.info(f"Last actual close price: {last_actual_close}")
            
            # Add technical indicators
            df = self.predictor.data_processor.add_technical_indicators(df)
            if df is None or df.empty:
                logger.error(f"Cannot predict future prices for {ticker}: Failed to add technical indicators")
                return None
            
            # Ensure data format is correct
            df.columns = [str(col) for col in df.columns]
            
            # Get model's expected feature count
            expected_features = model.input_shape[2]
            
            # Handle feature count mismatch
            if df.shape[1] != expected_features:
                logger.warning(f"Feature count mismatch: Current {df.shape[1]}, Expected {expected_features}")
                
                if df.shape[1] > expected_features:
                    # If we have more features, need to select features matching the model
                    # Load feature importance (if available)
                    importance_path = os.path.join(self.predictor.base_path, ticker, f"{ticker}_feature_importance.csv")
                    
                    if os.path.exists(importance_path):
                        feature_importance = pd.read_csv(importance_path)
                        top_features = feature_importance['Feature'].head(expected_features).tolist()
                        
                        # Ensure 'Close' is included in selected features
                        if 'Close' not in top_features and 'Close' in df.columns:
                            top_features[-1] = 'Close'  # Replace last feature
                        
                        logger.info(f"Selecting top {expected_features} features based on feature importance")
                        df = df[top_features]
                    else:
                        # If no feature importance info, retain original OHLCV and some basic technical indicators
                        basic_features = ['Open', 'High', 'Low', 'Close', 'Volume']
                        additional_features = [col for col in df.columns if col not in basic_features]
                        
                        # Add extra features as needed to reach expected feature count
                        features_to_use = basic_features + additional_features[:expected_features - len(basic_features)]
                        
                        logger.info(f"Selecting basic features and additional technical indicators, total {expected_features} features")
                        df = df[features_to_use[:expected_features]]
                        
                elif df.shape[1] < expected_features:
                    logger.error(f"Current feature count ({df.shape[1]}) less than model expects ({expected_features}), cannot predict")
                    return None
            
            # Use scaler to transform data
            try:
                df_scaled = scaler.transform(df.values)
            except Exception as e:
                logger.error(f"Error scaling data: {e}")
                return None
            
            # Prepare latest sequence
            if len(df_scaled) < self.predictor.sequence_length:
                logger.error(f"Insufficient data points, need at least {self.predictor.sequence_length} but only have {len(df_scaled)}")
                return None
                
            last_sequence = df_scaled[-self.predictor.sequence_length:]
            
            # Predict future N days
            future_prices = []
            future_lower_bounds = []
            future_upper_bounds = []
            
            # Get model evaluation metrics for confidence interval
            metrics_path = os.path.join(self.predictor.base_path, ticker, f"{ticker}_metrics.json")
            if os.path.exists(metrics_path) and include_confidence_intervals:
                with open(metrics_path, 'r') as f:
                    import json
                    metrics = json.load(f)
                confidence_interval = metrics.get('confidence_interval', [-5, 5])
                prediction_error = metrics.get('volatility', df['Close'].std() * 0.1)
            else:
                # If no confidence interval available, use default
                confidence_interval = [-5, 5]
                prediction_error = df['Close'].std() * 0.1
            
            # Stability factor - increases with prediction days to reflect growing uncertainty
            stability_factor = 1.0
            
            # Start sequence prediction
            current_sequence = last_sequence.copy()
            
            for day in range(days):
                # Reshape sequence for model input
                current_sequence_reshaped = current_sequence.reshape(1, self.predictor.sequence_length, df.shape[1])
                
                # Use model to predict next value
                next_pred = model.predict(current_sequence_reshaped, verbose=0)[0, 0]
                
                # Prepare array for inverse scaling
                next_pred_for_inverse = np.zeros((1, df.shape[1]))
                next_pred_for_inverse[0, 0] = next_pred  # Assume first column is Close price
                
                # Inverse scale to get actual price
                try:
                    next_price = scaler.inverse_transform(next_pred_for_inverse)[0, 0]
                except Exception as e:
                    logger.error(f"Error inverse scaling: {e}")
                    return None
                
                # Apply prediction stability constraints - limit daily price changes
                if day == 0:
                    # First day prediction, use last known price as reference
                    prev_price = last_actual_close
                else:
                    prev_price = future_prices[-1]
                
                # Calculate reasonable price change range
                # Use adaptive scale based on stock's historical volatility
                historical_volatility = df['Close'].pct_change().std() * 100  # Historical daily volatility (%)
                
                # Limit daily maximum change to 2x historical volatility but not more than 8%
                max_daily_change_pct = min(historical_volatility * 2, 8.0)
                
                # As prediction days increase, gradually relax constraints to reflect uncertainty
                adaptive_max_change_pct = max_daily_change_pct * (1 + day * 0.05)
                max_allowed_change = prev_price * (adaptive_max_change_pct / 100)
                
                # Define reasonable price change range
                min_price = prev_price * (1 - adaptive_max_change_pct / 100)
                max_price = prev_price * (1 + adaptive_max_change_pct / 100)
                
                # Adjust predicted price
                if next_price > max_price:
                    next_price = max_price
                    logger.info(f"Day {day+1} predicted price exceeds reasonable upper limit, adjusted")
                elif next_price < min_price:
                    next_price = min_price
                    logger.info(f"Day {day+1} predicted price below reasonable lower limit, adjusted")
                
                # Calculate confidence intervals
                # Gradually widen confidence interval as days increase to reflect uncertainty
                stability_factor = 1.0 + day * 0.05  # Increase by 5% each day
                lower_bound = next_price + confidence_interval[0] * prediction_error * stability_factor
                upper_bound = next_price + confidence_interval[1] * prediction_error * stability_factor
                
                # Store prediction results
                future_prices.append(next_price)
                future_lower_bounds.append(lower_bound)
                future_upper_bounds.append(upper_bound)
                
                # Update sequence (remove first element, add new prediction)
                # Note: We only update Close column value, other columns remain unchanged
                new_values = current_sequence[-1].copy()
                new_values[0] = next_pred  # Update Close column prediction
                
                # For some technical indicators, we could try to update more column values
                current_sequence = np.vstack([current_sequence[1:], [new_values]])
            
            # Generate future dates
            last_date = df.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
            
            # Create prediction dataframe
            future_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted_Close': future_prices,
                'Lower_Bound': future_lower_bounds,
                'Upper_Bound': future_upper_bounds
            })
            future_df.set_index('Date', inplace=True)
            
            logger.info(f"Future {days} day price prediction complete for {ticker}, range: {future_df['Predicted_Close'].min():.2f} - {future_df['Predicted_Close'].max():.2f}")
            return future_df
            
        except Exception as e:
            logger.error(f"Error predicting future prices for {ticker}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None