import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import seaborn as sns
import numpy as np
import pandas as pd
import os
import datetime
import logging
import scipy.stats as stats

logger = logging.getLogger("StockPredictor.Visualizer")

class Visualizer:
    def __init__(self, predictor):
        """
        Initialize with reference to parent predictor
        
        Parameters:
            predictor: Parent StockPricePredictor instance
        """
        self.predictor = predictor
    
    def visualize_results(self, ticker, test_dates, y_test, y_pred, metrics=None, history=None, future_df=None):
        """
        Visualize prediction results with professional analysis charts
        
        Parameters:
            ticker (str): Stock ticker symbol
            test_dates (array): Test dates
            y_test (array): Actual values
            y_pred (array): Predicted values
            metrics (dict, optional): Evaluation metrics
            history (History, optional): Training history
            future_df (DataFrame, optional): Future predictions
        """
        logger.info(f"Generating visualization results for {ticker}...")
        try:
            # Set plot style
            plt.style.use('seaborn-v0_8-whitegrid')
            
            # Create canvas
            fig = plt.figure(figsize=(15, 12))
            fig.suptitle(f'{ticker} Stock Price Prediction & Analysis', fontsize=22, fontweight='bold', y=0.98)
            
            # Define subplot grid
            gs = plt.GridSpec(3, 2, figure=fig, height_ratios=[2, 1, 1], hspace=0.4, wspace=0.3)
            
            # First subplot: Stock price prediction (occupies top row)
            ax1 = fig.add_subplot(gs[0, :])
            
            # Plot actual stock price
            ax1.plot(test_dates, y_test, label='Actual Price', color='#1f77b4', linewidth=2)
            
            # Plot predicted stock price
            ax1.plot(test_dates, y_pred, label='Predicted Price', color='#ff7f0e', linestyle='--', linewidth=2)
            
            # If future predictions available, add to chart
            if future_df is not None:
                # Plot future predictions
                ax1.plot(future_df.index, future_df['Predicted_Close'], 
                         label='Future Forecast', color='#2ca02c', linestyle='-.', linewidth=2)
                
                # Plot confidence interval
                if 'Lower_Bound' in future_df.columns and 'Upper_Bound' in future_df.columns:
                    ax1.fill_between(future_df.index, future_df['Lower_Bound'], future_df['Upper_Bound'],
                                    color='#2ca02c', alpha=0.2, label='Prediction Confidence Interval')
                
                # Add vertical line separating historical data and predictions
                ax1.axvline(x=test_dates[-1], color='gray', linestyle='--', alpha=0.7)
                ax1.text(test_dates[-1], ax1.get_ylim()[1] * 0.98, ' Prediction Start', 
                         verticalalignment='top', horizontalalignment='left', fontsize=10)
            
            # Set chart title and labels
            ax1.set_title('Stock Price: Actual vs Predicted', fontsize=18, pad=15)
            ax1.set_xlabel('Date', fontsize=14)
            ax1.set_ylabel('Price ($)', fontsize=14)
            
            # Configure legend
            ax1.legend(loc='best', fontsize=12, framealpha=0.9)
            
            # Beautify x-axis date labels
            ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            ax1.xaxis.set_major_locator(plt.MaxNLocator(10))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Add grid lines
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Second subplot: Loss curve (bottom left)
            ax2 = fig.add_subplot(gs[1, 0])
            if history is not None:
                ax2.plot(history.history['loss'], label='Training Loss', color='#1f77b4')
                ax2.plot(history.history['val_loss'], label='Validation Loss', color='#d62728')
                ax2.set_title('Model Training Loss Curve', fontsize=16)
                ax2.set_xlabel('Epoch', fontsize=12)
                ax2.set_ylabel('Loss (MSE)', fontsize=12)
                ax2.legend(loc='upper right', fontsize=10)
                ax2.grid(True, linestyle='--', alpha=0.7)
                
                # Find best validation loss and mark it
                best_epoch = np.argmin(history.history['val_loss'])
                best_val_loss = history.history['val_loss'][best_epoch]
                ax2.scatter(best_epoch, best_val_loss, c='red', s=100, zorder=10)
                ax2.annotate(f'Best: {best_val_loss:.4f}', 
                            xy=(best_epoch, best_val_loss),
                            xytext=(best_epoch + 5, best_val_loss * 1.1),
                            arrowprops=dict(arrowstyle='->'), fontsize=10)
            else:
                ax2.text(0.5, 0.5, 'Training history unavailable', ha='center', va='center', fontsize=14)
                ax2.set_title('Model Training Loss Curve', fontsize=16)
            
            # Third subplot: Prediction error analysis (bottom right)
            ax3 = fig.add_subplot(gs[1, 1])
            if y_test is not None and y_pred is not None:
                errors = y_pred - y_test
                ax3.scatter(y_test, errors, alpha=0.6, color='#1f77b4')
                ax3.axhline(y=0, color='r', linestyle='--')
                ax3.set_title('Prediction Error Analysis', fontsize=16)
                ax3.set_xlabel('Actual Price', fontsize=12)
                ax3.set_ylabel('Prediction Error', fontsize=12)
                ax3.grid(True, linestyle='--', alpha=0.7)
                
                # Add trend line
                z = np.polyfit(y_test, errors, 1)
                p = np.poly1d(z)
                ax3.plot(np.sort(y_test), p(np.sort(y_test)), "r--", alpha=0.8, 
                         label=f'Trend: y={z[0]:.2e}x+{z[1]:.2f}')
                ax3.legend(loc='best', fontsize=10)
            
            # Fourth subplot: Prediction error distribution (bottom left)
            ax4 = fig.add_subplot(gs[2, 0])
            if y_test is not None and y_pred is not None:
                # Plot error histogram and KDE
                sns.histplot(errors, kde=True, ax=ax4, color='#1f77b4', bins=30)
                ax4.set_title('Prediction Error Distribution', fontsize=16)
                ax4.set_xlabel('Prediction Error', fontsize=12)
                ax4.set_ylabel('Frequency', fontsize=12)
                
                # Add normal distribution fit curve
                mu, std = stats.norm.fit(errors)
                xmin, xmax = ax4.get_xlim()
                x = np.linspace(xmin, xmax, 100)
                p = stats.norm.pdf(x, mu, std)
                scale_factor = len(errors) * (xmax - xmin) / 30  # Adjust scaling to match histogram
                ax4.plot(x, p * scale_factor, 'r--', linewidth=2, 
                         label=f'Normal Distribution (μ={mu:.2f}, σ={std:.2f})')
                ax4.legend(loc='best', fontsize=10)
            
            # Fifth subplot: Performance metrics (bottom right)
            ax5 = fig.add_subplot(gs[2, 1])
            if metrics is not None:
                # Create performance metrics table
                metrics_to_show = {
                    'RMSE': f"{metrics.get('rmse', 0):.4f}",
                    'MAE': f"{metrics.get('mae', 0):.4f}",
                    'MAPE': f"{metrics.get('mape', 0):.2f}%",
                    'R²': f"{metrics.get('r2', 0):.4f}",
                    'Direction Accuracy': f"{metrics.get('direction_accuracy', 0):.2f}%",
                    'Relative Volatility': f"{metrics.get('relative_volatility', 0):.4f}",
                    'Maximum Drawdown': f"{metrics.get('max_drawdown', 0):.2f}%",
                    'Risk-adjusted Return': f"{metrics.get('risk_adjusted_return', 0):.4f}",
                    'Risk Level': f"{metrics.get('risk_level', 'N/A')}"
                }
                
                # Draw table
                ax5.axis('off')
                table_data = [[k, v] for k, v in metrics_to_show.items()]
                table = ax5.table(cellText=table_data, colLabels=['Metric', 'Value'], 
                                 loc='center', cellLoc='left', colWidths=[0.5, 0.5])
                table.auto_set_font_size(False)
                table.set_fontsize(11)
                table.scale(1, 1.5)
                
                # Set table title
                ax5.set_title('Model Performance Metrics', fontsize=16, pad=20)
                
                # Add risk assessment color indicator
                risk_level = metrics.get('risk_level', 'N/A')
                risk_colors = {
                    'Low Risk': '#4daf4a',      # Green
                    'Medium-Low Risk': '#80c34d',    # Light green
                    'Medium Risk': '#ffce4f',      # Yellow
                    'Medium-High Risk': '#ff9b4f',    # Orange
                    'High Risk': '#d53e4f'       # Red
                }
                
                # Find row with risk level
                for i, row in enumerate(table_data):
                    if row[0] == 'Risk Level':
                        risk_row = i + 1  # +1 because table has header row
                        # Set risk level cell color
                        table[(risk_row, 1)].set_facecolor(risk_colors.get(risk_level, '#ffffff'))
                        # Adjust risk level cell text color to white or black
                        if risk_level in ['Medium-High Risk', 'High Risk']:
                            table[(risk_row, 1)].get_text().set_color('white')
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save chart
            ticker_dir = os.path.join(self.predictor.base_path, ticker)
            os.makedirs(ticker_dir, exist_ok=True)
            plot_path = os.path.join(ticker_dir, f"{ticker}_prediction_analysis.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Analysis chart saved to: {plot_path}")
            
            # Show chart
            # plt.show()
            
            logger.info(f"Visualization results generation complete for {ticker}")
            
        except Exception as e:
            logger.error(f"Error generating visualization results for {ticker}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def visualize_stock_comparison(self, comparison_df, future_predictions, days_to_predict):
        """
        Visualize stock comparison results
        
        Parameters:
            comparison_df (DataFrame): Comparison results dataframe
            future_predictions (dict): Mapping of ticker to future predictions
            days_to_predict (int): Number of days predicted
        """
        try:
            plt.figure(figsize=(16, 10))
            
            # Set style
            plt.style.use('seaborn-v0_8-whitegrid')
            
            # Create subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [1.2, 1]})
            
            # Sort data for plotting
            plot_df = comparison_df.sort_values('Expected Return (%)')
            
            # Plot expected return bar chart
            bars = ax1.barh(plot_df['Ticker'], plot_df['Expected Return (%)'], color='skyblue')
            
            # Set different colors for positive/negative returns
            for i, bar in enumerate(bars):
                if bar.get_width() < 0:
                    bar.set_color('lightcoral')
            
            # Add value labels
            for i, v in enumerate(plot_df['Expected Return (%)']):
                text_color = 'black'
                if abs(v) > 20:  # For very large values, white is more readable
                    text_color = 'white'
                ax1.text(v + (1 if v >= 0 else -1), i, f'{v:.2f}%', va='center', color=text_color, fontweight='bold')
            
            # Set title and labels
            ax1.set_title(f'Expected Return Comparison over {days_to_predict} Days', fontsize=16)
            ax1.set_xlabel('Expected Return (%)', fontsize=12)
            ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Plot future price trend comparison
            # Select top 5 stocks by expected return
            top_stocks = comparison_df.sort_values('Expected Return (%)', ascending=False).head(5)['Ticker'].tolist()
            
            for ticker in top_stocks:
                if ticker in future_predictions:
                    # Normalize prices, starting value = 100
                    initial_price = future_predictions[ticker]['Predicted_Close'].iloc[0]
                    normalized_prices = future_predictions[ticker]['Predicted_Close'] / initial_price * 100
                    
                    # Plot price line
                    ax2.plot(future_predictions[ticker].index, normalized_prices, 
                             label=f'{ticker} ({comparison_df[comparison_df["Ticker"] == ticker]["Expected Return (%)"].values[0]:.2f}%)')
            
            # Set title and labels
            ax2.set_title(f'Top 5 Stocks Normalized Price Trends over {days_to_predict} Days (Start=100)', fontsize=16)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('Normalized Price', fontsize=12)
            ax2.legend(loc='best')
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # Format dates
            ax2.xaxis.set_major_formatter(DateFormatter('%m-%d'))
            plt.xticks(rotation=45)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save chart
            os.makedirs(os.path.join(self.predictor.base_path, 'comparisons'), exist_ok=True)
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_path = os.path.join(self.predictor.base_path, 'comparisons', f'stock_comparison_{timestamp}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Stock comparison chart saved to: {plot_path}")
            
            # Show chart
            # plt.show()
            
        except Exception as e:
            logger.error(f"Error visualizing stock comparison: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def visualize_backtest(self, ticker, backtest_df, prediction_window):
        """
        Visualize backtest results
        
        Parameters:
            ticker (str): Stock ticker symbol
            backtest_df (DataFrame): Backtest results
            prediction_window (int): Prediction window size
        """
        try:
            plt.figure(figsize=(16, 12))
            
            # Set style
            plt.style.use('seaborn-v0_8-whitegrid')
            
            # Create subplots
            fig = plt.figure(figsize=(16, 12))
            gs = plt.GridSpec(3, 2, figure=fig, height_ratios=[2, 1, 1], hspace=0.4, wspace=0.3)
            
            # 1. Prediction error over time
            ax1 = fig.add_subplot(gs[0, :])
            ax1.plot(pd.to_datetime(backtest_df['Backtest Date']), backtest_df['Prediction Error (%)'], 
                    marker='o', linestyle='-', color='#1f77b4')
            ax1.axhline(y=0, color='r', linestyle='--', alpha=0.7)
            ax1.set_title(f'{ticker} - Prediction Error Over Time ({prediction_window}-day window)', fontsize=16)
            ax1.set_xlabel('Backtest Date', fontsize=12)
            ax1.set_ylabel('Prediction Error (%)', fontsize=12)
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Add mean error line
            mean_error = backtest_df['Prediction Error (%)'].mean()
            ax1.axhline(y=mean_error, color='g', linestyle='-.', alpha=0.7, 
                       label=f'Mean Error: {mean_error:.2f}%')
            
            # Add direction accuracy annotation
            direction_accuracy = backtest_df['Direction Correct'].mean() * 100
            ax1.text(0.02, 0.05, f'Direction Accuracy: {direction_accuracy:.2f}%', 
                    transform=ax1.transAxes, fontsize=12, 
                    bbox=dict(facecolor='white', alpha=0.8))
            
            ax1.legend(loc='best')
            
            # 2. Return comparison (actual vs predicted)
            ax2 = fig.add_subplot(gs[1, 0])
            backtest_df_sorted = backtest_df.sort_values('Backtest Date')
            ax2.plot(pd.to_datetime(backtest_df_sorted['Backtest Date']), 
                    backtest_df_sorted['Actual Return (%)'], 
                    marker='o', linestyle='-', color='#ff7f0e', label='Actual Return')
            ax2.plot(pd.to_datetime(backtest_df_sorted['Backtest Date']), 
                    backtest_df_sorted['Predicted Return (%)'], 
                    marker='x', linestyle='--', color='#1f77b4', label='Predicted Return')
            ax2.axhline(y=0, color='r', linestyle='--', alpha=0.7)
            ax2.set_title('Actual Return vs Predicted Return', fontsize=16)
            ax2.set_xlabel('Backtest Date', fontsize=12)
            ax2.set_ylabel('Return (%)', fontsize=12)
            ax2.legend(loc='best')
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # 3. Error distribution
            ax3 = fig.add_subplot(gs[1, 1])
            sns.histplot(backtest_df['Prediction Error (%)'], kde=True, bins=20, ax=ax3, color='#1f77b4')
            ax3.axvline(x=0, color='r', linestyle='--', alpha=0.7)
            ax3.axvline(x=mean_error, color='g', linestyle='-.', alpha=0.7, 
                       label=f'Mean Error: {mean_error:.2f}%')
            ax3.set_title('Prediction Error Distribution', fontsize=16)
            ax3.set_xlabel('Prediction Error (%)', fontsize=12)
            ax3.set_ylabel('Frequency', fontsize=12)
            ax3.legend(loc='best')
            
            # 4. Scatter plot: Actual price vs Predicted price
            ax4 = fig.add_subplot(gs[2, 0])
            ax4.scatter(backtest_df[f'Actual Price ({prediction_window} days)'], 
                       backtest_df[f'Predicted Price ({prediction_window} days)'], 
                       alpha=0.7, color='#1f77b4')
            
            # Add diagonal line (perfect prediction line)
            min_val = min(backtest_df[f'Actual Price ({prediction_window} days)'].min(), backtest_df[f'Predicted Price ({prediction_window} days)'].min())
            max_val = max(backtest_df[f'Actual Price ({prediction_window} days)'].max(), backtest_df[f'Predicted Price ({prediction_window} days)'].max())
            ax4.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
            
            ax4.set_title('Actual Price vs Predicted Price', fontsize=16)
            ax4.set_xlabel(f'Actual Price ({prediction_window} days)', fontsize=12)
            ax4.set_ylabel(f'Predicted Price ({prediction_window} days)', fontsize=12)
            ax4.grid(True, linestyle='--', alpha=0.7)
            
            # 5. Direction accuracy over time
            ax5 = fig.add_subplot(gs[2, 1])
            # Calculate rolling direction accuracy
            window_size = min(5, len(backtest_df) // 2)  # Need at least 2 points
            if window_size >= 2:
                rolling_accuracy = backtest_df['Direction Correct'].rolling(window=window_size).mean() * 100
                ax5.plot(pd.to_datetime(backtest_df['Backtest Date']), 
                        rolling_accuracy, 
                        marker='o', linestyle='-', color='#2ca02c')
                ax5.axhline(y=50, color='r', linestyle='--', alpha=0.7, label='Random Guess (50%)')
                ax5.axhline(y=direction_accuracy, color='g', linestyle='-.', alpha=0.7, 
                           label=f'Average Accuracy: {direction_accuracy:.2f}%')
                ax5.set_title(f'Direction Accuracy Trend ({window_size}-period rolling window)', fontsize=16)
                ax5.set_xlabel('Backtest Date', fontsize=12)
                ax5.set_ylabel('Direction Accuracy (%)', fontsize=12)
                ax5.legend(loc='best')
                ax5.grid(True, linestyle='--', alpha=0.7)
            else:
                ax5.text(0.5, 0.5, 'Insufficient data points to calculate rolling accuracy', 
                        ha='center', va='center', fontsize=14)
                ax5.set_title('Direction Accuracy Trend', fontsize=16)
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Add overall title
            plt.suptitle(f'{ticker} - Backtest Analysis Report ({len(backtest_df)} backtest points, {prediction_window}-day window)', 
                        fontsize=20, y=0.98)
            
            # Save chart
            os.makedirs(os.path.join(self.predictor.base_path, 'backtests'), exist_ok=True)
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_path = os.path.join(self.predictor.base_path, 'backtests', f'{ticker}_backtest_{timestamp}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Backtest analysis chart for {ticker} saved to: {plot_path}")
            
            # Show chart
            # plt.show()
            
        except Exception as e:
            logger.error(f"Error visualizing backtest results for {ticker}: {e}")
            import traceback
            logger.error(traceback.format_exc())
