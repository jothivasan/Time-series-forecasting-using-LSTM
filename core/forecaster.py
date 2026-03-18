"""
Forecasting Engine for Multi-Domain LSTM System
Handles short-term, mid-term, and long-term forecasting
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class ForecastingEngine:
    """
    Advanced forecasting engine with multi-horizon support
    """
    
    def __init__(self, model, data_processor):
        """
        Initialize forecasting engine
        
        Args:
            model: Trained LSTM model
            data_processor: Data processor instance
        """
        self.model = model
        self.data_processor = data_processor
        self.forecasts = {}
    
    def forecast_short_term(self, initial_sequence, steps=3):
        """
        Short-term forecasting (1-3 steps ahead)
        
        Args:
            initial_sequence (np.array): Initial sequence
            steps (int): Number of steps to forecast
            
        Returns:
            dict: Forecast results
        """
        predictions = self.model.forecast_multi_step(initial_sequence, steps)
        
        # Inverse transform to original scale
        predictions_original = self.data_processor.inverse_transform(predictions)
        
        result = {
            'horizon': 'short_term',
            'steps': steps,
            'predictions': predictions_original.tolist(),
            'scaled_predictions': predictions.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        self.forecasts['short_term'] = result
        return result
    
    def forecast_mid_term(self, initial_sequence, steps=12):
        """
        Mid-term forecasting (4-12 steps ahead)
        
        Args:
            initial_sequence (np.array): Initial sequence
            steps (int): Number of steps to forecast
            
        Returns:
            dict: Forecast results
        """
        predictions = self.model.forecast_multi_step(initial_sequence, steps)
        
        # Inverse transform to original scale
        predictions_original = self.data_processor.inverse_transform(predictions)
        
        result = {
            'horizon': 'mid_term',
            'steps': steps,
            'predictions': predictions_original.tolist(),
            'scaled_predictions': predictions.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        self.forecasts['mid_term'] = result
        return result
    
    def forecast_long_term(self, initial_sequence, steps=24):
        """
        Long-term forecasting (13+ steps ahead)
        
        Args:
            initial_sequence (np.array): Initial sequence
            steps (int): Number of steps to forecast
            
        Returns:
            dict: Forecast results
        """
        predictions = self.model.forecast_multi_step(initial_sequence, steps)
        
        # Inverse transform to original scale
        predictions_original = self.data_processor.inverse_transform(predictions)
        
        result = {
            'horizon': 'long_term',
            'steps': steps,
            'predictions': predictions_original.tolist(),
            'scaled_predictions': predictions.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        self.forecasts['long_term'] = result
        return result
    
    def forecast_custom_horizon(self, initial_sequence, steps):
        """
        Custom horizon forecasting
        
        Args:
            initial_sequence (np.array): Initial sequence
            steps (int): Number of steps to forecast
            
        Returns:
            dict: Forecast results
        """
        predictions = self.model.forecast_multi_step(initial_sequence, steps)
        
        # Inverse transform to original scale
        predictions_original = self.data_processor.inverse_transform(predictions)
        
        result = {
            'horizon': 'custom',
            'steps': steps,
            'predictions': predictions_original.tolist(),
            'scaled_predictions': predictions.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def forecast_with_confidence_intervals(self, initial_sequence, steps, confidence=0.95):
        """
        Forecast with confidence intervals (simplified version)
        
        Args:
            initial_sequence (np.array): Initial sequence
            steps (int): Number of steps to forecast
            confidence (float): Confidence level
            
        Returns:
            dict: Forecast with confidence intervals
        """
        # Get point predictions
        predictions = self.model.forecast_multi_step(initial_sequence, steps)
        predictions_original = self.data_processor.inverse_transform(predictions)
        
        # Simple confidence interval estimation (can be improved with Monte Carlo)
        std_estimate = np.std(predictions_original) * 1.5
        z_score = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
        
        margin = z_score * std_estimate
        
        lower_bound = predictions_original - margin
        upper_bound = predictions_original + margin
        
        result = {
            'horizon': 'custom',
            'steps': steps,
            'predictions': predictions_original.tolist(),
            'lower_bound': lower_bound.tolist(),
            'upper_bound': upper_bound.tolist(),
            'confidence_level': confidence,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def generate_forecast_dates(self, last_date, steps, frequency='D'):
        """
        Generate future dates for forecasts
        
        Args:
            last_date (datetime): Last date in the dataset
            steps (int): Number of steps to forecast
            frequency (str): Frequency ('D', 'M', 'H', etc.)
            
        Returns:
            list: List of forecast dates
        """
        if isinstance(last_date, str):
            last_date = pd.to_datetime(last_date)
        
        if frequency == 'D':
            dates = [last_date + timedelta(days=i+1) for i in range(steps)]
        elif frequency == 'M':
            dates = pd.date_range(start=last_date, periods=steps+1, freq='M')[1:]
        elif frequency == 'H':
            dates = [last_date + timedelta(hours=i+1) for i in range(steps)]
        elif frequency == 'W':
            dates = [last_date + timedelta(weeks=i+1) for i in range(steps)]
        else:
            dates = [last_date + timedelta(days=i+1) for i in range(steps)]
        
        return [str(d) for d in dates]
    
    def create_forecast_dataframe(self, forecast_result, start_date, frequency='D'):
        """
        Create a pandas DataFrame from forecast results
        
        Args:
            forecast_result (dict): Forecast result dictionary
            start_date (datetime): Starting date for forecast
            frequency (str): Frequency of data
            
        Returns:
            pd.DataFrame: Forecast dataframe
        """
        steps = forecast_result['steps']
        predictions = forecast_result['predictions']
        
        dates = self.generate_forecast_dates(start_date, steps, frequency)
        
        df = pd.DataFrame({
            'date': dates,
            'forecast': [p[0] if isinstance(p, list) else p for p in predictions]
        })
        
        if 'lower_bound' in forecast_result:
            df['lower_bound'] = [p[0] if isinstance(p, list) else p for p in forecast_result['lower_bound']]
            df['upper_bound'] = [p[0] if isinstance(p, list) else p for p in forecast_result['upper_bound']]
        
        return df
    
    def compare_forecast_horizons(self, initial_sequence, horizons=[3, 12, 24]):
        """
        Compare forecasts across different horizons
        
        Args:
            initial_sequence (np.array): Initial sequence
            horizons (list): List of forecast horizons
            
        Returns:
            dict: Comparison results
        """
        comparison = {}
        
        for horizon in horizons:
            if horizon <= 3:
                result = self.forecast_short_term(initial_sequence, horizon)
            elif horizon <= 12:
                result = self.forecast_mid_term(initial_sequence, horizon)
            else:
                result = self.forecast_long_term(initial_sequence, horizon)
            
            comparison[f'horizon_{horizon}'] = result
        
        return comparison
    
    def get_all_forecasts(self):
        """Get all generated forecasts"""
        return self.forecasts
    
    def export_forecasts(self, filepath):
        """
        Export forecasts to JSON
        
        Args:
            filepath (str): Path to save forecasts
        """
        import json
        
        with open(filepath, 'w') as f:
            json.dump(self.forecasts, f, indent=2)
        
        print(f"Forecasts exported to: {filepath}")
