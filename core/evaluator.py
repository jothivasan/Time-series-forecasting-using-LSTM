"""
Performance Evaluator for Multi-Domain LSTM Forecasting
Provides comprehensive evaluation metrics and domain-wise comparison
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class PerformanceEvaluator:
    """
    Evaluate LSTM model performance across different domains
    """
    
    def __init__(self):
        """Initialize evaluator"""
        self.evaluation_results = {}
        self.domain_comparisons = {}
    
    def calculate_metrics(self, y_true, y_pred, domain_name=None):
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            y_true (np.array): True values
            y_pred (np.array): Predicted values
            domain_name (str): Name of the domain
            
        Returns:
            dict: Evaluation metrics
        """
        # Flatten arrays
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        
        # R² Score
        r2 = r2_score(y_true, y_pred)
        
        # Additional metrics
        # Mean Percentage Error (MPE)
        mpe = np.mean((y_true - y_pred) / (y_true + 1e-10)) * 100
        
        # Normalized RMSE
        nrmse = rmse / (np.max(y_true) - np.min(y_true) + 1e-10)
        
        # Symmetric MAPE
        smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)) * 100
        
        # Directional Accuracy (for trend prediction)
        if len(y_true) > 1:
            true_direction = np.sign(np.diff(y_true))
            pred_direction = np.sign(np.diff(y_pred))
            directional_accuracy = np.mean(true_direction == pred_direction) * 100
        else:
            directional_accuracy = 0
        
        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2_score': float(r2),
            'mpe': float(mpe),
            'nrmse': float(nrmse),
            'smape': float(smape),
            'directional_accuracy': float(directional_accuracy)
        }
        
        # Store results
        if domain_name:
            self.evaluation_results[domain_name] = metrics
        
        return metrics
    
    def calculate_error_distribution(self, y_true, y_pred):
        """
        Calculate error distribution statistics
        
        Args:
            y_true (np.array): True values
            y_pred (np.array): Predicted values
            
        Returns:
            dict: Error distribution metrics
        """
        errors = y_true - y_pred
        
        distribution = {
            'mean_error': float(np.mean(errors)),
            'std_error': float(np.std(errors)),
            'min_error': float(np.min(errors)),
            'max_error': float(np.max(errors)),
            'median_error': float(np.median(errors)),
            'q25_error': float(np.percentile(errors, 25)),
            'q75_error': float(np.percentile(errors, 75))
        }
        
        return distribution
    
    def compare_domains(self, domain_results):
        """
        Compare performance across multiple domains
        
        Args:
            domain_results (dict): Dictionary of domain-wise results
                                  {domain_name: {'y_true': [...], 'y_pred': [...]}}
            
        Returns:
            pd.DataFrame: Comparison table
        """
        comparison_data = []
        
        for domain_name, results in domain_results.items():
            metrics = self.calculate_metrics(
                results['y_true'],
                results['y_pred'],
                domain_name
            )
            
            comparison_data.append({
                'Domain': domain_name,
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'MAPE': metrics['mape'],
                'R² Score': metrics['r2_score'],
                'Directional Accuracy': metrics['directional_accuracy']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        self.domain_comparisons = comparison_df
        
        return comparison_df
    
    def rank_domains_by_performance(self, metric='rmse'):
        """
        Rank domains by performance metric
        
        Args:
            metric (str): Metric to rank by
            
        Returns:
            pd.DataFrame: Ranked domains
        """
        if self.domain_comparisons is None or len(self.domain_comparisons) == 0:
            return pd.DataFrame()
        
        # For R² and Directional Accuracy, higher is better
        ascending = metric.lower() not in ['r2_score', 'r² score', 'directional accuracy']
        
        ranked = self.domain_comparisons.sort_values(
            by=metric.upper() if metric.upper() in self.domain_comparisons.columns else metric,
            ascending=ascending
        )
        
        return ranked
    
    def calculate_forecast_horizon_metrics(self, y_true, y_pred, horizons):
        """
        Calculate metrics for different forecast horizons
        
        Args:
            y_true (np.array): True values
            y_pred (np.array): Predicted values
            horizons (list): List of horizon breakpoints
            
        Returns:
            dict: Horizon-wise metrics
        """
        horizon_metrics = {}
        
        for i, horizon in enumerate(horizons):
            start_idx = horizons[i-1] if i > 0 else 0
            end_idx = horizon
            
            if end_idx <= len(y_true):
                metrics = self.calculate_metrics(
                    y_true[start_idx:end_idx],
                    y_pred[start_idx:end_idx]
                )
                
                horizon_name = f"Horizon_{start_idx+1}_to_{end_idx}"
                horizon_metrics[horizon_name] = metrics
        
        return horizon_metrics
    
    def generate_evaluation_report(self, domain_name, y_true, y_pred, additional_info=None):
        """
        Generate comprehensive evaluation report
        
        Args:
            domain_name (str): Name of the domain
            y_true (np.array): True values
            y_pred (np.array): Predicted values
            additional_info (dict): Additional information to include
            
        Returns:
            dict: Comprehensive evaluation report
        """
        metrics = self.calculate_metrics(y_true, y_pred, domain_name)
        error_dist = self.calculate_error_distribution(y_true, y_pred)
        
        report = {
            'domain': domain_name,
            'metrics': metrics,
            'error_distribution': error_dist,
            'sample_size': len(y_true),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        if additional_info:
            report['additional_info'] = additional_info
        
        return report
    
    def get_performance_summary(self):
        """Get summary of all evaluated domains"""
        if not self.evaluation_results:
            return "No evaluation results available"
        
        summary = "\n" + "="*70 + "\n"
        summary += "PERFORMANCE SUMMARY ACROSS DOMAINS\n"
        summary += "="*70 + "\n\n"
        
        for domain, metrics in self.evaluation_results.items():
            summary += f"Domain: {domain}\n"
            summary += f"  RMSE: {metrics['rmse']:.4f}\n"
            summary += f"  MAE:  {metrics['mae']:.4f}\n"
            summary += f"  MAPE: {metrics['mape']:.2f}%\n"
            summary += f"  R²:   {metrics['r2_score']:.4f}\n"
            summary += "-" * 70 + "\n"
        
        summary += "="*70 + "\n"
        
        return summary
    
    def export_results(self, filepath):
        """
        Export evaluation results to JSON
        
        Args:
            filepath (str): Path to save results
        """
        import json
        
        export_data = {
            'evaluation_results': self.evaluation_results,
            'domain_comparisons': self.domain_comparisons.to_dict() if isinstance(self.domain_comparisons, pd.DataFrame) else {},
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Evaluation results exported to: {filepath}")
