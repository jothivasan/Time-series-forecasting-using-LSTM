"""
Generic Data Processor for Multi-Domain Time Series Forecasting
Handles domain-agnostic preprocessing, feature engineering, and sequence generation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class GenericDataProcessor:
    """
    Domain-agnostic data processor for time series forecasting
    Automatically adapts to different dataset structures and domains
    """
    
    def __init__(self, domain_config=None):
        """
        Initialize the data processor
        
        Args:
            domain_config (dict): Domain-specific configuration
        """
        self.domain_config = domain_config or {}
        self.scaler = None
        self.feature_columns = []
        self.target_column = None
        self.date_column = None
        self.original_data = None
        self.processed_data = None
        self.statistics = {}
        
    def load_data(self, file_path, date_column=None, target_column=None):
        """
        Load data from CSV file
        
        Args:
            file_path (str): Path to CSV file
            date_column (str): Name of date/time column
            target_column (str): Name of target column to forecast
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        try:
            # Load CSV
            df = pd.read_csv(file_path)
            
            # Auto-detect date column if not provided
            if date_column is None:
                date_column = self._detect_date_column(df)
            
            # Auto-detect target column if not provided
            if target_column is None:
                target_column = self._detect_target_column(df, date_column)
            
            self.date_column = date_column
            self.target_column = target_column
            
            # Set date as index
            if date_column in df.columns:
                df[date_column] = pd.to_datetime(df[date_column])
                df = df.set_index(date_column)
                df = df.sort_index()
            
            self.original_data = df.copy()
            self.processed_data = df.copy()
            
            # Calculate basic statistics
            self._calculate_statistics()
            
            return df
            
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def _detect_date_column(self, df):
        """Auto-detect date/time column"""
        date_keywords = ['date', 'time', 'datetime', 'timestamp', 'month', 'year', 'day']
        
        for col in df.columns:
            if any(keyword in col.lower() for keyword in date_keywords):
                return col
        
        # Try to parse first column as date
        try:
            pd.to_datetime(df.iloc[:, 0])
            return df.columns[0]
        except:
            return None
    
    def _detect_target_column(self, df, date_column):
        """Auto-detect target column (usually the last numeric column)"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if date_column in numeric_cols:
            numeric_cols.remove(date_column)
        
        if len(numeric_cols) > 0:
            return numeric_cols[-1]  # Return last numeric column
        
        return None
    
    def _calculate_statistics(self):
        """Calculate dataset statistics"""
        df = self.original_data
        
        self.statistics = {
            'total_records': len(df),
            'date_range': {
                'start': str(df.index[0]) if hasattr(df.index[0], 'strftime') else str(df.index[0]),
                'end': str(df.index[-1]) if hasattr(df.index[-1], 'strftime') else str(df.index[-1])
            },
            'features': {},
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.astype(str).to_dict()
        }
        
        # Feature-wise statistics
        for col in df.select_dtypes(include=[np.number]).columns:
            self.statistics['features'][col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'median': float(df[col].median())
            }
    
    def handle_missing_values(self, strategy='forward_fill'):
        """
        Handle missing values in the dataset
        
        Args:
            strategy (str): Strategy to handle missing values
                          ('forward_fill', 'backward_fill', 'interpolate', 'mean', 'median')
        """
        df = self.processed_data.copy()
        
        if strategy == 'forward_fill':
            df = df.fillna(method='ffill')
        elif strategy == 'backward_fill':
            df = df.fillna(method='bfill')
        elif strategy == 'interpolate':
            df = df.interpolate(method='linear')
        elif strategy == 'mean':
            df = df.fillna(df.mean())
        elif strategy == 'median':
            df = df.fillna(df.median())
        
        # Fill any remaining NaN with 0
        df = df.fillna(0)
        
        self.processed_data = df
        return df
    
    def create_lag_features(self, columns=None, lags=[1, 2, 3]):
        """
        Create lag features for specified columns
        
        Args:
            columns (list): Columns to create lag features for
            lags (list): List of lag values
        """
        df = self.processed_data.copy()
        
        if columns is None:
            columns = [self.target_column]
        
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Drop rows with NaN created by lagging
        df = df.dropna()
        
        self.processed_data = df
        return df
    
    def create_rolling_features(self, columns=None, windows=[3, 7, 14]):
        """
        Create rolling statistics features
        
        Args:
            columns (list): Columns to create rolling features for
            windows (list): List of window sizes
        """
        df = self.processed_data.copy()
        
        if columns is None:
            columns = [self.target_column]
        
        for col in columns:
            if col in df.columns:
                for window in windows:
                    df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                    df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
        
        # Drop rows with NaN created by rolling
        df = df.dropna()
        
        self.processed_data = df
        return df
    
    def create_date_features(self):
        """Create date-based features (month, day of week, etc.)"""
        df = self.processed_data.copy()
        
        if hasattr(df.index, 'month'):
            df['month'] = df.index.month
            df['day_of_week'] = df.index.dayofweek
            df['day_of_month'] = df.index.day
            df['quarter'] = df.index.quarter
            df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        self.processed_data = df
        return df
    
    def scale_data(self, method='minmax', feature_columns=None):
        """
        Scale the data using specified method
        
        Args:
            method (str): Scaling method ('minmax', 'standard', 'robust')
            feature_columns (list): Columns to scale (None = all numeric)
        """
        df = self.processed_data.copy()
        
        # Select scaler
        if method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Select columns to scale
        if feature_columns is None:
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        self.feature_columns = feature_columns
        
        # Scale data
        df[feature_columns] = self.scaler.fit_transform(df[feature_columns])
        
        self.processed_data = df
        return df
    
    def create_sequences(self, lookback=12, target_column=None, feature_columns=None):
        """
        Create sequences for LSTM training
        
        Args:
            lookback (int): Number of time steps to look back
            target_column (str): Target column to predict
            feature_columns (list): Feature columns to use
            
        Returns:
            tuple: (X, y) sequences
        """
        df = self.processed_data.copy()
        
        if target_column is None:
            target_column = self.target_column
        
        if feature_columns is None:
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Ensure target column is in features
        if target_column not in feature_columns:
            feature_columns.append(target_column)
        
        data = df[feature_columns].values
        
        X, y = [], []
        
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i, feature_columns.index(target_column)])
        
        return np.array(X), np.array(y)
    
    def split_data(self, X, y, train_ratio=0.7, val_ratio=0.2):
        """
        Split data into train, validation, and test sets
        
        Args:
            X (np.array): Input sequences
            y (np.array): Target values
            train_ratio (float): Ratio of training data
            val_ratio (float): Ratio of validation data
            
        Returns:
            tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        n = len(X)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def inverse_transform(self, data, column_name=None):
        """
        Inverse transform scaled data back to original scale
        
        Args:
            data (np.array): Scaled data
            column_name (str): Name of the column to inverse transform
            
        Returns:
            np.array: Data in original scale
        """
        if self.scaler is None:
            return data
        
        # Create dummy array with same shape as training data
        dummy = np.zeros((len(data), len(self.feature_columns)))
        
        # Find column index
        if column_name is None:
            column_name = self.target_column
        
        col_idx = self.feature_columns.index(column_name)
        dummy[:, col_idx] = data.flatten()
        
        # Inverse transform
        inverse = self.scaler.inverse_transform(dummy)
        
        return inverse[:, col_idx]
    
    def get_statistics(self):
        """Get dataset statistics"""
        return self.statistics
    
    def get_processed_data(self):
        """Get processed dataframe"""
        return self.processed_data
    
    def validate_temporal_order(self):
        """Validate that data is in correct temporal order"""
        if hasattr(self.processed_data.index, 'to_pydatetime'):
            dates = self.processed_data.index
            is_sorted = all(dates[i] <= dates[i+1] for i in range(len(dates)-1))
            return is_sorted
        return True
    
    def detect_anomalies(self, column=None, threshold=3):
        """
        Detect anomalies using z-score method
        
        Args:
            column (str): Column to check for anomalies
            threshold (float): Z-score threshold
            
        Returns:
            pd.DataFrame: Rows with anomalies
        """
        if column is None:
            column = self.target_column
        
        df = self.processed_data.copy()
        
        if column in df.columns:
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            anomalies = df[z_scores > threshold]
            return anomalies
        
        return pd.DataFrame()
    
    def get_data_quality_report(self):
        """Generate comprehensive data quality report"""
        df = self.original_data
        
        report = {
            'total_records': len(df),
            'total_features': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'temporal_order_valid': self.validate_temporal_order(),
            'numeric_features': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_features': df.select_dtypes(include=['object']).columns.tolist(),
            'date_range': self.statistics.get('date_range', {}),
            'anomalies_detected': len(self.detect_anomalies())
        }
        
        return report
