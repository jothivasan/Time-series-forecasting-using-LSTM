"""
Multi-Domain LSTM Time Series Forecasting System - Backend
Flask application for generic time series forecasting across multiple domains
"""

from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import pandas as pd
import json
import os
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import traceback

# Import custom modules
from core.data_processor import GenericDataProcessor
from core.lstm_model import GenericLSTMModel
from core.evaluator import PerformanceEvaluator
from core.forecaster import ForecastingEngine

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create necessary directories
os.makedirs('uploads', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Global variables
current_processor = None
current_model = None
current_domain = None
training_history = None
evaluation_results = None

# Domain configurations (kept empty for compatibility)
DOMAIN_CONFIG = {'domains': {}}


def convert_to_json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)
    else:
        return obj


@app.route('/')
def index():
    """Render main dashboard"""
    return render_template('dashboard.html')




@app.route('/api/upload_dataset', methods=['POST'])
def upload_dataset():
    """Upload and analyze custom dataset"""
    global current_processor, current_domain
    
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'})
        
        file = request.files['file']
        domain_id = request.form.get('domain_id', 'custom')
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Save uploaded file
        # Clear uploads folder first to keep only the latest file
        import shutil
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            shutil.rmtree(app.config['UPLOAD_FOLDER'])
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Get domain configuration
        domain_config = DOMAIN_CONFIG['domains'].get(domain_id, {})
        
        # Initialize data processor
        current_processor = GenericDataProcessor(domain_config)
        
        # Load and analyze data
        date_column = domain_config.get('date_column')
        target_column = domain_config.get('target_column')
        
        df = current_processor.load_data(filepath, date_column, target_column)
        
        # Get statistics and quality report
        stats = current_processor.get_statistics()
        quality_report = current_processor.get_data_quality_report()
        
        # Convert to JSON serializable
        stats = convert_to_json_serializable(stats)
        quality_report = convert_to_json_serializable(quality_report)
        
        # Get data preview
        preview = df.head(10).reset_index()
        preview_data = preview.to_dict('records')
        
        # Convert datetime objects to strings
        for record in preview_data:
            for key, value in record.items():
                if isinstance(value, (pd.Timestamp, datetime)):
                    record[key] = str(value)
                elif isinstance(value, (np.integer, np.floating)):
                    record[key] = float(value)
        
        current_domain = domain_id
        
        return jsonify({
            'success': True,
            'statistics': stats,
            'quality_report': quality_report,
            'preview': preview_data,
            'columns': list(df.columns),
            'domain': domain_id
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()})




@app.route('/api/preprocess', methods=['POST'])
def preprocess_data():
    """Preprocess the loaded dataset"""
    global current_processor
    
    try:
        if current_processor is None:
            return jsonify({'success': False, 'error': 'No dataset loaded'})
        
        data = request.json
        
        # Get preprocessing parameters
        missing_value_strategy = data.get('missing_value_strategy', 'forward_fill')
        scaling_method = data.get('scaling_method', 'minmax')
        create_lag = data.get('create_lag_features', False)
        create_rolling = data.get('create_rolling_features', False)
        create_date = data.get('create_date_features', False)
        
        # Apply preprocessing steps
        current_processor.handle_missing_values(missing_value_strategy)
        
        if create_lag:
            lags = data.get('lags', [1, 2, 3])
            current_processor.create_lag_features(lags=lags)
        
        if create_rolling:
            windows = data.get('windows', [3, 7])
            current_processor.create_rolling_features(windows=windows)
        
        if create_date:
            current_processor.create_date_features()
        
        # Scale data
        current_processor.scale_data(method=scaling_method)
        
        # Get processed data info
        processed_df = current_processor.get_processed_data()
        
        return jsonify({
            'success': True,
            'message': 'Data preprocessed successfully',
            'processed_shape': processed_df.shape,
            'processed_columns': list(processed_df.columns)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()})


@app.route('/api/train_model', methods=['POST'])
def train_model():
    """Train LSTM model"""
    global current_processor, current_model, training_history
    
    try:
        if current_processor is None:
            return jsonify({'success': False, 'error': 'No dataset loaded'})
        
        data = request.json
        
        # Get training parameters
        lookback = data.get('lookback', 12)
        lstm_units = data.get('lstm_units', 100)
        dropout_rate = data.get('dropout_rate', 0.2)
        learning_rate = data.get('learning_rate', 0.001)
        epochs = data.get('epochs', 100)
        batch_size = data.get('batch_size', 32)
        
        # Create sequences
        X, y = current_processor.create_sequences(lookback=lookback)
        
        if len(X) == 0:
            return jsonify({'success': False, 'error': f'Dataset too small ({len(current_processor.processed_data)} records) for lookback period ({lookback}). Please reduce lookback or use a larger dataset.'})
        
        # Split data
        X_train, y_train, X_val, y_val, X_test, y_test = current_processor.split_data(X, y)
        
        if len(X_train) == 0:
             return jsonify({'success': False, 'error': 'Insufficient data for training after splitting. Try reducing lookback or increasing dataset size.'})
        
        # Initialize model
        model_config = {
            'lstm_units': lstm_units,
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'batch_size': batch_size
        }
        
        current_model = GenericLSTMModel(config=model_config)
        current_model.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Train model
        history = current_model.train(X_train, y_train, X_val, y_val, verbose=0)
        training_history = history
        
        # Evaluate on test set
        test_metrics = current_model.evaluate(X_test, y_test)
        
        # Save model
        model_path = f'models/{current_domain}_lstm_model.h5'
        current_model.save_model(model_path, metadata={'domain': current_domain})
        
        return jsonify({
            'success': True,
            'message': 'Model trained successfully',
            'training_history': history,
            'test_metrics': test_metrics,
            'model_path': model_path
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()})


@app.route('/api/forecast', methods=['POST'])
def generate_forecast():
    """Generate forecasts"""
    global current_processor, current_model
    
    try:
        if current_processor is None or current_model is None:
            return jsonify({'success': False, 'error': 'Model not trained'})
        
        data = request.json
        
        horizon_type = data.get('horizon_type', 'mid_term')
        steps = data.get('steps', 12)
        
        # Get last sequence from processed data
        lookback = data.get('lookback', 12)
        X, y = current_processor.create_sequences(lookback=lookback)
        last_sequence = X[-1]
        
        # Initialize forecasting engine
        forecaster = ForecastingEngine(current_model, current_processor)
        
        # Generate forecast based on horizon type
        if horizon_type == 'short_term':
            forecast_result = forecaster.forecast_short_term(last_sequence, min(steps, 3))
        elif horizon_type == 'mid_term':
            forecast_result = forecaster.forecast_mid_term(last_sequence, min(steps, 12))
        elif horizon_type == 'long_term':
            forecast_result = forecaster.forecast_long_term(last_sequence, steps)
        else:
            forecast_result = forecaster.forecast_custom_horizon(last_sequence, steps)
        
        # Generate forecast dates
        last_date = current_processor.processed_data.index[-1]
        
        # Get frequency from domain config, default to 'daily' for custom domains
        if current_domain and current_domain in DOMAIN_CONFIG['domains']:
            frequency = DOMAIN_CONFIG['domains'][current_domain].get('frequency', 'daily')
        else:
            frequency = 'daily'  # Default for custom uploads
            
        freq_map = {'daily': 'D', 'monthly': 'M', 'hourly': 'H', 'weekly': 'W'}
        freq_code = freq_map.get(frequency, 'D')
        
        forecast_dates = forecaster.generate_forecast_dates(last_date, steps, freq_code)
        
        forecast_result['dates'] = forecast_dates
        
        return jsonify({
            'success': True,
            'forecast': forecast_result
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()})


@app.route('/api/evaluate', methods=['POST'])
def evaluate_model():
    """Evaluate model performance"""
    global current_processor, current_model, evaluation_results
    
    try:
        if current_processor is None or current_model is None:
            return jsonify({'success': False, 'error': 'Model not trained'})
        
        data = request.json
        lookback = data.get('lookback', 12)
        
        # Create sequences
        X, y = current_processor.create_sequences(lookback=lookback)
        
        # Split data
        X_train, y_train, X_val, y_val, X_test, y_test = current_processor.split_data(X, y)
        
        # Make predictions
        y_pred = current_model.predict(X_test)
        
        # Initialize evaluator
        evaluator = PerformanceEvaluator()
        
        # Calculate metrics
        metrics = evaluator.calculate_metrics(y_test, y_pred, current_domain)
        error_dist = evaluator.calculate_error_distribution(y_test, y_pred)
        
        # Generate report
        report = evaluator.generate_evaluation_report(
            current_domain,
            y_test,
            y_pred,
            additional_info={'lookback': lookback}
        )
        
        evaluation_results = report
        
        return jsonify({
            'success': True,
            'metrics': metrics,
            'error_distribution': error_dist,
            'report': report
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()})


@app.route('/api/visualize', methods=['POST'])
def create_visualization():
    """Create visualization plots"""
    try:
        data = request.json
        plot_type = data.get('plot_type', 'forecast')
        
        if plot_type == 'training_history' and training_history:
            # Plot training history
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            fig.patch.set_facecolor('#0a0a0a')
            
            # Loss plot
            ax1.set_facecolor('#0a0a0a')
            ax1.plot(training_history['loss'], label='Training Loss', color='#00ff88', linewidth=2)
            ax1.plot(training_history['val_loss'], label='Validation Loss', color='#ff6b6b', linewidth=2)
            ax1.set_xlabel('Epoch', color='white')
            ax1.set_ylabel('Loss', color='white')
            ax1.set_title('Model Loss', color='white', fontsize=14, pad=15)
            ax1.legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white')
            ax1.grid(True, alpha=0.2, color='white')
            ax1.tick_params(colors='white')
            for spine in ax1.spines.values():
                spine.set_color('white')
            
            # MAE plot
            ax2.set_facecolor('#0a0a0a')
            ax2.plot(training_history['mae'], label='Training MAE', color='#00ff88', linewidth=2)
            ax2.plot(training_history['val_mae'], label='Validation MAE', color='#ff6b6b', linewidth=2)
            ax2.set_xlabel('Epoch', color='white')
            ax2.set_ylabel('MAE', color='white')
            ax2.set_title('Model MAE', color='white', fontsize=14, pad=15)
            ax2.legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white')
            ax2.grid(True, alpha=0.2, color='white')
            ax2.tick_params(colors='white')
            for spine in ax2.spines.values():
                spine.set_color('white')
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', facecolor='#0a0a0a', dpi=150)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return jsonify({'success': True, 'plot': image_base64})
        
        return jsonify({'success': False, 'error': 'Invalid plot type or no data available'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 CUSTOM DATASET LSTM TIME SERIES FORECASTING SYSTEM")
    print("="*80)
    print("\n📊 Generic LSTM-Based Forecasting Framework")
    print("🌐 Access the application at: http://localhost:5001")
    print("\n" + "="*80 + "\n")
    
    app.run(debug=True, port=5001, threaded=True)
