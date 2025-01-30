from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from functools import wraps
from datetime import datetime
import secrets
import logging
from logging.handlers import RotatingFileHandler
import json
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_limiter.storage import RedisStorage
import redis

app = Flask(__name__, static_folder='static')
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"],
        "supports_credentials": False
    }
})
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.secret_key = secrets.token_hex(32)  # Generate a secure secret key

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure logging
if not os.path.exists('logs'):
    os.makedirs('logs')
    
handler = RotatingFileHandler('logs/api.log', maxBytes=10000, backupCount=3)
handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)
app.logger.info('API startup')

# Security headers
@app.after_request
def add_security_headers(response):
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Content-Security-Policy'] = "default-src * 'unsafe-inline' 'unsafe-eval'; img-src * data:;"
    
    # Ensure CORS headers are set
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Accept'
    return response

# Add Redis configuration after app initialization
REDIS_URL = "redis://localhost:6379/0"  # Update this with your Redis server URL
redis_client = redis.from_url(REDIS_URL)

# Update the limiter configuration
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri=REDIS_URL,
    storage_options={"connection_pool": redis_client.connection_pool},
    default_limits=["200 per day", "50 per hour"]
)

# Add error handler for rate limit exceeded
@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        'error': 'Rate limit exceeded',
        'message': str(e.description)
    }), 429

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def load_data(file_path):
    """Load dataset from file, automatically detecting delimiter and headers."""
    try:
        df = pd.read_csv(file_path, low_memory=False)
        return df
    except Exception as e:
        return None

def clean_data(df):
    """Handle missing values and drop irrelevant columns dynamically."""
    df_cleaned = df.copy()
    df_cleaned.drop_duplicates(inplace=True)
    df_cleaned.fillna(df_cleaned.median(numeric_only=True), inplace=True)
    df_cleaned.dropna(axis=1, how='all', inplace=True)
    return df_cleaned

def create_boxplot(data):
    plt.figure(figsize=(8, 4))
    plt.boxplot(data)
    return plt.gcf()

def create_heatmap(corr_matrix):
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar()
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                    ha='center', va='center')
    return plt.gcf()

@app.route('/', methods=['GET'])
@limiter.limit("10 per minute")
def home():
    return jsonify({
        'status': 'API is running',
        'endpoints': {
            '/': 'Home - This message',
            '/test': 'Test endpoint',
            '/analyze': 'Upload and analyze data (POST request)'
        }
    })

@app.route('/analyze', methods=['POST'])
@limiter.limit("30 per hour")
def analyze():
    try:
        app.logger.info('Analysis request received')
        
        if 'file' not in request.files:
            app.logger.warning('No file provided in request')
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            app.logger.warning('Empty filename provided')
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            app.logger.warning(f'Invalid file type: {file.filename}')
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Log file details
        app.logger.info(f'Processing file: {file.filename}')
        
        try:
            # Save file with unique name
            filename = secure_filename(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            app.logger.info(f'File saved to: {filepath}')
            
            # Load and verify data
            df = load_data(filepath)
            if df is None:
                app.logger.error('Failed to load data from file')
                return jsonify({'error': 'Failed to load data - invalid CSV format'}), 400
            
            app.logger.info(f'Data loaded successfully: {len(df)} rows, {len(df.columns)} columns')
            
            # Process data and generate results
            results = []
            
            # Basic data summary
            try:
                df_info = io.StringIO()
                df.info(buf=df_info)
                summary_stats = df.describe().to_dict()
                missing_values = df.isnull().sum().to_dict()
                
                results.append({
                    'title': 'Dataset Overview',
                    'insights': [
                        f"Dataset contains {len(df)} rows and {len(df.columns)} columns",
                        f"Column Information:\n{df_info.getvalue()}",
                        f"Missing Values: {json.dumps(missing_values, indent=2)}"
                    ]
                })
            except Exception as e:
                app.logger.error(f'Error in data summary: {str(e)}')
                
            # Clean the data
            df = clean_data(df)
            
            # Outlier Detection
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
            for col in numerical_cols[:5]:
                fig = create_boxplot(df[col].dropna())
                plt.title(f"Outlier Detection: {col}")
                results.append({
                    'title': f'Outlier Analysis - {col}',
                    'visualization': fig_to_base64(fig),
                    'insights': [
                        "Boxplots highlight extreme values that could indicate potential errors or significant variations",
                        "Points beyond the whiskers are considered outliers"
                    ]
                })
                plt.close()
            
            # Data Distribution
            plt.figure(figsize=(12, 6))
            df.hist(bins=30, figsize=(12, 10), color='blue', alpha=0.7)
            plt.suptitle("Distribution of Numerical Features")
            results.append({
                'title': 'Distribution Analysis',
                'visualization': fig_to_base64(plt.gcf()),
                'insights': [
                    "Histograms provide a view of data distribution, identifying skewness or irregularities",
                    "Helps identify patterns and potential data quality issues"
                ]
            })
            plt.close()
            
            # Correlation Analysis
            corr_matrix = df.corr(numeric_only=True)
            fig = create_heatmap(corr_matrix)
            plt.title("Feature Correlation Heatmap")
            results.append({
                'title': 'Correlation Analysis',
                'visualization': fig_to_base64(fig),
                'insights': [
                    "Heatmap shows correlations between numerical variables",
                    "Darker colors indicate stronger correlations"
                ]
            })
            plt.close()
            
            # Classification Analysis (if applicable)
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                try:
                    target = categorical_cols[-1]
                    label_enc = LabelEncoder()
                    df[target] = label_enc.fit_transform(df[target])
                    features = df.select_dtypes(include=['int64', 'float64']).drop(columns=[target], errors='ignore')
                    target_values = df[target]
                    
                    X_train, X_test, y_train, y_test = train_test_split(features, target_values, test_size=0.3, random_state=42)
                    model = RandomForestClassifier()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    results.append({
                        'title': 'Classification Analysis',
                        'insights': [
                            f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}",
                            f"Target Variable: {target}",
                            "Classification model predicts categorical outcomes",
                            "Useful for customer satisfaction or segmentation analysis"
                        ]
                    })
                except Exception as e:
                    print(f"Classification analysis failed: {str(e)}")
            
            # Time Series Analysis (if applicable)
            date_cols = [col for col in df.columns if 'date' in col.lower()]
            if date_cols:
                try:
                    df_copy = df.copy()
                    df_copy[date_cols[0]] = pd.to_datetime(df_copy[date_cols[0]], errors='coerce')
                    df_copy.dropna(subset=[date_cols[0]], inplace=True)
                    df_copy.set_index(date_cols[0], inplace=True)
                    df_copy.sort_index(inplace=True)
                    
                    target_col = df_copy.select_dtypes(include=['int64', 'float64']).columns[0]
                    ts_data = df_copy[target_col].resample('M').sum()
                    
                    # ARIMA Forecasting
                    model = ARIMA(ts_data, order=(5,1,0))
                    model_fit = model.fit()
                    forecast = model_fit.forecast(steps=12)
                    
                    plt.figure(figsize=(12, 6))
                    plt.plot(ts_data.index, ts_data.values, label='Actual')
                    plt.plot(pd.date_range(start=ts_data.index[-1], periods=12, freq='M'), 
                            forecast, label='Forecast', linestyle='dashed')
                    plt.title(f"Time Series Analysis: {target_col}")
                    plt.legend()
                    plt.xticks(rotation=45)
                    
                    results.append({
                        'title': 'Time Series Analysis',
                        'visualization': fig_to_base64(plt.gcf()),
                        'insights': [
                            "Time series plot shows temporal patterns in the data",
                            "Dashed line shows forecasted values for next 12 periods",
                            "Useful for identifying trends and seasonality"
                        ]
                    })
                    plt.close()
                except Exception as e:
                    print(f"Time series analysis failed: {str(e)}")
            
            if not results:
                raise ValueError('No analysis results generated')
                
            app.logger.info(f'Analysis completed successfully with {len(results)} sections')
            return jsonify(results)
            
        except Exception as e:
            app.logger.error(f'Error processing file: {str(e)}', exc_info=True)
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
        finally:
            # Clean up
            if os.path.exists(filepath):
                os.remove(filepath)
                app.logger.info(f'Cleaned up file: {filepath}')
    
    except Exception as e:
        app.logger.error(f'Unexpected error: {str(e)}', exc_info=True)
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/test', methods=['GET'])
@limiter.limit("10 per minute")
def test():
    return jsonify({'status': 'Server is running'}), 200

@app.route('/health', methods=['GET', 'OPTIONS'])
def health_check():
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        # Check Redis connection
        redis_client.ping()
        
        # Check if upload directory exists and is writable
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            return jsonify({'status': 'error', 'message': 'Upload directory missing'}), 500
        if not os.access(app.config['UPLOAD_FOLDER'], os.W_OK):
            return jsonify({'status': 'error', 'message': 'Upload directory not writable'}), 500
            
        # Test matplotlib
        plt.figure()
        plt.close()
        
        return jsonify({
            'status': 'healthy',
            'upload_dir': 'ok',
            'matplotlib': 'ok',
            'redis': 'ok',
            'version': '1.0',
            'timestamp': datetime.now().isoformat()
        })
    except redis.ConnectionError as e:
        app.logger.error(f'Redis connection failed: {str(e)}')
        return jsonify({
            'status': 'error',
            'message': 'Redis connection failed',
            'timestamp': datetime.now().isoformat()
        }), 500
    except Exception as e:
        app.logger.error(f'Health check failed: {str(e)}', exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/test-analysis', methods=['GET'])
def test_analysis():
    # Create sample data
    sample_data = [
        {
            'title': 'Test Analysis',
            'insights': [
                'Dataset contains 100 rows and 5 columns',
                'Sample insight 1',
                'Sample insight 2'
            ],
            'visualization': None
        }
    ]
    return jsonify(sample_data)

if __name__ == '__main__':
    app.run(
        debug=False,  # Set to False in production
        host='0.0.0.0',
        port=8000,
        threaded=True  # Enable threading for better performance
    ) 
