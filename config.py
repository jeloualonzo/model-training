"""
Configuration
- MobileNetV2 α=0.5
- Early stopping enabled
- TF.js conversion settings
- S3 upload settings
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Model Architecture (EXACT WebGL Match)
    MODEL = {
        'backbone': 'MobileNetV2',
        'alpha': 0.5,
        'input_shape': (224, 224, 1),
        'feature_dim': 1280,
        'num_classes': None,
        'dropout_rate': 0.6,
        'l2_regularization': 0.01,
    }
    
    # Training Hyperparameters
    TRAINING = {
        'epochs': 70,
        'batch_size': 32,
        'learning_rate': 0.0005,
        'validation_split': 0.15,
        'augmentation_count': 6,
        'min_samples_per_class': 1,
        'early_stopping_patience': 15,
        'early_stopping_min_delta': 0.001,
        'early_stopping_restore_best': True,
    }
    
    # Augmentation settings
    AUGMENTATION = {
        'enabled': True,
        'per_sample': 4,
        
        'geometric': {
            'rotation_range': 5,
            'perspective_skew': 6,
            'crop_amount': 0.10,
        },
        'lighting': {
            'brightness_range': (0.8, 1.2),
            'contrast_range': (0.85, 1.15),
            'shadow_intensity': (0.15, 0.40),
            'glare_intensity': 0.40,
        },
        'camera_quality': {
            'blur_max': 3,
            'noise_level': (6, 15),
            'compression': 0.12,
        },
        'writing_variations': {
            'rushed_jitter': 0.20,
            'stroke_thinning': 0.15,
            'stroke_dropout': 0.12,
            'stroke_erosion': 0.08,
            'pen_pressure': 0.05,
            'clean': 0.40,
        }
    }
    
    # API Settings
    API = {
        'cors_origins': '*',
        'host': '0.0.0.0',
        'port': 5000,
        'debug': False,
    }
    
    # Storage
    STORAGE = {
        'models_dir': 'trained_models',
        'tfjs_dir': 'tfjs_models',
        'save_format': 'keras',
    }
    
    # Validation options
    VALIDATION = {
        'enabled': True,
        'num_test_samples': 10,
        'max_acceptable_diff': 0.0001,
        'strict_mode': False,
    }
    
    # TensorFlow.js Conversion
    TFJS = {
        'enabled': os.getenv('VITE_ENABLE_S3_STORAGE', 'true').lower() == 'true',
        'quantization': None,
        'auto_upload': True,
        'validate_conversion': True,
        'save_comparison_report': True,
    }
    
    # AWS S3 Configuration
    S3 = {
        'enabled': os.getenv('VITE_ENABLE_S3_STORAGE', 'true').lower() == 'true',
        'access_key': os.getenv('VITE_AWS_ACCESS_KEY_ID'),
        'secret_key': os.getenv('VITE_AWS_SECRET_ACCESS_KEY'),
        'bucket': os.getenv('VITE_S3_BUCKET', 'signatures-model-storage'),
        'region': os.getenv('VITE_AWS_REGION', 'ap-southeast-1'),
        'public_base_url': os.getenv('VITE_S3_PUBLIC_BASE_URL'),
    }
    
    # Supabase Configuration
    SUPABASE = {
        'enabled': True,
        'url': os.getenv('SUPABASE_URL'),
        'key': os.getenv('SUPABASE_KEY'),
    }

config = Config()