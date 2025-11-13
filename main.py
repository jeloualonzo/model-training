# filepath: python/main.py
"""
Flask API Server 
"""
import boto3
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed 
import io
from PIL import Image
import uuid
from supabase_recorder import SupabaseRecorder
from io import BytesIO
import base64
import cv2 
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import multiprocessing
import os
from datetime import datetime
import threading
import time
from config import config
from training import SignatureTrainer
from utils import (
    get_latest_model, 
    get_model_metadata, 
    list_trained_models,
    validate_training_data,
    format_training_summary
)

# ===========================
# PARALLEL DOWNLOAD HELPER
# ===========================
def download_signature_batch(s3_client, bucket, s3_keys, max_workers=50):
    """
    Download and process S3 objects in parallel (optimized for T4 GPU)
    
    Args:
        s3_client: boto3 S3 client
        bucket: S3 bucket name
        s3_keys: List of S3 keys to download
        max_workers: Download workers (default 50 for T4 GPU)
    
    Returns:
        List of tuples: (s3_key, img_base64) or (s3_key, None) on error
    """
    
    # STEP 1: Parallel download (I/O-bound)
    def download_raw(s3_key):
        try:
            response = s3_client.get_object(Bucket=bucket, Key=s3_key)
            return (s3_key, response['Body'].read())
        except Exception as e:
            print(f"   ❌ Download failed {s3_key}: {e}")
            return (s3_key, None)
    
    print(f"   🔧 Downloading {len(s3_keys)} files with {max_workers} workers...")
    download_start = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        raw_data = list(executor.map(download_raw, s3_keys))
    
    download_time = time.time() - download_start
    print(f"   ⏱️  Download: {download_time:.2f}s ({len(s3_keys)/download_time:.1f} files/sec)")
    
    # STEP 2: Parallel processing (CPU-bound)
    def process_image(item):
        s3_key, img_bytes = item
        if img_bytes is None:
            return (s3_key, None)
        
        try:
            img = Image.open(io.BytesIO(img_bytes))
            
            if img.mode != 'L':
                img = img.convert('L')
            
            img = img.resize((224, 224), Image.LANCZOS)
            
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            return (s3_key, f"data:image/png;base64,{img_base64}")
        except Exception as e:
            print(f"   ❌ Process failed {s3_key}: {e}")
            return (s3_key, None)
    
    cpu_workers = multiprocessing.cpu_count()
    print(f"   🔧 Processing {len(raw_data)} images with {cpu_workers} CPU workers...")
    process_start = time.time()
    
    with ThreadPoolExecutor(max_workers=cpu_workers) as executor:
        results = list(executor.map(process_image, raw_data))
    
    process_time = time.time() - process_start
    print(f"   ⏱️  Processing: {process_time:.2f}s ({len(raw_data)/process_time:.1f} images/sec)")
    print(f"   📊 Total: {download_time + process_time:.2f}s")
    
    return results

# Initialize Flask app
app = Flask(__name__)

# Configure CORS - Allow all origins
CORS(app, resources={
    r"/*": {
        "origins": config.API['cors_origins'],
        "methods": ["GET", "POST", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# ===========================
# GLOBAL STATE
# ===========================
trainer = SignatureTrainer()
supabase_recorder = None
training_status = {
    'is_training': False,
    'progress': 0,
    'current_class': None,
    'total_classes': 0,
    'completed_classes': 0,
    'error': None,
    'start_time': None,
    'record_id': None, 
    'model_uuid': None,
}

# ===========================
# INITIALIZATION
# ===========================

# Initialize Supabase recorder
def init_supabase():
    """Initialize Supabase recorder on first use"""
    global supabase_recorder
    if supabase_recorder is None:
        try:
            if config.SUPABASE['enabled']:
                supabase_recorder = SupabaseRecorder(
                    config.SUPABASE['url'],
                    config.SUPABASE['key']
                )
                print("✅ Supabase recorder initialized")
        except Exception as e:
            print(f"❌ Failed to initialize Supabase: {e}")
            raise
    return supabase_recorder

# ===========================
# S3 DEFAULT SIGNATURES LOADER
# ===========================
def load_default_signatures_from_s3():
    """
    ✅ OPTIMIZED: Download default signatures in parallel
    """
    try:
        print("\n" + "="*60)
        print("📥 LOADING DEFAULT SIGNATURES FROM S3")
        print("="*60)
        
        s3 = boto3.client('s3',
            region_name=config.S3['region'],
            aws_access_key_id=config.S3['access_key'],
            aws_secret_access_key=config.S3['secret_key']
        )
        bucket = config.S3['bucket']
        
        # ✅ STEP 1: List all objects
        print("\n📂 Listing S3 objects...")
        
        unknown_response = s3.list_objects_v2(
            Bucket=bucket,
            Prefix='Default-classes/unknown/'
        )
        
        non_sig_response = s3.list_objects_v2(
            Bucket=bucket,
            Prefix='Default-classes/non-signature/'
        )
        
        unknown_keys = [
            obj['Key'] for obj in unknown_response.get('Contents', [])
            if not obj['Key'].endswith('/') and 
            obj['Key'].lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
        ]
        
        non_sig_keys = [
            obj['Key'] for obj in non_sig_response.get('Contents', [])
            if not obj['Key'].endswith('/') and 
            obj['Key'].lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
        ]
        
        print(f"   Found {len(unknown_keys)} unknown images")
        print(f"   Found {len(non_sig_keys)} non-signature images")
        
        # ✅ STEP 2: Download in parallel
        print("\n📥 Downloading in parallel...")
        start_time = time.time()
        
        all_keys = unknown_keys + non_sig_keys
        download_results = download_signature_batch(s3, bucket, all_keys)
        
        elapsed = time.time() - start_time
        print(f"✅ Downloaded {len(download_results)} images in {elapsed:.2f}s")
        
        # ✅ STEP 3: Split into categories
        image_map = {s3_key: img_data for s3_key, img_data in download_results if img_data is not None}
        
        unknown_images = [image_map[key] for key in unknown_keys if key in image_map]
        non_sig_images = [image_map[key] for key in non_sig_keys if key in image_map]
        
        print(f"\n✅ Loaded {len(unknown_images)} unknown, {len(non_sig_images)} non-signatures")
        
        return {
            'unknown': unknown_images,
            'non-signature': non_sig_images
        }
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR loading defaults from S3: {e}")
        import traceback
        traceback.print_exc()
        return None

# ===========================
# BASIC API ENDPOINTS
# ===========================
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with S3/Supabase status"""
    return jsonify({
        'status': 'online',
        'tfjs_enabled': config.TFJS['enabled'],
        's3_enabled': config.S3['enabled'],
        's3_bucket': config.S3['bucket'] if config.S3['enabled'] else None,
        'supabase_enabled': config.SUPABASE['enabled'],
        'timestamp': datetime.now().isoformat()
    })


@app.route('/config', methods=['GET'])
def get_config():
    """Get backend configuration"""
    return jsonify({
        'model': {
            'input_shape': list(config.MODEL['input_shape']),
            'embedding_dim': config.MODEL['feature_dim'],
            'backbone': config.MODEL['backbone'],
            'dropout': config.MODEL['dropout_rate'],
            'l2_regularization': config.MODEL['l2_regularization']
        },
        'training': {
            'epochs': config.TRAINING['epochs'],
            'batch_size': config.TRAINING['batch_size'],
            'learning_rate': config.TRAINING['learning_rate'],
            'validation_split': config.TRAINING['validation_split'],
            'augmentation_count': config.TRAINING['augmentation_count'],
            'min_samples': config.TRAINING['min_samples_per_class']
        }
    })

# ===========================
# PRODUCTION TRAINING ENDPOINT
# ===========================
@app.route('/train_production', methods=['POST'])
def train_production():
    """
    ✅ OPTIMIZED: Production training with parallel S3 downloads
    """
    global training_status
    
    try:
        data = request.json
        student_ids = data.get('student_ids', [])
        
        if not student_ids or len(student_ids) < 2:
            return jsonify({
                'success': False,
                'error': 'Need at least 2 students to train'
            }), 400
        
        print(f"\n🚀 Production Training: {len(student_ids)} students")
        
        # Initialize S3 client
        s3 = boto3.client('s3',
            region_name=config.S3['region'],
            aws_access_key_id=config.S3['access_key'],
            aws_secret_access_key=config.S3['secret_key']
        )
        bucket = config.S3['bucket']
        
        # Fetch student data from Supabase
        from supabase import create_client
        supabase = create_client(config.SUPABASE['url'], config.SUPABASE['key'])
        
        student_classes = []
        
        # ✅ STEP 1: Collect all S3 keys first
        all_s3_keys = []
        student_key_mapping = {}  # Map student_id → list of s3_keys
        
        for student_id in student_ids:
            try:
                # Get student info
                student_response = supabase.table('students') \
                    .select('*') \
                    .eq('student_id', student_id) \
                    .single() \
                    .execute()
                
                if not student_response.data:
                    print(f"   ⚠️ Student {student_id} not found")
                    continue
                
                student_data = student_response.data
                
                # Get student's signature S3 keys
                signatures_response = supabase.table('student_signatures') \
                    .select('s3_key') \
                    .eq('student_id', student_data['id']) \
                    .execute()
                
                if not signatures_response.data or len(signatures_response.data) < 3:
                    print(f"   ⚠️ Student {student_id} has insufficient signatures")
                    continue
                
                s3_keys = [sig['s3_key'] for sig in signatures_response.data]
                all_s3_keys.extend(s3_keys)
                student_key_mapping[student_id] = {
                    'student_data': student_data,
                    's3_keys': s3_keys
                }
                
                print(f"   ✅ {student_id}: {len(s3_keys)} signatures queued")
                
            except Exception as e:
                print(f"   ❌ Error processing {student_id}: {e}")
                continue
        
        if len(student_key_mapping) < 2:
            return jsonify({
                'success': False,
                'error': f'Only {len(student_key_mapping)} students successfully loaded (need 2+)'
            }), 400
        
        # ✅ STEP 2: Download ALL signatures in parallel
        print(f"\n📥 Downloading {len(all_s3_keys)} signatures in parallel...")
        start_time = time.time()
        
        download_results = download_signature_batch(s3, bucket, all_s3_keys)
        
        elapsed = time.time() - start_time
        print(f"✅ Downloaded {len(download_results)} images in {elapsed:.2f}s")
        
        # Create lookup map
        image_map = {s3_key: img_data for s3_key, img_data in download_results if img_data is not None}
        
        # ✅ STEP 3: Organize by student
        for student_id, data in student_key_mapping.items():
            student_data = data['student_data']
            s3_keys = data['s3_keys']
            
            samples = []
            for s3_key in s3_keys:
                img_data = image_map.get(s3_key)
                if img_data:
                    samples.append({
                        'thumbnail': img_data,
                        'timestamp': int(time.time())
                    })
            
            if len(samples) >= 3:
                student_classes.append({
                    'student': {
                        'id': student_data['id'],
                        'student_id': student_data['student_id'],
                        'firstname': student_data['firstname'],
                        'middlename': student_data.get('middlename'),
                        'surname': student_data['surname'],
                        'program': student_data['program'],
                        'year': student_data['year']
                    },
                    'samples': samples
                })
        
        print(f"\n✅ Loaded {len(student_classes)} students with signatures")
        
        # ✅ STEP 4: Load default classes
        print("\n📥 Loading default classes from S3...")
        default_signatures = load_default_signatures_from_s3()
        
        if default_signatures is None:
            return jsonify({
                'success': False,
                'error': 'Failed to load default signatures from S3'
            }), 500
        
        # ✅ STEP 5: Combine and train
        all_classes = [
            {
                'student': None,
                'isDefault': True,
                'defaultName': 'Unknown',
                'samples': [
                    {'thumbnail': img, 'timestamp': int(time.time())}
                    for img in default_signatures['unknown']
                ]
            },
            {
                'student': None,
                'isDefault': True,
                'defaultName': 'Non-signature',
                'samples': [
                    {'thumbnail': img, 'timestamp': int(time.time())}
                    for img in default_signatures['non-signature']
                ]
            },
            *student_classes
        ]
        
        # Check if already training
        if training_status['is_training']:
            return jsonify({
                'success': False, 
                'error': 'Training already in progress'
            }), 400
        
        # Start training in background
        def train_background():
            global training_status
            
            record_id = None
            model_uuid = str(uuid.uuid4())
            
            try:
                training_status['is_training'] = True
                training_status['progress'] = 0
                training_status['error'] = None
                training_status['start_time'] = time.time()
                training_status['model_uuid'] = model_uuid
                
                valid_classes = [c for c in all_classes if c.get('samples')]
                training_status['total_classes'] = len(valid_classes)
                training_status['completed_classes'] = 0
                
                # Calculate counts
                total_samples = sum(len(c.get('samples', [])) for c in valid_classes)
                student_count = len([c for c in valid_classes if not c.get('isDefault', False)])
                genuine_count = total_samples
                
                # Record training START in Supabase
                if config.SUPABASE['enabled']:
                    try:
                        recorder = init_supabase()
                        record_id = recorder.record_training_start(
                            model_uuid=model_uuid,
                            sample_count=total_samples,
                            student_count=student_count,
                            genuine_count=genuine_count
                        )
                        training_status['record_id'] = record_id
                        print(f"✅ Supabase record created: ID {record_id}")
                    except Exception as e:
                        print(f"⚠️ Failed to record training start in Supabase: {e}")
                
                # Train model
                metrics = trainer.train(all_classes)
                
                # Add additional metrics
                metrics['model_uuid'] = model_uuid
                metrics['batch_size'] = config.TRAINING['batch_size']
                metrics['learning_rate'] = config.TRAINING['learning_rate']
                metrics['augmentation_per_sample'] = config.AUGMENTATION['per_sample']
                
                # Save model + Convert to TF.js + Upload to S3
                save_result = trainer.save_model_with_tfjs()
                
                # ✅ UPDATED: Update Supabase with COMPLETE status including class_info
                if config.SUPABASE['enabled'] and record_id:
                    try:
                        print(f"\n🔄 Updating Supabase record {record_id}...")
                        recorder = init_supabase()
                        recorder.update_training_complete(
                            record_id=record_id,
                            metrics=metrics,
                            s3_info=save_result.get('s3_info', {}),
                            keras_model_path=save_result['keras_model_path'],
                            tfjs_model_path=save_result.get('tfjs_model_path', ''),
                            class_info=metrics.get('class_info')  # ✅ NEW: Pass per-class signature counts
                        )
                        print(f"✅ Supabase record {record_id} updated successfully")
                    except Exception as e:
                        print(f"⚠️ Failed to update Supabase: {e}")
                
                # Print summary
                print(format_training_summary(metrics))
                
                training_status['progress'] = 100
                training_status['completed_classes'] = training_status['total_classes']
                training_status['model_url'] = save_result.get('model_url')
                
            except Exception as e:
                print(f"❌ Training failed: {e}")
                import traceback
                traceback.print_exc()
                training_status['error'] = str(e)
                
                # Update Supabase with FAILURE status
                if config.SUPABASE['enabled'] and record_id:
                    try:
                        recorder = init_supabase()
                        recorder.update_training_failed(record_id, str(e))
                        print(f"✅ Supabase record {record_id} marked as failed")
                    except Exception as supabase_error:
                        print(f"⚠️ Failed to update Supabase failure status: {supabase_error}")
            
            finally:
                training_status['is_training'] = False
        
        thread = threading.Thread(target=train_background, daemon=True)
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Training started',
            'total_classes': len(all_classes),
            'student_classes': len(student_classes),
            'download_time_seconds': elapsed
        })
        
    except Exception as e:
        print(f"❌ Training endpoint error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/training_status', methods=['GET'])
def get_training_status():
    """Get current training status with model URL"""
    response = {
        'is_training': training_status['is_training'],
        'progress': training_status['progress'],
        'current_class': training_status['current_class'],
        'total_classes': training_status['total_classes'],
        'completed_classes': training_status['completed_classes'],
        'error': training_status['error'],
        'model_uuid': training_status.get('model_uuid'),
        'record_id': training_status.get('record_id'),
        'model_url': training_status.get('model_url')
    }
    
    if training_status['start_time']:
        response['elapsed_time'] = time.time() - training_status['start_time']
    
    return jsonify(response)

# ===========================
# SUPABASE MODEL ENDPOINTS
# ===========================
@app.route('/models/supabase/latest', methods=['GET'])
def get_latest_supabase_model():
    """Get latest model from Supabase"""
    try:
        if not config.SUPABASE['enabled']:
            return jsonify({'success': False, 'error': 'Supabase not enabled'}), 400
        
        recorder = init_supabase()
        model = recorder.get_latest_model()
        
        if not model:
            return jsonify({'success': False, 'error': 'No models found'}), 404
        
        return jsonify({
            'success': True,
            'model': model
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/models/supabase/<model_uuid>', methods=['GET'])
def get_supabase_model_by_uuid(model_uuid):
    """Get specific model from Supabase by UUID"""
    try:
        if not config.SUPABASE['enabled']:
            return jsonify({'success': False, 'error': 'Supabase not enabled'}), 400
        
        recorder = init_supabase()
        model = recorder.get_model_by_uuid(model_uuid)
        
        if not model:
            return jsonify({'success': False, 'error': f'Model {model_uuid} not found'}), 404
        
        return jsonify({
            'success': True,
            'model': model
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/models/supabase/list', methods=['GET'])
def list_supabase_models():
    """List models from Supabase"""
    try:
        if not config.SUPABASE['enabled']:
            return jsonify({'success': False, 'error': 'Supabase not enabled'}), 400
        
        limit = request.args.get('limit', 10, type=int)
        
        recorder = init_supabase()
        models = recorder.list_models(limit=limit)
        
        return jsonify({
            'success': True,
            'models': models,
            'count': len(models)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ===========================
# MODEL MANAGEMENT ENDPOINTS
# ===========================
@app.route('/models/list', methods=['GET'])
def list_models():
    """List all trained models"""
    try:
        models = list_trained_models()
        return jsonify({
            'success': True,
            'models': models
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/download_model', methods=['GET'])
def download_model():
    """Download trained model as ZIP"""
    try:
        from flask import send_file
        import zipfile
        
        # Get latest model path
        latest_model = get_latest_model()
        if not latest_model:
            return jsonify({'error': 'No trained model found'}), 404
        
        # Create ZIP file in memory
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add model file
            zip_file.write(latest_model, os.path.basename(latest_model))
            
            # Add metadata file
            metadata_path = latest_model.replace('model_', 'metadata_').replace('.keras', '.json')
            if os.path.exists(metadata_path):
                zip_file.write(metadata_path, os.path.basename(metadata_path))
        
        zip_buffer.seek(0)
        
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name='signature_model.zip'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ===========================
# RUN SERVER
# ===========================
if __name__ == '__main__':
    # Create directories
    os.makedirs(config.STORAGE['models_dir'], exist_ok=True)
    
    print("\n" + "="*60)
    print("🚀 SIGNATURE RECOGNITION API SERVER")
    print("="*60)
    print(f"📦 Model: {config.MODEL['backbone']}")
    print(f"🔧 Configuration: WebGL-matched")
    print(f"🌐 Server: http://{config.API['host']}:{config.API['port']}")
    print("="*60)
    print("\n📋 Available Endpoints:")
    print("   Training:")
    print("     POST /train_production - Train from Supabase")
    print("     GET  /training_status - Get training progress")
    print("\n   Model Management:")
    print("     GET  /models/list - List local models")
    print("     GET  /download_model - Download model as ZIP")
    print("     GET  /models/supabase/latest - Get latest from Supabase")
    print("     GET  /models/supabase/<uuid> - Get specific from Supabase")
    print("     GET  /models/supabase/list - List Supabase models")
    print("\n   Utilities:")
    print("     GET  /health - Health check")
    print("     GET  /config - Get configuration")
    print("="*60 + "\n")
    
    app.run(
        host=config.API['host'],
        port=config.API['port'],
        debug=config.API['debug']
    )