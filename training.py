# filepath: python/training.py (UPDATED)
from classifier_extractor import ClassifierExtractor
from pathlib import Path
import uuid
from tfjs_converter import TFJSConverter
from s3_uploader import S3Uploader
from supabase_recorder import SupabaseRecorder
import numpy as np
import cv2
from typing import List, Dict, Tuple
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tqdm import tqdm
import json
import os
from datetime import datetime
from config import config
from model import create_classifier
from augmentation import augment_signature

class SignatureTrainer:
    def __init__(self):
        self.model = None
        self.feature_extractor = None
        self.class_labels = []
        self.num_classes = 0
        
    def prepare_data(
        self, 
        classes: List[Dict]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        ✅ UPDATED: No validation split - use all data for training
        
        Augmentation rules:
        - Default classes (unknown/non-signature): NO augmentation
        - Student captured signatures: NO augmentation
        - Student uploaded signatures: WITH augmentation
        """
        print("\n🔄 Preparing training data...")
        
        train_features = []
        train_labels = []
        val_features = []  # Will create dummy validation set
        val_labels = []
        class_labels = []
        
        valid_classes = [cls for cls in classes if cls.get('samples')]
        self.num_classes = len(valid_classes)
        
        for class_idx, cls in enumerate(valid_classes):
            # Detect class type
            is_default = cls.get('isDefault', False)
            
            if is_default:
                default_name = cls.get('defaultName', 'Unknown')
                class_name = default_name.upper()
                class_labels.append(class_name)
                print(f"\n📚 Default Class {class_idx + 1}/{self.num_classes}: {class_name}")
            else:
                student = cls.get('student', {})
                student_name = f"{student.get('student_id', 'Unknown')} - {student.get('firstname', '')} {student.get('surname', '')}"
                class_labels.append(student_name)
                print(f"\n📚 Student Class {class_idx + 1}/{self.num_classes}: {student_name}")
            
            samples = cls.get('samples', [])
            print(f"   Original samples: {len(samples)}")
            
            # ✅ NO VALIDATION SPLIT - Use all data for training
            train_samples = samples
            
            print(f"   Using all {len(train_samples)} samples for training (no validation split)")
            
            # ✅ NEW: Count signatures by type
            if not is_default:
                uploaded_count = len([s for s in train_samples if s.get('type') == 'uploaded'])
                captured_count = len([s for s in train_samples if s.get('type') == 'captured'])
                print(f"   Training breakdown: {uploaded_count} uploaded (with aug), {captured_count} captured (no aug)")
            
            # Process TRAINING samples (WITH selective augmentation)
            for sample in tqdm(train_samples, desc=f"   Processing"):
                img = self._decode_base64_image(sample['thumbnail'])
                if img is None:
                    continue
                
                # Always add original
                train_features.append(img)
                train_labels.append(class_idx)
                
                # ✅ AUGMENTATION RULES:
                # 1. Default classes: NO augmentation
                # 2. Captured signatures: NO augmentation
                # 3. Uploaded signatures: WITH augmentation
                should_augment = (
                    not is_default and  # Not a default class
                    sample.get('type') == 'uploaded'  # Only uploaded signatures
                )
                
                if should_augment:
                    for aug_idx in range(config.AUGMENTATION['per_sample']):
                        try:
                            aug_img = augment_signature(img.copy())
                            train_features.append(aug_img)
                            train_labels.append(class_idx)
                        except Exception as e:
                            print(f"⚠️ Augmentation failed: {e}")
                            continue
            
            # Summary
            train_count = len([l for l in train_labels if l == class_idx])
            
            if is_default:
                print(f"   ✅ Training: {train_count} (NO augmentation - default class)")
            else:
                uploaded_originals = len([s for s in train_samples if s.get('type') == 'uploaded'])
                captured_originals = len([s for s in train_samples if s.get('type') == 'captured'])
                uploaded_augmented = uploaded_originals * (config.AUGMENTATION['per_sample'] + 1)
                
                print(f"   ✅ Training: {train_count} total")
                print(f"      • {uploaded_augmented} uploaded (with {config.AUGMENTATION['per_sample']}x aug)")
                print(f"      • {captured_originals} captured (no aug)")
        
        # Convert to numpy arrays
        X_train = np.array(train_features, dtype=np.float32)
        y_train = np.array(train_labels, dtype=np.int32)
        
        # ✅ Create dummy validation set (required by Keras, but won't be used)
        # Use a small subset of training data
        val_size = min(len(train_features), 10)
        X_val = X_train[:val_size].copy()
        y_val = y_train[:val_size].copy()
        
        # Normalize images
        X_train = (X_train - 127.5) / 127.5
        X_val = (X_val - 127.5) / 127.5
        
        # Add channel dimension
        if len(X_train.shape) == 3:
            X_train = np.expand_dims(X_train, axis=-1)
        if len(X_val.shape) == 3:
            X_val = np.expand_dims(X_val, axis=-1)
        
        # One-hot encode labels
        y_train = keras.utils.to_categorical(y_train, num_classes=self.num_classes)
        y_val = keras.utils.to_categorical(y_val, num_classes=self.num_classes)
        
        print(f"\n✅ Dataset prepared:")
        print(f"   Training samples: {len(X_train)} (shape: {X_train.shape})")
        print(f"   Validation samples: {len(X_val)} (dummy set from training)")
        print(f"   Classes: {self.num_classes}")
        print(f"   Class labels: {class_labels}")
        
        return X_train, y_train, X_val, y_val, class_labels
    
    def train(self, classes: List[Dict]) -> Dict:
        """
        Train model without validation split
        
        Returns:
            Training metrics including per-class signature counts
        """
        print("\n" + "="*60)
        print("🚀 STARTING TRAINING")
        print("="*60)
        
        start_time = datetime.now()
        
        # ✅ NEW: Track per-class signature counts BEFORE data preparation
        class_info = []
        
        for class_idx, cls in enumerate(classes):
            if not cls.get('samples'):
                continue
            
            is_default = cls.get('isDefault', False)
            original_sample_count = len(cls.get('samples', []))
            
            if is_default:
                default_name = cls.get('defaultName', 'Unknown')
                class_info.append({
                    'label': default_name.upper(),
                    'signature_count': original_sample_count,
                    'is_default': True
                })
            else:
                student = cls.get('student', {})
                student_name = f"{student.get('student_id', 'Unknown')} - {student.get('firstname', '')} {student.get('surname', '')}"
                class_info.append({
                    'label': student_name,
                    'signature_count': original_sample_count,
                    'is_default': False
                })
        
        # Prepare data without validation split
        X_train, y_train, X_val, y_val, self.class_labels = self.prepare_data(classes)
        
        # Create model
        print("\n🏗️ Building model...")
        self.model, self.feature_extractor = create_classifier(self.num_classes)
        
        # Calculate batch size
        effective_batch_size = min(
            config.TRAINING['batch_size'],
            max(1, len(X_train) // 10)
        )
        
        # Setup callbacks
        callbacks = []
        
        # ✅ Early Stopping (monitor training loss instead of validation loss)
        if config.TRAINING['early_stopping_patience']:
            early_stop = EarlyStopping(
                monitor='loss',  # ✅ Changed from 'val_loss' to 'loss'
                patience=config.TRAINING['early_stopping_patience'],
                min_delta=config.TRAINING.get('early_stopping_min_delta', 0.001),
                restore_best_weights=config.TRAINING.get('early_stopping_restore_best', True),
                verbose=1,
                mode='min'
            )
            callbacks.append(early_stop)
            print(f"\n✅ Early stopping enabled:")
            print(f"   Monitor: loss (training loss)")
            print(f"   Patience: {config.TRAINING['early_stopping_patience']} epochs")
            print(f"   Min delta: {config.TRAINING.get('early_stopping_min_delta', 0.001)}")
            print(f"   Restore best weights: {config.TRAINING.get('early_stopping_restore_best', True)}")
        
        # ✅ Learning Rate Reduction on Plateau (monitor training loss)
        reduce_lr = ReduceLROnPlateau(
            monitor='loss',  # ✅ Changed from 'val_loss' to 'loss'
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1,
            mode='min'
        )
        callbacks.append(reduce_lr)
        
        print(f"\n📋 Training Configuration:")
        print(f"   Epochs: {config.TRAINING['epochs']}")
        print(f"   Batch size: {effective_batch_size}")
        print(f"   Learning rate: {config.TRAINING['learning_rate']}")
        print(f"   Validation: Dummy set (training on 100% of data)")
        print(f"   Callbacks: {len(callbacks)} enabled")
        
        # Train with callbacks
        print("\n🔥 Training started...\n")
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config.TRAINING['epochs'],
            batch_size=effective_batch_size,
            shuffle=True,
            verbose=1,
            callbacks=callbacks
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # ✅ Extract metrics (handle missing validation metrics)
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history.get('val_accuracy', [final_train_acc])[-1]
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history.get('val_loss', [final_train_loss])[-1]
        
        epochs_trained = len(history.history['accuracy'])
        
        # ✅ Best epoch based on training loss
        best_epoch = np.argmin(history.history['loss']) + 1
        best_val_loss = np.min(history.history.get('val_loss', history.history['loss']))
        best_val_acc = history.history.get('val_accuracy', history.history['accuracy'])[best_epoch - 1]
        
        metrics = {
            'training_accuracy': float(final_train_acc),
            'validation_accuracy': float(final_val_acc),
            'training_loss': float(final_train_loss),
            'validation_loss': float(final_val_loss), 
                        'training_time': training_time,
            'epochs_requested': config.TRAINING['epochs'],
            'epochs_trained': epochs_trained,
            'best_epoch': best_epoch,
            'best_val_loss': float(best_val_loss),
            'best_val_accuracy': float(best_val_acc),
            'early_stopped': epochs_trained < config.TRAINING['epochs'],
            'total_samples': len(X_train),
            'validation_samples': len(X_val),
            'num_classes': self.num_classes,
            'class_labels': self.class_labels,
            'class_info': class_info,  # ✅ NEW: Per-class signature counts
            'history': {
                'accuracy': [float(x) for x in history.history['accuracy']],
                'val_accuracy': [float(x) for x in history.history.get('val_accuracy', history.history['accuracy'])],
                'loss': [float(x) for x in history.history['loss']],
                'val_loss': [float(x) for x in history.history.get('val_loss', history.history['loss'])]
            }
        }
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE")
        print("="*60)
        print(f"📊 Final Training Accuracy: {final_train_acc*100:.2f}%")
        print(f"📊 Final Validation Accuracy: {final_val_acc*100:.2f}% (dummy set)")
        print(f"⏱️  Training Time: {training_time:.1f} seconds")
        
        if metrics['early_stopped']:
            print(f"\n🛑 Early stopping triggered!")
            print(f"   Stopped at epoch: {epochs_trained}/{config.TRAINING['epochs']}")
            print(f"   Best epoch: {best_epoch}")
            print(f"   Best loss: {best_val_loss:.4f}")
            print(f"   Best accuracy: {best_val_acc*100:.2f}%")
            print(f"   Time saved: ~{(config.TRAINING['epochs'] - epochs_trained) * (training_time / epochs_trained):.1f}s")
        else:
            print(f"\n✅ Completed all {epochs_trained} epochs")
            print(f"   Best epoch: {best_epoch}")
            print(f"   Best accuracy: {best_val_acc*100:.2f}%")
        
        # ✅ NEW: Print signature counts per class
        print(f"\n📊 Signature Counts:")
        for cls in class_info:
            label = cls['label']
            count = cls['signature_count']
            cls_type = "Default" if cls['is_default'] else "Student"
            print(f"   [{cls_type}] {label}: {count} signatures")
        
        print("="*60 + "\n")
        
        return metrics
    
    def save_model(self, save_dir: str = None) -> str:
        """Save trained model to disk"""
        if self.model is None:
            raise ValueError("No model to save. Train first.")
        
        if save_dir is None:
            save_dir = config.STORAGE['models_dir']
        
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(save_dir, f"model_{timestamp}.keras")
        metadata_path = os.path.join(save_dir, f"metadata_{timestamp}.json")
        
        # Save model
        self.model.save(model_path)
        
        # Save metadata
        metadata = {
            'created_at': timestamp,
            'num_classes': self.num_classes,
            'class_labels': self.class_labels,
            'model_architecture': 'MobileNetV2',
            'input_shape': list(config.MODEL['input_shape']),
            'training_config': {
                'epochs': config.TRAINING['epochs'],
                'batch_size': config.TRAINING['batch_size'],
                'learning_rate': config.TRAINING['learning_rate'],
                'validation_split': config.TRAINING['validation_split'],
                'early_stopping_enabled': config.TRAINING.get('early_stopping_patience') is not None,
                'early_stopping_patience': config.TRAINING.get('early_stopping_patience'),
            },
            'has_default_classes': any(label in ['UNKNOWN', 'NON-SIGNATURE'] for label in self.class_labels)
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Model saved: {model_path}")
        print(f"✅ Metadata saved: {metadata_path}")
        
        return model_path
    
    def _decode_base64_image(self, base64_str: str) -> np.ndarray:
        """Decode base64 image to numpy array (grayscale 224x224)"""
        try:
            # Remove data URI prefix if present
            if ',' in base64_str:
                base64_str = base64_str.split(',')[1]
            
            # Decode base64
            import base64
            img_bytes = base64.b64decode(base64_str)
            
            # Convert to numpy array
            nparr = np.frombuffer(img_bytes, np.uint8)
            
            # Decode image
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            # Resize to 224x224
            if img.shape != (224, 224):
                img = cv2.resize(img, (224, 224))
            
            return img
            
        except Exception as e:
            print(f"❌ Failed to decode image: {e}")
            return None
            
    def save_model_with_tfjs(self, save_dir: str = None) -> Dict[str, str]:
        """
        ✅ UPDATED: Save model + convert CLASSIFIER ONLY to TF.js
        """
        if self.model is None:
            raise ValueError("No model to save. Train first.")
        
        if save_dir is None:
            save_dir = config.STORAGE['models_dir']
        
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_uuid = str(uuid.uuid4())
        
        # Step 1: Save FULL model (for backup/reference)
        full_model_path = os.path.join(save_dir, f"model_full_{timestamp}.keras")
        self.model.save(full_model_path)
        print(f"✅ Full model saved: {full_model_path}")
        original_size_mb = os.path.getsize(full_model_path) / (1024 * 1024)
        print(f"   Size: {original_size_mb:.2f} MB")
        
        # ✅ Step 2: Extract classifier ONLY
        print("\n" + "="*60)
        print("🔧 EXTRACTING CLASSIFIER FOR TF.JS EXPORT")
        print("="*60)
        
        extractor = ClassifierExtractor()
        classifier_only = extractor.extract_classifier_from_model(
            full_model=self.model,
            class_labels=self.class_labels
        )
        
        # ✅ Step 3: Verify classifier extraction
        verification_passed = extractor.verify_classifier(
            full_model=self.model,
            classifier_only=classifier_only,
            num_tests=5
        )
        
        if not verification_passed:
            raise ValueError("Classifier verification failed! Not exporting to TF.js.")
        
        # ✅ Step 4: Save classifier only
        classifier_path = os.path.join(save_dir, f"classifier_only_{timestamp}.keras")
        classifier_only.save(classifier_path)
        classifier_size_mb = os.path.getsize(classifier_path) / (1024 * 1024)
        print(f"\n✅ Classifier-only model saved: {classifier_path}")
        print(f"   Size: {classifier_size_mb:.2f} MB")
        print(f"   Reduction: {((original_size_mb - classifier_size_mb) / original_size_mb * 100):.1f}%")
        
        # Prepare metadata
        metadata = {
            'created_at': timestamp,
            'model_uuid': model_uuid,
            'format': 'classifier-only',  # ✅ NEW FORMAT
            'requires_mobilenet': True,    # ✅ Flag for browser
            'mobilenet_config': {          # ✅ MobileNet info
                'version': 2,
                'alpha': 0.5,
                'input_resolution': 224,
                'feature_size': 1280
            },
            'num_classes': self.num_classes,
            'class_labels': self.class_labels,
            'model_architecture': 'MobileNetV2_Classifier',
            'input_shape': [224, 224, 1],
            'classifier_input_shape': [1280],  # ✅ Classifier expects features
            'training_config': {
                'epochs': config.TRAINING['epochs'],
                'batch_size': config.TRAINING['batch_size'],
                'learning_rate': config.TRAINING['learning_rate'],
                'validation_split': config.TRAINING['validation_split'],
                'early_stopping_enabled': config.TRAINING.get('early_stopping_patience') is not None,
                'early_stopping_patience': config.TRAINING.get('early_stopping_patience'),
            },
            'has_default_classes': any(
                label in ['UNKNOWN', 'NON-SIGNATURE'] 
                for label in self.class_labels
            )
        }
        
        # Save metadata
        metadata_path = os.path.join(save_dir, f"metadata_{timestamp}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        result = {
            'keras_model_path': full_model_path,
            'classifier_model_path': classifier_path,  # ✅ NEW
            'metadata_path': metadata_path,
            'model_uuid': model_uuid,
            'original_size_mb': original_size_mb,
            'classifier_size_mb': classifier_size_mb  # ✅ NEW
        }
        
        # ✅ Step 5: Convert CLASSIFIER ONLY to TF.js (if enabled)
        if config.TFJS['enabled']:
            try:
                print("\n" + "="*60)
                print("🔄 CONVERTING CLASSIFIER TO TENSORFLOW.JS")
                print("="*60)
                
                converter = TFJSConverter(config.STORAGE['tfjs_dir'])
                
                # ✅ CRITICAL: Convert classifier_only, NOT full model
                tfjs_dir = converter.convert_model(
                    classifier_path,  # ✅ Use classifier, not full model
                    metadata,
                    quantization=config.TFJS['quantization'],
                    validate=config.TFJS.get('validate_conversion', True)
                )
                
                # Verify the conversion
                model_json_path = Path(tfjs_dir) / 'model.json'
                with open(model_json_path, 'r') as f:
                    model_json = json.load(f)
                
                # Check that it's the classifier (should have ~3-5 layers, not 160+)
                num_layers = len(
                    model_json['modelTopology']['model_config']['config']['layers']
                )
                print(f"\n🔍 Converted model has {num_layers} layers")
                
                if num_layers > 10:
                    raise ValueError(
                        f"❌ ERROR: Converted model has {num_layers} layers! "
                        f"Expected ~3-5 layers for classifier only. "
                        f"Full MobileNet was likely included by mistake."
                    )
                
                print(f"✅ Verification passed: Classifier has {num_layers} layers")
                
                tfjs_files = converter.get_model_files(tfjs_dir)
                result['tfjs_model_path'] = tfjs_dir
                result['tfjs_files'] = tfjs_files
                
                # Calculate TF.js size
                tfjs_size_mb = sum(
                    os.path.getsize(w) for w in tfjs_files['weights']
                ) / (1024 * 1024)
                tfjs_size_mb += os.path.getsize(tfjs_files['model_json']) / (1024 * 1024)
                result['tfjs_size_mb'] = tfjs_size_mb
                result['size_reduction_percent'] = (
                    (original_size_mb - tfjs_size_mb) / original_size_mb
                ) * 100
                
                print(f"\n✅ TF.js model created: {tfjs_dir}")
                print(f"   Original full model: {original_size_mb:.2f} MB")
                print(f"   Classifier only: {classifier_size_mb:.2f} MB")
                print(f"   TF.js (quantized): {tfjs_size_mb:.2f} MB")
                print(f"   Total reduction: {result['size_reduction_percent']:.1f}%")
                
                # ✅ Step 6: Upload to S3 (if enabled)
                if config.S3['enabled'] and config.TFJS['auto_upload']:
                    print("\n" + "="*60)
                    print("📤 UPLOADING TO S3")
                    print("="*60)
                    
                    uploader = S3Uploader(
                        aws_access_key=config.S3['access_key'],
                        aws_secret_key=config.S3['secret_key'],
                        bucket_name=config.S3['bucket'],
                        region=config.S3['region']
                    )
                    
                    s3_result = uploader.upload_tfjs_model(
                        tfjs_dir,
                        model_uuid,
                        tfjs_files
                    )
                    
                    result['s3_info'] = s3_result
                    result['model_url'] = s3_result['model_url']
                    
                    print(f"✅ Model uploaded to S3")
                    print(f"   URL: {result['model_url']}")
                
            except Exception as e:
                print(f"⚠️ TF.js conversion/upload failed: {e}")
                print("   Keras models still saved successfully")
                import traceback
                traceback.print_exc()
        
        print("\n" + "="*60)
        print("✅ MODEL SAVE COMPLETE")
        print("="*60)
        print(f"📦 Full Keras Model: {full_model_path}")
        print(f"   Size: {result['original_size_mb']:.2f} MB (float32)")
        print(f"🔧 Classifier Only: {classifier_path}")
        print(f"   Size: {result['classifier_size_mb']:.2f} MB")
        
        if 'tfjs_model_path' in result:
            print(f"🌐 TF.js Model: {result['tfjs_model_path']}")
            print(f"   Size: {result['tfjs_size_mb']:.2f} MB ({config.TFJS['quantization']})")
            print(f"   Reduction: {result['size_reduction_percent']:.1f}%")
        
        if 'model_url' in result:
            print(f"☁️ S3 URL: {result['model_url']}")
        
        print(f"🆔 Model UUID: {model_uuid}")
        print("="*60 + "\n")
        
        return result
