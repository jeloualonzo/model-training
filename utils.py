# filepath: python/utils.py
"""
Utility functions for model training and management
"""
import os
import json
import glob
from datetime import datetime
from typing import List, Dict, Optional, Tuple
def get_latest_model(models_dir: str = 'trained_models') -> Optional[str]:
    """Get path to the most recently trained model"""
    if not os.path.exists(models_dir):
        return None
    
    model_files = glob.glob(os.path.join(models_dir, 'model_*.keras'))
    if not model_files:
        return None
    
    # Sort by modification time
    model_files.sort(key=os.path.getmtime, reverse=True)
    return model_files[0]
def get_model_metadata(model_path: str) -> Optional[Dict]:
    """Get metadata for a trained model"""
    base_name = os.path.splitext(model_path)[0]
    metadata_path = base_name.replace('model_', 'metadata_') + '.json'
    
    if not os.path.exists(metadata_path):
        return None
    
    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Failed to load metadata: {e}")
        return None
def list_trained_models(models_dir: str = 'trained_models') -> List[Dict]:
    """List all trained models with metadata"""
    if not os.path.exists(models_dir):
        return []
    
    model_files = glob.glob(os.path.join(models_dir, 'model_*.keras'))
    models = []
    
    for model_path in sorted(model_files, key=os.path.getmtime, reverse=True):
        metadata = get_model_metadata(model_path)
        
        models.append({
            'model_path': model_path,
            'created_at': metadata.get('created_at') if metadata else 'Unknown',
            'num_classes': metadata.get('num_classes') if metadata else 0,
            'class_labels': metadata.get('class_labels') if metadata else [],
            'size_mb': os.path.getsize(model_path) / (1024 * 1024)
        })
    
    return models
def validate_training_data(classes: List[Dict]) -> Tuple[bool, str]:
    """
    ✅ FIXED: Validate training data with default class support
    """
    if not classes:
        return False, "No classes provided"
    
    valid_classes = [cls for cls in classes if cls.get('samples')]
    
    if len(valid_classes) < 2:
        return False, f"Need at least 2 classes with samples, got {len(valid_classes)}"
    
    for idx, cls in enumerate(valid_classes):
        # ✅ CHECK: Is this a default class?
        is_default = cls.get('isDefault', False)
        
        if is_default:
            # Default class validation
            default_name = cls.get('defaultName')
            if not default_name:
                return False, f"Class {idx} is marked as default but has no defaultName"
            
            samples = cls.get('samples', [])
            if not samples:
                return False, f"Default class '{default_name}' has no samples"
            
            print(f"✅ Default class '{default_name}' validated: {len(samples)} samples")
        else:
            # Regular student class validation
            student = cls.get('student')
            if not student:
                return False, f"Class {idx} has no student information"
            
            samples = cls.get('samples', [])
            if not samples:
                student_id = student.get('student_id', 'Unknown')
                return False, f"Class {idx} ({student_id}) has no samples"
            
            print(f"✅ Student class '{student.get('student_id')}' validated: {len(samples)} samples")
    
    return True, "Validation passed"
def format_training_summary(metrics: Dict) -> str:
    """
    ✅ ENHANCED: Format training metrics with early stopping info
    """
    early_stopped = metrics.get('early_stopped', False)
    epochs_trained = metrics.get('epochs_trained', metrics.get('epochs', 50))
    epochs_requested = metrics.get('epochs_requested', metrics.get('epochs', 50))
    best_epoch = metrics.get('best_epoch', epochs_trained)
    best_val_acc = metrics.get('best_val_accuracy', metrics['validation_accuracy'])
    best_val_loss = metrics.get('best_val_loss', metrics['validation_loss'])
    
    summary = f"""
╔════════════════════════════════════════════════════════════════════════════════════════╗
║                   TRAINING SUMMARY                                              ║
╠════════════════════════════════════════════════════════════════════════════════════════╣
║ 📊 Final Training Accuracy:    {metrics['training_accuracy']*100:6.2f}%               ║
║ 📊 Final Validation Accuracy:  {metrics['validation_accuracy']*100:6.2f}%             ║
║ 📉 Training Loss:              {metrics['training_loss']:6.4f}                        ║
║ 📉 Validation Loss:            {metrics['validation_loss']:6.4f}                      ║
║                                                                                        ║
║ 🏆 Best Epoch:                 {best_epoch:3d}                                        ║
║ 🏆 Best Validation Accuracy:   {best_val_acc*100:6.2f}%                               ║
║ 🏆 Best Validation Loss:       {best_val_loss:6.4f}                                   ║
║                                                                                        ║
║ ⏱️  Training Time:              {metrics['training_time']:6.1f}s                      ║
║ 🎯 Epochs Requested:           {epochs_requested:3d}                                  ║
║ 🎯 Epochs Trained:             {epochs_trained:3d}                                    ║"""
    
    if early_stopped:
        time_per_epoch = metrics['training_time'] / epochs_trained
        time_saved = (epochs_requested - epochs_trained) * time_per_epoch
        summary += f"""
║ 🛑 Early Stopped:              YES (saved ~{time_saved:.1f}s)                         ║"""
    else:
        summary += f"""
║ ✅ Completed All Epochs:       YES                                                    ║"""
    
    summary += f"""
║                                                                                        ║
║ 📚 Total Training Samples:     {metrics['total_samples']:5d}                          ║
║ 📚 Validation Samples:         {metrics['validation_samples']:5d}                     ║
║ 👥 Number of Classes:          {metrics['num_classes']:3d}                            ║
╚════════════════════════════════════════════════════════════════════════════════════════╝
"""
    return summary