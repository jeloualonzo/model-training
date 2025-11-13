# filepath: python/supabase_recorder.py
"""
Supabase Recorder for Trained Models
Records model metadata to Supabase database
"""
import json
from datetime import datetime
from typing import Dict, Optional, List
from supabase import create_client, Client
class SupabaseRecorder:
    def __init__(self, supabase_url: str, supabase_key: str):
        self.client: Client = create_client(supabase_url, supabase_key)
        print(f"✅ Supabase Recorder initialized")
    
    def record_training_start(
        self,
        model_uuid: str,
        sample_count: int,
        student_count: int,
        genuine_count: int
    ) -> int:
        """
        Record training start in database
        
        Returns:
            Database record ID
        """
        try:
            data = {
                'model_uuid': model_uuid,
                'status': 'training',
                'sample_count': sample_count,
                'student_count': student_count,
                'genuine_count': genuine_count,
                'training_date': datetime.now().isoformat(),
                'model_path': '',  # Will update later
                's3_key': '',      # Will update later
            }
            
            response = self.client.table('global_trained_models').insert(data).execute()
            
            if not response.data:
                raise Exception("No data returned from insert")
            
            record_id = response.data[0]['id']
            print(f"✅ Training record created: ID {record_id}")
            return record_id
            
        except Exception as e:
            print(f"❌ Failed to record training start: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def update_training_complete(
        self,
        record_id: int,
        metrics: Dict,
        s3_info: Dict[str, str],
        keras_model_path: str,
        tfjs_model_path: str,
        class_info: Optional[List[Dict]] = None  # ✅ NEW: per-class signature counts
    ):
        """
        ✅ UPDATED: Update record with training results including per-student signature counts
        Converts all NumPy types to Python native types for JSON serialization
        """
        try:
            # Helper function to convert NumPy types
            def convert_to_native(obj):
                """Recursively convert NumPy types to Python native types"""
                import numpy as np
                
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_to_native(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_native(item) for item in obj]
                else:
                    return obj
            
            # Prepare training metrics (convert all NumPy types)
            training_metrics = {
                'final_train_accuracy': float(metrics.get('training_accuracy', 0)),
                'final_val_accuracy': float(metrics.get('validation_accuracy', 0)),
                'final_train_loss': float(metrics.get('training_loss', 0)),
                'final_val_loss': float(metrics.get('validation_loss', 0)),
                'best_epoch': int(metrics.get('best_epoch', 0)),
                'best_val_accuracy': float(metrics.get('best_val_accuracy', 0)),
                'best_val_loss': float(metrics.get('best_val_loss', 0)),
                'training_time': float(metrics.get('training_time', 0)),
                'history': convert_to_native(metrics.get('history', {}))
            }
            
            # Prepare training config
            training_config = {
                'epochs_requested': int(metrics.get('epochs_requested', 0)),
                'epochs_trained': int(metrics.get('epochs_trained', 0)),
                'early_stopped': bool(metrics.get('early_stopped', False)),
                'batch_size': int(metrics.get('batch_size', 32)),
                'learning_rate': float(metrics.get('learning_rate', 0.0005)),
                'augmentation_per_sample': int(metrics.get('augmentation_per_sample', 4))
            }
            
            # ✅ NEW: Enhanced class_labels with signature counts
            class_labels_enhanced = []
            student_signature_total = 0  # ✅ NEW: Track student signatures only
            
            if class_info:
                for cls in class_info:
                    label = cls.get('label', 'Unknown')
                    signature_count = cls.get('signature_count', 0)
                    is_default = cls.get('is_default', False)
                    
                    # Store as "label|count" format for easy parsing
                    if is_default:
                        # Default classes don't need count in label
                        class_labels_enhanced.append(label)
                    else:
                        # Student classes include count
                        class_labels_enhanced.append(f"{label}|{signature_count}")
                        student_signature_total += signature_count  # ✅ Add to student total only
            else:
                # Fallback to simple labels
                class_labels_enhanced = list(metrics.get('class_labels', []))
            
            # ✅ NEW: Override sample_count with student signatures only
            final_sample_count = student_signature_total if student_signature_total > 0 else metrics.get('total_samples', 0)
            
            # ✅ Prepare data with proper type conversion
            data = {
                'status': 'completed',
                'model_path': str(keras_model_path),
                's3_key': str(s3_info.get('model_json', '')),
                's3_base_url': str(s3_info.get('model_url', '')),
                'model_format': 'keras+tfjs',
                'accuracy': float(metrics.get('validation_accuracy', 0)),
                'training_metrics': training_metrics,
                'training_config': training_config,
                'class_labels': class_labels_enhanced,  # ✅ Enhanced with signature counts
                'num_classes': int(metrics.get('num_classes', 0)),
                'model_architecture': 'MobileNetV2',
                'input_shape': [224, 224, 1],
                'has_default_classes': bool(metrics.get('has_default_classes', False)),
                'epochs_trained': int(metrics.get('epochs_trained', 0)),
                'best_epoch': int(metrics.get('best_epoch', 0)),
                'early_stopped': bool(metrics.get('early_stopped', False)),
                'sample_count': final_sample_count,  # ✅ UPDATED: Student signatures only
                'genuine_count': final_sample_count,  # ✅ UPDATED: Student signatures only
                'updated_at': datetime.now().isoformat()
            }
            
            print(f"🔄 Updating Supabase record {record_id}...")
            print(f"   Student signatures only: {student_signature_total}")
            print(f"   Total classes: {len(class_labels_enhanced)}")
            
            response = self.client.table('global_trained_models')\
                .update(data)\
                .eq('id', record_id)\
                .execute()
            
            if not response.data:
                print(f"⚠️ Update returned no data for record {record_id}")
            else:
                print(f"✅ Training record updated: ID {record_id}")
                print(f"   S3 URL: {data['s3_base_url']}")
                print(f"   Accuracy: {data['accuracy']*100:.2f}%")
                print(f"   Class labels: {len(class_labels_enhanced)} classes")
                print(f"   Sample count (students only): {final_sample_count}")
            
        except Exception as e:
            print(f"❌ Failed to update training record: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def update_training_failed(self, record_id: int, error_message: str):
        """Mark training as failed"""
        try:
            data = {
                'status': 'failed',
                'training_metrics': {'error': error_message},
                'updated_at': datetime.now().isoformat()
            }
            
            self.client.table('global_trained_models')\
                .update(data)\
                .eq('id', record_id)\
                .execute()
            
            print(f"❌ Training marked as failed: ID {record_id}")
            
        except Exception as e:
            print(f"❌ Failed to update training failure: {e}")
    
    def get_latest_model(self) -> Optional[Dict]:
        """Get latest active model"""
        try:
            response = self.client.table('global_trained_models')\
                .select('*')\
                .eq('status', 'completed')\
                .eq('is_active', True)\
                .order('created_at', desc=True)\
                .limit(1)\
                .execute()
            
            if response.data:
                return response.data[0]
            return None
            
        except Exception as e:
            print(f"❌ Failed to get latest model: {e}")
            return None
    
    def get_model_by_uuid(self, model_uuid: str) -> Optional[Dict]:
        """Get model by UUID"""
        try:
            response = self.client.table('global_trained_models')\
                .select('*')\
                .eq('model_uuid', model_uuid)\
                .execute()
            
            if response.data:
                return response.data[0]
            return None
            
        except Exception as e:
            print(f"❌ Failed to get model: {e}")
            return None
    
    def list_models(self, limit: int = 10) -> list:
        """List recent models"""
        try:
            response = self.client.table('global_trained_models')\
                .select('*')\
                .order('created_at', desc=True)\
                .limit(limit)\
                .execute()
            
            return response.data
            
        except Exception as e:
            print(f"❌ Failed to list models: {e}")
            return []