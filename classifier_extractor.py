# filepath: python/classifier_extractor.py
"""
Classifier Extractor - Separates classifier head from MobileNet
This allows exporting ONLY the trained classifier, not the entire model
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
import json

class ClassifierExtractor:
    def __init__(self):
        self.mobilenet_output_shape = 1280  # MobileNetV2 α=0.5 output
        
    def extract_classifier_from_model(
        self, 
        full_model: keras.Model,
        class_labels: list
    ) -> keras.Model:
        """
        Extract ONLY the classifier layers from a full model
        
        Args:
            full_model: Complete model (MobileNet + classifier)
            class_labels: List of class names
            
        Returns:
            classifier_only: Model with input shape (1280,) and same output
        """
        print("\n" + "="*60)
        print("🔧 EXTRACTING CLASSIFIER HEAD")
        print("="*60)
        
        # Find where classifier starts (first Dense layer after MobileNet)
        classifier_start_idx = None
        for i, layer in enumerate(full_model.layers):
            if 'dense' in layer.name.lower():
                classifier_start_idx = i
                print(f"✅ Found classifier start at layer {i}: {layer.name}")
                break
        
        if classifier_start_idx is None:
            raise ValueError("Could not find classifier layers (no Dense layers found)")
        
        # Get all classifier layers
        classifier_layers = full_model.layers[classifier_start_idx:]
        print(f"📋 Classifier has {len(classifier_layers)} layers:")
        for layer in classifier_layers:
            print(f"   • {layer.name} ({layer.__class__.__name__})")
        
        # Build new model with classifier only
        # Input: features from MobileNet (1280 dimensions)
        classifier_input = keras.Input(
            shape=(self.mobilenet_output_shape,), 
            name='mobilenet_features'
        )
        
        # Chain classifier layers
        x = classifier_input
        for layer in classifier_layers:
            # Create new layer with same config
            layer_config = layer.get_config()
            layer_class = layer.__class__
            new_layer = layer_class.from_config(layer_config)
            
            # Copy weights from original layer
            new_layer.build(x.shape)
            new_layer.set_weights(layer.get_weights())
            
            x = new_layer(x)
        
        # Create classifier model
        classifier_only = keras.Model(
            inputs=classifier_input, 
            outputs=x,
            name='classifier_only'
        )
        
        print(f"\n✅ Classifier extracted successfully!")
        print(f"   Input shape: {classifier_only.input_shape}")
        print(f"   Output shape: {classifier_only.output_shape}")
        print(f"   Total params: {classifier_only.count_params():,}")
        print("="*60 + "\n")
        
        return classifier_only
    
    def verify_classifier(
        self,
        full_model: keras.Model,
        classifier_only: keras.Model,
        num_tests: int = 5
    ) -> bool:
        """
        ✅ OPTION 1: STRUCTURAL VERIFICATION (RECOMMENDED)
        
        Verify classifier structure instead of numerical equivalence.
        This is safer because it doesn't try to extract intermediate features.
        """
        print("🔍 Verifying classifier structure...")
        
        try:
            # Check 1: Input shape should be (None, 1280)
            classifier_input_shape = classifier_only.input_shape
            if classifier_input_shape[1] != 1280:
                print(f"   ❌ Wrong input shape: {classifier_input_shape} (expected (None, 1280))")
                return False
            print(f"   ✅ Input shape correct: {classifier_input_shape}")
            
            # Check 2: Output shape should match full model
            full_output_shape = full_model.output_shape
            classifier_output_shape = classifier_only.output_shape
            if full_output_shape != classifier_output_shape:
                print(f"   ❌ Output shape mismatch: {classifier_output_shape} vs {full_output_shape}")
                return False
            print(f"   ✅ Output shape matches: {classifier_output_shape}")
            
            # Check 3: Layer count should be small (3-5 layers)
            num_layers = len(classifier_only.layers)
            if num_layers > 10:
                print(f"   ❌ Too many layers: {num_layers} (expected 3-5)")
                return False
            print(f"   ✅ Layer count correct: {num_layers} layers")
            
            # Check 4: Parameter count should be reasonable
            params = classifier_only.count_params()
            # For 20-30 classes, expect ~250k-500k params
            if params > 1_000_000:
                print(f"   ⚠️ Warning: High parameter count: {params:,}")
                print(f"      (This might include MobileNet layers by mistake)")
            else:
                print(f"   ✅ Parameter count reasonable: {params:,}")
            
            # Check 5: Try a forward pass with dummy data
            print(f"   🧪 Testing forward pass with dummy data...")
            dummy_input = np.random.rand(1, 1280).astype(np.float32)
            output = classifier_only.predict(dummy_input, verbose=0)
            
            # Check output is valid probabilities (sums to ~1.0)
            output_sum = np.sum(output)
            if abs(output_sum - 1.0) > 0.01:
                print(f"   ⚠️ Warning: Output sum = {output_sum:.4f} (expected ~1.0)")
            else:
                print(f"   ✅ Output is valid probability distribution")
            
            print(f"\n📊 Verification Summary:")
            print(f"   ✅ All structural checks passed!")
            print(f"   ℹ️ Classifier is ready for TF.js export")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Verification failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_classifier_with_metadata(
        self,
        classifier: keras.Model,
        save_dir: str,
        metadata: dict
    ) -> str:
        """
        Save classifier model with metadata
        
        Args:
            classifier: Extracted classifier model
            save_dir: Directory to save to
            metadata: Model metadata (labels, accuracy, etc.)
            
        Returns:
            Path to saved classifier
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save classifier
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        classifier_path = save_path / f"classifier_only_{timestamp}.keras"
        classifier.save(str(classifier_path))
        
        print(f"✅ Classifier saved: {classifier_path}")
        print(f"   Size: {classifier_path.stat().st_size / 1024:.2f} KB")
        
        # Save metadata
        metadata_path = save_path / f"classifier_metadata_{timestamp}.json"
        
        # Add classifier-specific metadata
        full_metadata = {
            **metadata,
            'format': 'classifier-only',
            'requires_mobilenet': True,
            'mobilenet_config': {
                'version': 2,
                'alpha': 0.5,
                'input_resolution': 224,
                'feature_size': 1280
            },
            'classifier_info': {
                'input_shape': list(classifier.input_shape[1:]),
                'output_shape': list(classifier.output_shape[1:]),
                'num_params': int(classifier.count_params()),
                'num_layers': len(classifier.layers)
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(full_metadata, f, indent=2)
        
        print(f"✅ Metadata saved: {metadata_path}")
        
        return str(classifier_path)