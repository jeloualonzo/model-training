# filepath: python/tfjs_converter.py
"""
TensorFlow.js Model Converter with POST-CONVERSION FIX
Fixes the model.json AFTER official converter creates it
"""
import os
import json
import subprocess
from pathlib import Path
from typing import Dict
import tensorflow as tf

class TFJSConverter:
    def __init__(self, output_dir: str = 'tfjs_models'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        print(f"✅ TF.js Converter initialized: {self.output_dir}")
    
    def convert_model(
        self,
        keras_model_path: str,
        metadata: Dict,
        quantization: str = None,
        validate: bool = True
    ) -> str:
        """
        Convert Keras model to TensorFlow.js format
        ✅ UPDATED: Validates classifier-only models
        """
        try:
            model_name = Path(keras_model_path).stem
            tfjs_dir = self.output_dir / model_name
            tfjs_dir.mkdir(exist_ok=True)
            
            print(f"\n🔄 Converting Keras model to TF.js using OFFICIAL converter...")
            print(f"   Input: {keras_model_path}")
            print(f"   Output: {tfjs_dir}")
            print(f"   Quantization: {quantization}")
            
            # ✅ Check if this is a classifier-only model
            is_classifier_only = metadata.get('format') == 'classifier-only'
            if is_classifier_only:
                print(f"   Format: CLASSIFIER ONLY (requires MobileNet in browser)")
            
            # Step 1: Load and validate model
            print(f"   📦 Loading model...")
            model = tf.keras.models.load_model(keras_model_path)
            
            # ✅ Validate classifier-only structure
            if is_classifier_only:
                num_layers = len(model.layers)
                input_shape = model.input_shape
                
                print(f"   🔍 Validating classifier structure...")
                print(f"      Layers: {num_layers}")
                print(f"      Input shape: {input_shape}")
                
                # Classifier should have:
                # - Few layers (3-5)
                # - Input shape (None, 1280) for MobileNet features
                if num_layers > 10:
                    raise ValueError(
                        f"Model has {num_layers} layers! "
                        f"Classifier-only should have ~3-5 layers. "
                        f"This might be the full model (MobileNet + classifier)."
                    )
                
                if input_shape[1] != 1280:
                    raise ValueError(
                        f"Classifier input shape is {input_shape}, expected (None, 1280). "
                        f"This doesn't match MobileNet v2 α=0.5 output."
                    )
                
                print(f"   ✅ Classifier validation passed")
            
            # Step 2: Convert to H5 format (for compatibility)
            print(f"   📦 Converting .keras → .h5 for compatibility...")
            h5_path = tfjs_dir / f"{model_name}.h5"
            model.save(str(h5_path), save_format='h5')
            print(f"   ✅ Saved as H5: {h5_path}")
            
            # Step 3: Build tensorflowjs_converter command
            cmd = ['tensorflowjs_converter']
            
            if quantization == None:
                cmd.append('   Quantization: float32 (no quantization)')
            elif quantization == 'uint8':
                cmd.append('   Quantization: {quantization}')
            
            # Step 3: Build tensorflowjs_converter command
            cmd = ['tensorflowjs_converter']
            # ✅ CORRECTED: Only add quantization flags when needed
            if quantization == 'float16':
                cmd.append('--quantize_float16')
                print(f"   Quantization: float16")
            elif quantization == 'uint8':
                cmd.append('--quantize_uint8')
                print(f"   Quantization: uint8")
            elif quantization == 'uint16':
                cmd.append('--quantize_uint16')
                print(f"   Quantization: uint16")
            else:
                print(f"   Quantization: None (float32)")

            cmd.append('--input_format=keras')
            cmd.append(str(h5_path))
            cmd.append(str(tfjs_dir))
            
            print(f"   🔧 Running converter: {' '.join(cmd)}")
            
            # Run converter
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Clean up H5
            if h5_path.exists():
                h5_path.unlink()
                print(f"   🗑️ Cleaned up temporary H5 file")
            
            # ✅ POST-CONVERSION FIX
            model_json_path = tfjs_dir / 'model.json'
            if not model_json_path.exists():
                raise FileNotFoundError(f"Conversion failed: model.json not found")
            
            print(f"\n🔧 POST-CONVERSION FIX: Fixing model.json...")
            self._fix_model_json_for_tfjs(model_json_path)
            
            # Find weight files
            weight_files = list(tfjs_dir.glob('group*.bin'))
            if not weight_files:
                weight_files = list(tfjs_dir.glob('*.bin'))
            
            if not weight_files:
                raise FileNotFoundError(f"Conversion failed: no weight files found")
            
            print(f"\n✅ Conversion complete!")
            print(f"   model.json: {model_json_path.stat().st_size / 1024:.2f} KB")
            print(f"   Weight files: {len(weight_files)}")
            
            total_size = 0
            for wf in weight_files:
                size = wf.stat().st_size / 1024 / 1024
                total_size += size
                print(f"   - {wf.name}: {size:.2f} MB")
            
            print(f"   Total TF.js size: {total_size:.2f} MB")
            
            # ✅ Save metadata with classifier flag
            metadata_path = tfjs_dir / 'metadata.json'
            metadata_dict = {
                'model_type': 'classifier' if is_classifier_only else 'full',
                'format': metadata.get('format', 'unknown'),
                'requires_mobilenet': metadata.get('requires_mobilenet', False),
                'mobilenet_config': metadata.get('mobilenet_config', {}),
                'architecture': metadata.get('model_architecture', 'MobileNetV2'),
                'num_classes': metadata.get('num_classes', 0),
                'class_labels': metadata.get('class_labels', []),
                'input_shape': metadata.get('classifier_input_shape' if is_classifier_only else 'input_shape', [224, 224, 1]),
                'has_default_classes': metadata.get('has_default_classes', False),
                'training_accuracy': metadata.get('training_accuracy'),
                'validation_accuracy': metadata.get('validation_accuracy'),
                'created_at': metadata.get('created_at'),
                'quantization': quantization,
                'training_config': metadata.get('training_config', {}),
                'converter': 'official_tensorflowjs_converter_with_post_fix',
                'converter_version': '4.22.0'
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
            
            print(f"   ✅ metadata.json saved")
            
            return str(tfjs_dir)
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Official converter failed!")
            print(f"   Return code: {e.returncode}")
            if e.stdout:
                print(f"   Stdout:\n{e.stdout}")
            if e.stderr:
                print(f"   Stderr:\n{e.stderr}")
            raise
        except Exception as e:
            print(f"❌ TF.js conversion failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _fix_model_json_for_tfjs(self, model_json_path: Path):
        """
        ✅ CRITICAL POST-FIX: Fix model.json after official converter
        Fixes InputLayer AND inbound_nodes in NESTED models too!
        """
        with open(model_json_path, 'r') as f:
            model_data = json.load(f)
        
        # Navigate to layers
        if 'modelTopology' not in model_data:
            print("   ⚠️  No modelTopology found, skipping fix")
            return
        
        topology = model_data['modelTopology']
        
        if 'model_config' not in topology or 'config' not in topology['model_config']:
            print("   ⚠️  Unexpected structure, skipping fix")
            return
        
        config = topology['model_config']['config']
        layers = config.get('layers', [])
        
        # ✅ Fix layers recursively (including nested models)
        fixed_input_layers, fixed_inbound_nodes, fixed_regularizers = self._fix_layers_recursive(layers, level=0)

        if fixed_input_layers > 0 or fixed_inbound_nodes > 0 or fixed_regularizers > 0:
            # Save fixed model.json
            with open(model_json_path, 'w') as f:
                json.dump(model_data, f, indent=2)
    
            print(f"   ✅ Fixed {fixed_input_layers} InputLayer(s) (including nested)")
            print(f"   ✅ Fixed {fixed_inbound_nodes} layer(s) with inbound_nodes")
            print(f"   ✅ Fixed {fixed_regularizers} regularizer(s)")  # ✅ NEW LINE
            print(f"   ✅ Saved fixed model.json")
        else:
            print(f"   ℹ️  No fixes needed")
    def _fix_layers_recursive(self, layers, level=0):
        """
        ✅ ENHANCED: Fix layers including regularizers for TF.js compatibility
        """
        fixed_input_layers = 0
        fixed_inbound_nodes = 0
        fixed_regularizers = 0  # ✅ NEW COUNTER
        indent = "   " * (level + 1)
        
        for layer in layers:
            layer_name = layer.get('name', 'unknown')
            
            # ============================================================
            # ✅ NEW: FIX REGULARIZERS (Python uses 'L2', TF.js uses 'l2')
            # ============================================================
            layer_config = layer.get('config', {})
            
            # Check all possible regularizer fields
            for regularizer_key in ['kernel_regularizer', 'bias_regularizer', 'activity_regularizer']:
                if regularizer_key in layer_config:
                    regularizer = layer_config[regularizer_key]
                    
                    # If regularizer exists and has a class_name
                    if regularizer and isinstance(regularizer, dict):
                        class_name = regularizer.get('class_name', '')
                        
                        # Fix uppercase regularizer names
                        if class_name == 'L2':
                            regularizer['class_name'] = 'l2'
                            fixed_regularizers += 1
                            if level == 0:
                                print(f"   🔧 Fixed L2 → l2 in {layer_name}.{regularizer_key}")
                        
                        elif class_name == 'L1':
                            regularizer['class_name'] = 'l1'
                            fixed_regularizers += 1
                            if level == 0:
                                print(f"   🔧 Fixed L1 → l1 in {layer_name}.{regularizer_key}")
                        
                        elif class_name == 'L1L2':
                            regularizer['class_name'] = 'l1_l2'
                            fixed_regularizers += 1
                            if level == 0:
                                print(f"   🔧 Fixed L1L2 → l1_l2 in {layer_name}.{regularizer_key}")
            
            # ============================================================
            # EXISTING CODE: FIX 1 - InputLayer config
            # ============================================================
            if layer.get('class_name') == 'InputLayer':
                if level > 0:
                    print(f"{indent}🔧 Fixing nested InputLayer: {layer_name}")
                else:
                    print(f"   🔧 Fixing InputLayer: {layer_name}")
                
                # Create clean config
                clean_config = {
                    'dtype': layer_config.get('dtype', 'float32'),
                    'sparse': layer_config.get('sparse', False),
                    'ragged': layer_config.get('ragged', False),
                    'name': layer_config.get('name', layer_name)
                }
                
                # ✅ CRITICAL: Convert ANY variant to batch_input_shape
                if 'batch_input_shape' in layer_config:
                    clean_config['batch_input_shape'] = layer_config['batch_input_shape']
                elif 'batchInputShape' in layer_config:
                    clean_config['batch_input_shape'] = layer_config['batchInputShape']
                elif 'batch_shape' in layer_config:
                    # ❗ THIS IS THE BUG - batch_shape should be batch_input_shape
                    clean_config['batch_input_shape'] = layer_config['batch_shape']
                    print(f"{indent}   ⚠️  Converted batch_shape → batch_input_shape")
                elif 'input_shape' in layer_config:
                    clean_config['batch_input_shape'] = [None] + list(layer_config['input_shape'])
                else:
                    # Default
                    clean_config['batch_input_shape'] = [None, 224, 224, 1]
                
                layer['config'] = clean_config
                
                if level > 0:
                    print(f"{indent}   ✅ Fixed: {clean_config['batch_input_shape']}")
                else:
                    print(f"      ✅ Fixed config keys: {list(clean_config.keys())}")
                
                fixed_input_layers += 1
            
            # ============================================================
            # EXISTING CODE: FIX 2 - inbound_nodes
            # ============================================================
            if layer.get('class_name') != 'InputLayer' and 'inbound_nodes' in layer:
                inbound_nodes = layer['inbound_nodes']
                
                if inbound_nodes and len(inbound_nodes) > 0:
                    fixed_nodes = []
                    
                    for node_group in inbound_nodes:
                        if isinstance(node_group, list):
                            fixed_group = []
                            
                            for node in node_group:
                                if isinstance(node, dict):
                                    node_name = node.get('name', 'unknown')
                                    node_index = node.get('node_index', 0)
                                    tensor_index = node.get('tensor_index', 0)
                                    fixed_group.append([node_name, node_index, tensor_index, {}])
                                elif isinstance(node, list):
                                    fixed_group.append(node)
                                else:
                                    fixed_group.append([str(node), 0, 0, {}])
                            
                            fixed_nodes.append(fixed_group)
                        else:
                            if isinstance(node_group, dict):
                                node_name = node_group.get('name', 'unknown')
                                fixed_nodes.append([[node_name, 0, 0, {}]])
                            else:
                                fixed_nodes.append([[str(node_group), 0, 0, {}]])
                    
                    if fixed_nodes != inbound_nodes:
                        layer['inbound_nodes'] = fixed_nodes
                        if level == 0:
                            print(f"   🔧 Fixed inbound_nodes for: {layer_name}")
                        fixed_inbound_nodes += 1
            
            # ============================================================
            # EXISTING CODE: RECURSIVE - Check for nested models
            # ============================================================
            if layer.get('class_name') in ['Functional', 'Model']:
                if level == 0:
                    print(f"   🔍 Checking nested model: {layer_name}")
                
                nested_config = layer.get('config', {})
                nested_layers = nested_config.get('layers', [])
                
                if nested_layers:
                    if level == 0:
                        print(f"      Found {len(nested_layers)} nested layers")
                    
                    # ✅ RECURSE into nested model
                    nested_fixed_input, nested_fixed_inbound, nested_fixed_regularizers = self._fix_layers_recursive(
                        nested_layers, 
                        level=level + 1
                    )
                    
                    fixed_input_layers += nested_fixed_input
                    fixed_inbound_nodes += nested_fixed_inbound
                    fixed_regularizers += nested_fixed_regularizers  # ✅ ADD REGULARIZER COUNT
        
        return fixed_input_layers, fixed_inbound_nodes, fixed_regularizers  # ✅ RETURN 3 VALUES
    
    def get_model_files(self, tfjs_dir: str) -> Dict[str, str]:
        """Get list of TF.js model files for upload"""
        tfjs_path = Path(tfjs_dir)
        
        files = {
            'model_json': str(tfjs_path / 'model.json'),
            'metadata': str(tfjs_path / 'metadata.json'),
            'weights': []
        }
        
        # Get weight files
        weight_files = list(tfjs_path.glob('group*.bin'))
        if not weight_files:
            weight_files = list(tfjs_path.glob('*.bin'))
        
        files['weights'] = [str(f) for f in sorted(weight_files)]
        
        # Verify all files exist
        missing_files = []
        if not Path(files['model_json']).exists():
            missing_files.append('model.json')
        if not Path(files['metadata']).exists():
            missing_files.append('metadata.json')
        for w in files['weights']:
            if not Path(w).exists():
                missing_files.append(Path(w).name)
        
        if missing_files:
            raise FileNotFoundError(f"Missing TF.js files: {missing_files}")
        
        return files
